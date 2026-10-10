"""Convert one local Diffusers component to a separate FP8 weight-only checkpoint.

CPU-only, optional TorchAO 0.18.0. No DDO runtime or model math forward changes.
Output is incomplete until conversion_report.json has status=passed.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def tensor_digest(tensor):
    import torch

    if tensor.device.type != "cpu":
        raise ValueError("Conversion validation requires CPU tensors")
    plain = tensor.detach().contiguous()
    raw = plain.reshape(-1).view(torch.uint8).numpy()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "bytes": raw.nbytes, "sha256": hashlib.sha256(memoryview(raw)).hexdigest()}


def state_fingerprints(state, wrapper_type):
    result = {}
    for name, tensor in state.items():
        if type(tensor) is wrapper_type:
            result[name] = {"kind": "float8_weight_only", "logical_dtype": str(tensor.dtype),
                            "shape": list(tensor.shape), "qdata": tensor_digest(tensor.qdata),
                            "scale": tensor_digest(tensor.scale)}
        else:
            result[name] = {"kind": "plain", **tensor_digest(tensor)}
    return result


def _convert_bounded(source, output, model_class, config_data, max_shard_size, wrapper_type):
    """Two tensor-at-a-time passes; retain only one bounded output shard.

    Pass one collects upstream wrapper metadata and hashes. Pass two checks the
    source and conversion again and writes shards carrying complete metadata.
    Memory is bounded by the output shard plus the largest input tensor and
    quantization/serialization working buffers, not by the full component.
    """
    import re
    import torch
    from accelerate import init_empty_weights
    from diffusers import TorchAoConfig
    from diffusers.quantizers.torchao.torchao_quantizer import TorchAoHfQuantizer
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    if isinstance(max_shard_size, int) and not isinstance(max_shard_size, bool):
        limit = max_shard_size
    else:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(KB|MB|GB)", str(max_shard_size).upper())
        if not match:
            raise ValueError("Shard size must be positive integer bytes or a decimal KB/MB/GB size")
        limit = int(float(match[1]) * {"KB": 1000, "MB": 1000000, "GB": 1000000000}[match[2]])
    if limit <= 0:
        raise ValueError("Shard size must be positive")
    with init_empty_weights(include_buffers=True):
        architecture = model_class.from_config(config_data)
    expected_geometry = {name: tuple(t.shape) for name, t in architecture.state_dict().items()}
    # Tied architecture owners need a separately validated serialization recipe.
    owners = list(architecture.named_parameters(remove_duplicate=False))
    if len({id(t) for _, t in owners}) != len(owners):
        raise ValueError("Shared parameter owners require a separately validated conversion recipe")
    linear_names = {f"{name}.weight" if name else "weight"
                    for name, module in architecture.named_modules() if isinstance(module, torch.nn.Linear)}
    if not linear_names or "weight" in linear_names:
        raise ValueError("Bounded conversion requires named nn.Linear weights")
    del architecture, owners
    indices = list(source.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise ValueError("Ambiguous source checkpoint indices")
    weight_map = None
    if indices:
        weight_map = json.loads(indices[0].read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(weight_map.values()))
        if any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
            raise ValueError("Source index must reference local shard basenames")
        paths = [source / name for name in names]
    else:
        paths = sorted(source.glob("*.safetensors"))
        if len(paths) != 1:
            raise ValueError("A single source checkpoint or explicit shard index is required")
    if any(not path.is_file() or path.resolve().parent != source for path in paths):
        raise ValueError("Missing source shard or shard outside component directory")
    entries = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in entries or name not in expected_geometry:
                    raise ValueError("Unexpected or duplicate source keys")
                if tuple(handle.get_slice(name).get_shape()) != expected_geometry[name]:
                    raise ValueError("Source shape differs from component architecture")
                entries[name] = path.name
    if set(entries) != set(expected_geometry) or (weight_map is not None and entries != weight_map):
        raise ValueError("Incomplete source keys or source index differs from shard contents")
    recipe = Float8WeightOnlyConfig(set_inductor_config=False)
    quantization_config = TorchAoConfig(recipe)

    def converted(name, tensor):
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            raise ValueError("Bounded source floating tensors must already be BF16")
        if name not in linear_names:
            return tensor
        if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
            raise ValueError("Converted linear weights must be BF16 matrices")
        module = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False,
                                 device="meta", dtype=torch.bfloat16)
        module.weight = torch.nn.Parameter(tensor, requires_grad=False)
        forward = type(module).forward
        quantize_(module, recipe)
        if type(module).forward is not forward or type(module.weight) is not wrapper_type:
            raise ValueError("Conversion changed the forward or selected wrapper recipe")
        return module.weight.detach()

    metadata, expected, source_hashes = {}, {}, {}
    input_bytes = largest_input = 0
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                tensor = handle.get_tensor(name)
                source_hashes[name] = tensor_digest(tensor)
                size = tensor.numel() * tensor.element_size()
                input_bytes += size
                largest_input = max(largest_input, size)
                value = converted(name, tensor)
                expected.update(state_fingerprints({name: value}, wrapper_type))
                physical, local_metadata = TorchAoHfQuantizer.get_state_dict_and_metadata(
                    None, {name: value}, safe_serialization=True)
                for key, item in local_metadata.items():
                    if key != "tensor_names":
                        if key in metadata:
                            raise ValueError("Duplicate serialization metadata")
                        metadata[key] = item
                del tensor, value, physical, local_metadata
    metadata["tensor_names"] = json.dumps(list(expected))
    metadata["format"] = "pt"
    pending, pending_bytes, peak_pending = {}, 0, 0
    output_map, shards = {}, []

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f"diffusion_pytorch_model-{len(shards) + 1:05d}.safetensors"
        save_file(pending, str(output / filename), metadata=metadata)
        output_map.update({key: filename for key in pending})
        shards.append(output / filename)
        pending = {}
        pending_bytes = 0

    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                tensor = handle.get_tensor(name)
                if tensor_digest(tensor) != source_hashes[name]:
                    raise ValueError("Source tensor changed between conversion passes")
                value = converted(name, tensor)
                if state_fingerprints({name: value}, wrapper_type)[name] != expected[name]:
                    raise ValueError("Quantized payload changed between conversion passes")
                physical, _ = TorchAoHfQuantizer.get_state_dict_and_metadata(
                    None, {name: value}, safe_serialization=True)
                # Keep all parts of one logical tensor together. Oversized
                # individual tensors get their own shard, as in HF sharding.
                size = sum(t.numel() * t.element_size() for t in physical.values())
                if pending and pending_bytes + size > limit:
                    flush()
                if output_map.keys() & physical.keys() or pending.keys() & physical.keys():
                    raise ValueError("Duplicate serialized physical keys")
                # Clone plain mapped tensors so closing input handles cannot
                # retain complete source mappings through the output buffer.
                pending.update({key: t.clone() for key, t in physical.items()})
                pending_bytes += size
                peak_pending = max(peak_pending, pending_bytes)
                del tensor, value, physical
                if pending_bytes >= limit:
                    flush()
    flush()
    saved_config = {**config_data, "quantization_config": quantization_config.to_dict()}
    (output / "config.json").write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    total_size = sum(item["qdata"]["bytes"] + item["scale"]["bytes"]
                     if item["kind"] == "float8_weight_only" else item["bytes"]
                     for item in expected.values())
    (output / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": output_map}, indent=2),
        encoding="utf-8")
    quantizer = TorchAoHfQuantizer(quantization_config)
    quantizer.set_metadata([str(path) for path in shards])
    restored = {}
    for path in shards:
        state = load_file(str(path), device="cpu")
        reconstructed = quantizer.maybe_update_state_dict(state)
        fingerprints = state_fingerprints(reconstructed, wrapper_type)
        if restored.keys() & fingerprints.keys():
            raise ValueError("Duplicate reconstructed checkpoint keys")
        restored.update(fingerprints)
        del state, reconstructed, fingerprints
    if quantizer._pending_flattened_state_dict or restored != expected:
        raise ValueError("Saved FP8 payloads did not reconstruct exactly")
    if torch.cuda.is_initialized():
        raise RuntimeError("Conversion unexpectedly initialized CUDA")
    return {"status": "passed", "serialized_reload_exact": True,
            "conversion_mode": "bounded_tensor_two_pass", "input_parameter_bytes": input_bytes,
            "largest_input_tensor_bytes": largest_input, "peak_output_buffer_bytes": peak_pending,
            "converted_linear_weights": len(linear_names), "payloads": expected,
            "shards": [{"name": path.name, "bytes": path.stat().st_size} for path in shards],
            "scope": "checkpoint_payload_validation_not_model_quality_or_performance"}


def convert_component(source, output, *, model_class=None, max_shard_size="2GB", stream_shards=False):
    """Save and reconstruct all serialized payloads; never overwrite a checkpoint.

    Source is a local component directory containing config.json, not an entire
    pipeline or a remote repository. By default one full CPU model is required;
    stream_shards instead constructs only a meta architecture and reads one
    source tensor at a time. Validation loads one output shard at a time.
    """
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("Source must be a local Diffusers component with config.json")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must be separate and non-nested")
    if output.exists():
        raise FileExistsError("Output already exists; conversion never overwrites it")
    config_data = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config_data.get("quantization_config"):
        raise ValueError("Source must not already have a quantization configuration")
    if importlib.metadata.version("torchao") != "0.18.0":
        raise ValueError("This experimental conversion recipe requires torchao==0.18.0")

    import torch
    import diffusers
    from diffusers import TorchAoConfig
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.quantizers.torchao.torchao_quantizer import TorchAoHfQuantizer
    from safetensors.torch import load_file
    from torchao.quantization import Float8WeightOnlyConfig, quantize_
    from ddo_vmm_allocator.quantized_payload import _supported_backend

    if torch.cuda.is_initialized():
        raise RuntimeError("Run conversion in a fresh CPU-only process")
    wrapper_type = _supported_backend()
    if model_class is None:
        class_name = config_data.get("_class_name")
        model_class = getattr(diffusers, class_name, None) if isinstance(class_name, str) else None
    if not isinstance(model_class, type) or not issubclass(model_class, ModelMixin):
        raise ValueError("A supported Diffusers ModelMixin class is required; no remote code is loaded")

    # Exclusive creation reserves the target and preserves failed diagnostics.
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running", "source": str(source),
              "output": str(output), "model_class": model_class.__name__,
              "recipe": "Float8WeightOnlyConfig(version=2, per-row E4M3, logical BF16)",
              "dependencies": {name: importlib.metadata.version(name)
                               for name in ("torch", "torchao", "diffusers", "safetensors")},
              "max_shard_size": max_shard_size, "cuda_initialized": False}
    try:
        if stream_shards:
            report.update(_convert_bounded(source, output, model_class, config_data,
                                           max_shard_size, wrapper_type))
            return report
        model = model_class.from_pretrained(str(source), local_files_only=True,
                                           torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model.eval().requires_grad_(False)
        if getattr(model, "hf_quantizer", None) is not None or getattr(model, "is_quantized", False):
            raise ValueError("Source loader returned a quantized model")
        if any(tensor.device.type != "cpu" for tensor in model.state_dict().values()):
            raise ValueError("Source model must be fully materialized on CPU")
        linears = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
        if not linears:
            raise ValueError("No nn.Linear weights to convert")
        pointers = [module.weight.untyped_storage().data_ptr() for module in linears]
        if len(pointers) != len(set(pointers)):
            raise ValueError("Shared linear weight storage requires a separately validated conversion recipe")
        if any(module.weight.dtype != torch.bfloat16 for module in linears):
            raise ValueError("All converted linear weights must be BF16")
        report["input_parameter_bytes"] = sum(t.numel() * t.element_size() for t in model.parameters())
        forwards = {type(module): type(module).forward for module in model.modules()}
        recipe = Float8WeightOnlyConfig(set_inductor_config=False)
        quantize_(model, recipe)
        if any(type(module).forward is not forwards[type(module)] for module in model.modules()):
            raise ValueError("Conversion changed a model class forward")
        if any(type(module.weight) is not wrapper_type for module in linears):
            raise ValueError("A linear weight did not use the selected FP8 recipe")
        expected = state_fingerprints(model.state_dict(), wrapper_type)
        report["converted_linear_weights"] = len(linears)
        report["payloads"] = expected
        quantization_config = TorchAoConfig(recipe)
        model.hf_quantizer = TorchAoHfQuantizer(quantization_config)
        model.hf_quantizer.validate_environment(torch_dtype=torch.bfloat16, device_map=None)
        model.register_to_config(quantization_config=quantization_config.to_dict())
        model.save_pretrained(str(output), safe_serialization=True, max_shard_size=max_shard_size)
        del model, linears
        gc.collect()

        shards = sorted(output.glob("*.safetensors"))
        if not shards:
            raise ValueError("No serialized checkpoint shards were written")
        quantizer = TorchAoHfQuantizer(quantization_config)
        quantizer.set_metadata([str(path) for path in shards])
        restored = {}
        for path in shards:
            state = load_file(str(path), device="cpu")
            reconstructed = quantizer.maybe_update_state_dict(state)
            fingerprints = state_fingerprints(reconstructed, wrapper_type)
            if restored.keys() & fingerprints.keys():
                raise ValueError("Duplicate reconstructed checkpoint keys")
            restored.update(fingerprints)
            del state, reconstructed, fingerprints
        if quantizer._pending_flattened_state_dict or restored != expected:
            raise ValueError("Saved FP8 data, scales or untouched tensors did not reconstruct exactly")
        if torch.cuda.is_initialized():
            raise RuntimeError("Conversion unexpectedly initialized CUDA")
        report.update(status="passed", serialized_reload_exact=True,
                      shards=[{"name": path.name, "bytes": path.stat().st_size} for path in shards],
                      scope="checkpoint_payload_validation_not_model_quality_or_performance")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local Diffusers component directory")
    parser.add_argument("--output", type=Path, required=True, help="New, non-existing component directory")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--stream-shards", action="store_true",
                        help="Read one source tensor at a time; avoid full CPU model loading")
    args = parser.parse_args()
    # This standalone utility is CPU-only; it never installs the native allocator.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    report = convert_component(args.source, args.output, max_shard_size=args.max_shard_size,
                               stream_shards=args.stream_shards)
    print(json.dumps({key: value for key, value in report.items() if key != "payloads"}, indent=2))


if __name__ == "__main__":
    main()
