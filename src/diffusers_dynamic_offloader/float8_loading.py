"""Explicit local TorchAO checkpoint loading for the experimental native route.

No CUDA placement, conversion, math-forward replacement or checkpoint key guesses.
Physical descriptor names come from upstream serialization of reconstructed tensors.
"""

import json
import math
from pathlib import Path


def validate_float8_native_config(config):
    if (config.execution_mode != "vmm_fast_runtime" or not config.vmm_native_transport
            or not config.vmm_component_tensor_ownership or config.vmm_execution_auxiliary_ownership
            or config.vmm_module_pattern != ".*" or not math.isfinite(config.vmm_source_cache_gb)
            or config.vmm_source_cache_gb <= 0):
        raise ValueError("FP8 weight-only requires native fast VMM, persistent source cache and full component ownership")


def load_float8_component_cpu(directory, model_class):
    """Reconstruct saved backend wrappers and bind their physical source ranges."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import TorchAoConfig
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.quantizers.torchao.torchao_quantizer import TorchAoHfQuantizer
    from safetensors.torch import load_file
    from torchao.quantization import Float8WeightOnlyConfig
    from ddo_vmm_allocator.quantized_payload import _supported_backend, describe_float8_weight_only_cpu
    from .dynamic_offload import _safetensor_file_slices
    from .vmm_weight_store import SafetensorSliceRegistry

    wrapper_type = _supported_backend()
    directory = Path(directory).resolve(strict=True)
    if not isinstance(model_class, type) or not issubclass(model_class, ModelMixin):
        raise ValueError("FP8 local loading requires an explicit Diffusers ModelMixin class")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    recipe_config = TorchAoConfig(Float8WeightOnlyConfig(set_inductor_config=False))
    if config.get("quantization_config") != recipe_config.to_dict():
        raise ValueError("Checkpoint does not declare the supported FP8 weight-only recipe")
    indices = list(directory.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise ValueError("Ambiguous checkpoint indices")
    if indices:
        weight_map = json.loads(indices[0].read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(weight_map.values()))
        if any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
            raise ValueError("Checkpoint index must reference local shard basenames")
        paths = [directory / name for name in names]
    else:
        paths = sorted(directory.glob("*.safetensors"))
        if len(paths) != 1:
            raise ValueError("A single checkpoint or explicit shard index is required")
    if any(not path.is_file() or path.resolve().parent != directory for path in paths):
        raise ValueError("Missing shard or shard outside checkpoint directory")
    diagnostics = {}
    available = _safetensor_file_slices(paths, diagnostics=diagnostics)
    if not available or any(diagnostics.values()):
        raise ValueError("Invalid or ambiguous physical checkpoint metadata")
    if indices and (set(weight_map) != set(available) or any(
            available[key].path.name != shard for key, shard in weight_map.items())):
        raise ValueError("Shard index differs from physical checkpoint keys")
    quantizer = TorchAoHfQuantizer(recipe_config)
    quantizer.set_metadata([str(path) for path in paths])
    if not quantizer.metadata:
        raise ValueError("Missing upstream TorchAO wrapper metadata")
    model_config = {key: value for key, value in config.items() if key != "quantization_config"}
    with init_empty_weights():
        model = model_class.from_config(model_config)
    expected = set(model.state_dict())
    seen, physical_seen = set(), set()
    wrapper_count = 0
    for path in paths:
        raw_state = load_file(str(path), device="cpu")
        state = quantizer.maybe_update_state_dict(raw_state)
        if set(state) - expected or seen & set(state):
            raise ValueError("Unexpected or duplicate logical checkpoint keys")
        for tensor in state.values():
            if type(tensor) is wrapper_type:
                describe_float8_weight_only_cpu(tensor)
                wrapper_count += 1
        physical, _ = TorchAoHfQuantizer.get_state_dict_and_metadata(None, state, safe_serialization=True)
        # Upstream flatten clones wrapper payloads. These temporary owners
        # live only through this shard; later shards may reuse their addresses.
        # Validate geometry locally, while physical_seen enforces global keys.
        # Only final published model owners belong in the persistent registry.
        shard_registry = SafetensorSliceRegistry()
        for key, tensor in physical.items():
            descriptor = available.get(key)
            if key in physical_seen or descriptor is None or not shard_registry.bind(tensor, descriptor):
                raise ValueError("Physical payload has no unique matching immutable source")
        physical_seen.update(physical)
        model.load_state_dict(state, strict=False, assign=True)
        seen.update(state)
        del raw_state, state, physical
    if quantizer._pending_flattened_state_dict or seen != expected or physical_seen != set(available):
        raise ValueError("Incomplete logical weights or physical payload sources")
    if not wrapper_count or any(t.device.type != "cpu" for t in model.state_dict().values()):
        raise ValueError("Fully materialized CPU FP8 component required")
    model.eval().requires_grad_(False)
    # load_state_dict/Parameter publication may detach backend payloads into
    # different CPU storage. Bind the final owners, not intermediate loader
    # tensors. Serialized names still come exclusively from upstream flatten.
    registry = SafetensorSliceRegistry()
    owners = dict(model.named_parameters())
    owners.update(dict(model.named_buffers()))
    for name in expected:
        tensor = owners[name]
        serialized, _ = TorchAoHfQuantizer.get_state_dict_and_metadata(
            None, {name: tensor.detach()}, safe_serialization=True)
        payloads = [tensor.qdata, tensor.scale] if type(tensor) is wrapper_type else [tensor]
        for payload in payloads:
            matches = [(key, value) for key, value in serialized.items()
                       if value.dtype == payload.dtype and value.shape == payload.shape]
            if len(matches) != 1:
                raise ValueError("Ambiguous upstream physical payload mapping")
            key, value = matches[0]
            if not torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
                               payload.contiguous().reshape(-1).view(torch.uint8)):
                raise ValueError("Serialized physical payload differs from final owner")
            if not registry.bind(payload, available[key]):
                raise ValueError("Final physical owner differs from immutable source geometry")
    model.register_to_config(quantization_config=config["quantization_config"])
    model.hf_quantizer = quantizer
    model._ddo_safetensor_slices = registry
    model._ddo_safetensor_slice_summary = {"bound": registry.descriptor_count,
                                         "available": len(available), "resolver": "torchao_physical_payloads",
                                         "header": diagnostics}
    model._ddo_float8_weight_only_loaded = True
    return model
