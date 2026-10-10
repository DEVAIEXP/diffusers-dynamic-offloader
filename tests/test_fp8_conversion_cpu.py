"""Small real Diffusers serialization tests for the reusable conversion tool."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torchao")
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

spec = importlib.util.spec_from_file_location(
    "ddo_fp8_conversion_tool", Path(__file__).resolve().parents[1] / "scripts/convert_fp8_weight_only.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class TinyComponent(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width=16):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(width, width, dtype=torch.bfloat16)
                                          for _ in range(2)])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


@pytest.mark.parametrize("max_shard_size", ["2GB", 256])
def test_saved_conversion_reconstructs_real_single_and_split_shards(tmp_path, max_shard_size):
    assert not torch.cuda.is_initialized()
    source, output = tmp_path / "bf16", tmp_path / "fp8"
    TinyComponent().save_pretrained(source)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        report = tool.convert_component(source, output, model_class=TinyComponent,
                                        max_shard_size=max_shard_size)
    assert report["status"] == "passed" and report["serialized_reload_exact"]
    assert report["converted_linear_weights"] == 2
    assert report["cuda_initialized"] is False
    assert len(report["shards"]) > 1 if max_shard_size == 256 else len(report["shards"]) == 1
    saved_config = json.loads((output / "config.json").read_text())
    assert saved_config["quantization_config"]["quant_method"] == "torchao"
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    with pytest.raises(FileExistsError):
        tool.convert_component(source, output, model_class=TinyComponent)
    assert not torch.cuda.is_initialized()


def test_nested_output_refused_before_creation(tmp_path):
    source = tmp_path / "bf16"
    TinyComponent().save_pretrained(source)
    with pytest.raises(ValueError, match="non-nested"):
        tool.convert_component(source, source / "fp8", model_class=TinyComponent)
    assert not (source / "fp8").exists()


class MixedComponent(TinyComponent):
    @register_to_config
    def __init__(self, width=16):
        super().__init__(width)
        self.embedding = torch.nn.Embedding(4, width, dtype=torch.bfloat16)
        self.norm = torch.nn.LayerNorm(width, dtype=torch.bfloat16)
        self.register_buffer("root_counter", torch.tensor(3, dtype=torch.int64))


@pytest.mark.parametrize("source_shard_size", ["2GB", 256])
@pytest.mark.parametrize("output_shard_size", ["2GB", 256])
def test_bounded_conversion_matches_full_converter_and_public_loader(
        tmp_path, source_shard_size, output_shard_size):
    from diffusers_dynamic_offloader.float8_loading import load_float8_component_cpu

    source, bounded, full = tmp_path / "bf16", tmp_path / "bounded", tmp_path / "full"
    model = MixedComponent().eval().requires_grad_(False)
    original = tool.state_fingerprints(model.state_dict(), object)
    model.save_pretrained(source, max_shard_size=source_shard_size)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    reference = tool.convert_component(source, full, model_class=MixedComponent,
                                       max_shard_size=output_shard_size)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
            patch.object(MixedComponent, "from_pretrained", side_effect=AssertionError("Full loading forbidden")):
        report = tool.convert_component(source, bounded, model_class=MixedComponent,
                                        max_shard_size=output_shard_size, stream_shards=True)
        restored = load_float8_component_cpu(bounded, MixedComponent)
    assert report["status"] == "passed" and report["serialized_reload_exact"]
    assert report["conversion_mode"] == "bounded_tensor_two_pass"
    assert report["payloads"] == reference["payloads"]
    assert report["payloads"]["embedding.weight"] == original["embedding.weight"]
    assert report["payloads"]["norm.weight"] == original["norm.weight"]
    assert report["payloads"]["root_counter"] == original["root_counter"]
    if output_shard_size == 256:
        largest_payload = max(item["qdata"]["bytes"] + item["scale"]["bytes"]
                              if item["kind"] == "float8_weight_only" else item["bytes"]
                              for item in report["payloads"].values())
        assert report["peak_output_buffer_bytes"] <= max(256, largest_payload)
    assert tool.state_fingerprints(restored.state_dict(), type(restored.layers[0].weight)) == report["payloads"]
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    assert not torch.cuda.is_initialized()


def test_bounded_conversion_refuses_incomplete_index_and_preserves_failure(tmp_path):
    source, output = tmp_path / "bf16", tmp_path / "failed"
    TinyComponent().save_pretrained(source, max_shard_size=256)
    index_path = source / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["layers.0.weight"] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="local shard basenames"):
        tool.convert_component(source, output, model_class=TinyComponent, stream_shards=True)
    assert json.loads((output / "conversion_report.json").read_text())["status"] == "failed"


def test_loader_accepts_recycled_temporary_flatten_addresses_between_shards(tmp_path):
    from diffusers.quantizers.torchao.torchao_quantizer import TorchAoHfQuantizer
    from diffusers_dynamic_offloader.float8_loading import load_float8_component_cpu

    source, output = tmp_path / "bf16", tmp_path / "fp8"
    TinyComponent().save_pretrained(source)
    report = tool.convert_component(source, output, model_class=TinyComponent,
                                    max_shard_size=256, stream_shards=True)
    original = TorchAoHfQuantizer.get_state_dict_and_metadata
    scratch = {}
    reuse_count = 0

    def recycling_flatten(self, state, safe_serialization=False):
        nonlocal reuse_count
        physical, metadata = original(self, state, safe_serialization=safe_serialization)
        geometries = [(t.dtype, tuple(t.shape)) for t in physical.values()]
        assert len(geometries) == len(set(geometries))
        for key, tensor in list(physical.items()):
            geometry = (tensor.dtype, tuple(tensor.shape))
            if geometry in scratch:
                reuse_count += 1
            else:
                scratch[geometry] = torch.empty_like(tensor)
            scratch[geometry].copy_(tensor)
            physical[key] = scratch[geometry]
        return physical, metadata

    with patch.object(TorchAoHfQuantizer, "get_state_dict_and_metadata", recycling_flatten), \
            patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        restored = load_float8_component_cpu(output, TinyComponent)
    assert reuse_count > 0
    assert tool.state_fingerprints(restored.state_dict(), type(restored.layers[0].weight)) == report["payloads"]
    assert restored._ddo_safetensor_slices.descriptor_count == 6
    for layer in restored.layers:
        assert restored._ddo_safetensor_slices.lookup(layer.weight.qdata) is not None
        assert restored._ddo_safetensor_slices.lookup(layer.weight.scale) is not None


def test_bounded_conversion_refuses_source_changes_between_passes(tmp_path):
    source, output = tmp_path / "bf16", tmp_path / "failed"
    TinyComponent().save_pretrained(source)
    original_digest = tool.tensor_digest
    calls = 0

    def changed_digest(tensor):
        nonlocal calls
        calls += 1
        value = original_digest(tensor)
        # Four source tensors, plus two payload digests per converted weight
        # and one per bias, precede the second pass's first source digest.
        if calls == 11:
            return {**value, "sha256": "changed"}
        return value

    with patch.object(tool, "tensor_digest", side_effect=changed_digest):
        with pytest.raises(ValueError, match="Source tensor changed"):
            tool.convert_component(source, output, model_class=TinyComponent, stream_shards=True)
    assert json.loads((output / "conversion_report.json").read_text())["status"] == "failed"


def test_failed_conversion_preserves_diagnostics(tmp_path):
    source, output = tmp_path / "bf16", tmp_path / "failed"
    TinyComponent().save_pretrained(source)
    with patch.object(TinyComponent, "from_pretrained", side_effect=RuntimeError("fixture load failure")):
        with pytest.raises(RuntimeError, match="fixture load failure"):
            tool.convert_component(source, output, model_class=TinyComponent)
    report = json.loads((output / "conversion_report.json").read_text())
    assert report["status"] == "failed" and "fixture load failure" in report["error"]


@pytest.mark.parametrize("max_shard_size", ["2GB", 256])
def test_explicit_ddo_loader_reconstructs_and_binds_physical_sources(tmp_path, max_shard_size):
    from diffusers_dynamic_offloader import DynamicOffloadConfig, from_pretrained_with_dynamic_offload
    from ddo_vmm_allocator.quantized_arena_adapter import discover_float8_component_layout

    source, output = tmp_path / "bf16", tmp_path / "fp8"
    TinyComponent().save_pretrained(source)
    report = tool.convert_component(source, output, model_class=TinyComponent, max_shard_size=max_shard_size)
    config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime", vmm_native_transport=True,
                                  vmm_component_tensor_ownership=True, vmm_float8_weight_only=True,
                                  vmm_source_cache_gb=0.01)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        loaded = from_pretrained_with_dynamic_offload(output, model_loader=TinyComponent,
                                                     dynamic_offload_config=config, local_files_only=True)
        model = loaded.module
        wrapper_type = type(model.layers[0].weight)
        assert tool.state_fingerprints(model.state_dict(), wrapper_type) == report["payloads"]
        layout, composites = discover_float8_component_layout(model, model._ddo_safetensor_slices)
        assert len(composites) == 2 and model._ddo_safetensor_slices.descriptor_count == 6
        assert sum(group.nbytes for group, _, _ in layout.parameters) == 704
        assert layout.payload_bytes == 1312  # Native parameter alignment includes padding.
        capacity = loaded.inspect_vmm_component(component="fp8_cpu").as_dict()["setup"]["component_capacity"]
        assert capacity["unique_parameter_bytes"] == 704 and capacity["execution_units"] == 2
        assert torch.isfinite(model(torch.randn(4, 16, dtype=torch.bfloat16))).all()
        from diffusers_dynamic_offloader.dynamic_offload import build_dynamic_offload_plan

        assert build_dynamic_offload_plan(model, config).module_count > 0
        with pytest.raises(ValueError, match="device maps"):
            from_pretrained_with_dynamic_offload(output, model_loader=TinyComponent,
                                                 dynamic_offload_config=config, device_map="cpu")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("change", [{"execution_mode": "linear_runtime"},
                                     {"execution_mode": "vmm_runtime"},
                                     {"vmm_native_transport": False},
                                     {"vmm_component_tensor_ownership": False},
                                     {"vmm_execution_auxiliary_ownership": True},
                                     {"vmm_source_cache_gb": 0}])
def test_fp8_configuration_refuses_unvalidated_routes(change):
    from dataclasses import replace
    from diffusers_dynamic_offloader import DynamicOffloadConfig
    from diffusers_dynamic_offloader.float8_loading import validate_float8_native_config

    config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime", vmm_native_transport=True,
                                  vmm_component_tensor_ownership=True, vmm_float8_weight_only=True,
                                  vmm_source_cache_gb=0.01)
    with pytest.raises(ValueError, match="persistent source cache"):
        validate_float8_native_config(replace(config, **change))


def test_opt_in_native_runtime_dispatch_uses_composite_adapter_with_cpu_doubles(tmp_path):
    from types import SimpleNamespace
    from diffusers_dynamic_offloader import DynamicOffloadConfig, from_pretrained_with_dynamic_offload
    from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadHook

    source, output = tmp_path / "bf16", tmp_path / "fp8"
    TinyComponent().save_pretrained(source)
    tool.convert_component(source, output, model_class=TinyComponent)
    config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime", vmm_native_transport=True,
                                  vmm_component_tensor_ownership=True, vmm_float8_weight_only=True,
                                  vmm_source_cache_gb=64 / 1024, vmm_source_registered_gb=64 / 1024,
                                  vmm_staging_buffer_gb=4 / 1024, vmm_resident_weight_budget_gb=6 / 1024)
    model = from_pretrained_with_dynamic_offload(output, model_loader=TinyComponent,
                                                 dynamic_offload_config=config).module
    adapter = SimpleNamespace(component_report={"schema_version": 1})
    report = SimpleNamespace(adapter=adapter, attached_modules=2, attached_tensors=6,
                             payload_bytes=1312, virtual_bytes=2**21, page_bytes=2**21)
    store = SimpleNamespace()
    hook = DynamicOffloadHook(config)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
            patch("diffusers_dynamic_offloader.vmm_model_arena._load_native_allocator",
                  return_value=(SimpleNamespace(path=Path("cpu-double.dll")), None)), \
            patch("ddo_vmm_allocator.quantized_arena_adapter.attach_float8_component_parameters",
                  return_value=(report, store)) as attach, \
            patch("ddo_vmm_allocator.model_arena_adapter.attach_model_linear_modules",
                  side_effect=AssertionError("Dense attachment forbidden")):
        hook._prepare_vmm_fast_runtime(model)
    assert attach.call_count == 1 and hook._vmm_fast_adapter is adapter and hook._vmm_fast_store is store
    assert "module_pattern" not in attach.call_args.kwargs
    assert "execution_auxiliary_ownership" not in attach.call_args.kwargs
    assert attach.call_args.kwargs["file_slices"] is model._ddo_safetensor_slices
    assert hook.state.planner_decisions["vmm_float8_weight_only"] is True
    assert not torch.cuda.is_initialized()
