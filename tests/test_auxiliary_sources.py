"""CPU contracts for the explicitly selected residual-loader ownership boundary."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadApplyResult
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
         patch.object(torch.cuda, "synchronize", side_effect=AssertionError("wait forbidden")):
        yield


def result_for(module):
    registry = SafetensorSliceRegistry()
    for name, tensor in module.named_parameters():
        if tensor.device.type == "cpu" and tensor.is_contiguous():
            assert registry.bind(tensor, SafetensorFileSlice(
                Path("shard.safetensors"), 0, tensor.numel() * tensor.element_size(), tuple(tensor.shape),
                "BF16" if tensor.dtype == torch.bfloat16 else "F32"))
    module.eval()
    hook = SimpleNamespace(execution_mode="vmm_fast_runtime", _vmm_runtime_active=True,
                           _vmm_file_slices=registry,
                           _vmm_fast_store=SimpleNamespace(runtime_stats=lambda: {
                               "cpu_source_bindings": 0, "faults": 0, "execution_plan_passes": 0}),
                           config=SimpleNamespace(vmm_component_tensor_ownership=False))
    return DynamicOffloadApplyResult(module=module, hook=hook)


def test_disabled_never_requires_native_hook():
    report = DynamicOffloadApplyResult(torch.nn.Linear(2, 3)).detach_auxiliary_cpu_sources()
    assert report.setup["auxiliary_source_detachment"]["status"] == "disabled"
    assert "copied_bytes=0" in report.format_profile()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_values_identity_device_alias_and_source_metadata_preserved(dtype):
    model = torch.nn.Module()
    one = torch.nn.Parameter(torch.arange(6, dtype=dtype).reshape(2, 3))
    two = torch.nn.Parameter(one.detach())
    model.register_parameter("one", one)
    model.register_parameter("same_object", one)
    model.register_parameter("same_bytes", two)
    model.register_buffer("buffer_alias", one.detach())
    result = result_for(model)
    old_pointer = one.data_ptr()
    old = one.detach()
    report = result.detach_auxiliary_cpu_sources(enabled=True)
    observation = report.setup["auxiliary_source_detachment"]
    assert observation["copied_bytes"] == 6 * one.element_size()
    assert observation["copied_storage_groups"] == 1
    assert observation["copied_named_references"] == 4
    assert model.one is one and model.same_object is one and model.same_bytes is two
    assert one.requires_grad and one.device.type == "cpu" and one.dtype == dtype
    assert one.data_ptr() != old_pointer
    assert one.data_ptr() == two.data_ptr() == model.buffer_alias.data_ptr()
    torch.testing.assert_close(one, old, rtol=0, atol=0)
    assert result.hook._vmm_file_slices.lookup(old) is None
    assert result.hook._vmm_file_slices.lookup(one).path == Path("shard.safetensors")
    pointer = one.data_ptr()
    assert result.detach_auxiliary_cpu_sources(enabled=True) is report
    assert one.data_ptr() == pointer
    json.dumps(report.as_dict())


def test_limit_refuses_before_copy_or_mutation():
    model = torch.nn.Linear(2, 3)
    result = result_for(model)
    old = model.weight.data_ptr()
    with patch.object(torch.Tensor, "clone", side_effect=AssertionError("copy forbidden")):
        with pytest.raises(ValueError, match="above the explicit"):
            result.detach_auxiliary_cpu_sources(enabled=True, max_copy_bytes=1)
    assert model.weight.data_ptr() == old


def test_partial_overlap_refused_before_mutation():
    base = torch.ones(12)
    model = torch.nn.Module()
    model.register_parameter("one", torch.nn.Parameter(base[:8]))
    model.register_parameter("overlap", torch.nn.Parameter(base[4:]))
    result = result_for(model)
    old = model.one.data_ptr()
    with pytest.raises(ValueError, match="Overlapping"):
        result.detach_auxiliary_cpu_sources(enabled=True)
    assert model.one.data_ptr() == old


def test_noncontiguous_source_refused():
    model = torch.nn.Linear(2, 3)
    result = result_for(model)
    descriptor = result.hook._vmm_file_slices.lookup(model.weight)
    old = model.weight.data_ptr()
    model.weight.data = model.weight.detach().t()
    # Keep the original address/size association to exercise layout refusal.
    assert result.hook._vmm_file_slices.lookup(model.weight) == descriptor
    with pytest.raises(ValueError, match="ownership/layout"):
        result.detach_auxiliary_cpu_sources(enabled=True)
    assert model.weight.data_ptr() == old


def test_registry_failure_rolls_back_all_parameters():
    model = torch.nn.Linear(2, 3)
    result = result_for(model)
    old = {name: (tensor.data_ptr(), tensor.detach().clone()) for name, tensor in model.named_parameters()}
    with patch.object(result.hook._vmm_file_slices, "rebind_many", side_effect=RuntimeError("relocation failure")):
        with pytest.raises(RuntimeError, match="relocation failure"):
            result.detach_auxiliary_cpu_sources(enabled=True)
    for name, tensor in model.named_parameters():
        assert tensor.data_ptr() == old[name][0]
        torch.testing.assert_close(tensor, old[name][1], rtol=0, atol=0)
        assert result.hook._vmm_file_slices.lookup(tensor) is not None
    assert not hasattr(result.hook, "_auxiliary_source_detachment_report")


@pytest.mark.parametrize("field,value", [("execution_mode", "linear_runtime"),
                                        ("execution_mode", "vmm_runtime"),
                                        ("_vmm_runtime_active", False)])
def test_other_runtime_or_inactive_refused(field, value):
    result = result_for(torch.nn.Linear(2, 3))
    setattr(result.hook, field, value)
    with pytest.raises(ValueError, match="attached vmm_fast"):
        result.detach_auxiliary_cpu_sources(enabled=True)


@pytest.mark.parametrize("counter,value", [("cpu_source_bindings", 1), ("faults", 1),
                                          ("execution_plan_passes", 1), ("faults", None)])
def test_live_or_unavailable_native_ownership_refused(counter, value):
    result = result_for(torch.nn.Linear(2, 3))
    stats = {"cpu_source_bindings": 0, "faults": 0, "execution_plan_passes": 0}
    stats[counter] = value
    result.hook._vmm_fast_store.runtime_stats = lambda: stats
    with pytest.raises(ValueError, match="no started execution"):
        result.detach_auxiliary_cpu_sources(enabled=True)


def test_unbound_tensor_unchanged_and_training_refused():
    model = torch.nn.Linear(2, 3)
    result = result_for(model)
    model.register_buffer("independent", torch.ones(20))
    pointer = model.independent.data_ptr()
    result.detach_auxiliary_cpu_sources(enabled=True)
    assert model.independent.data_ptr() == pointer
    other = result_for(torch.nn.Linear(2, 3))
    other.module.train()
    with pytest.raises(ValueError, match="eval component"):
        other.detach_auxiliary_cpu_sources(enabled=True)


def test_registry_rebind_is_atomic_when_new_shape_invalid():
    result = result_for(torch.nn.Linear(2, 3))
    registry = result.hook._vmm_file_slices
    old = result.module.weight.detach()
    original = registry.lookup(old)
    with pytest.raises(ValueError, match="conflicts"):
        registry.rebind_many([(old, torch.ones(99))])
    assert registry.lookup(old) == original
