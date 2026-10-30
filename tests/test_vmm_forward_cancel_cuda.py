"""Generic interrupted-forward/resume/unload gate; CUDA approval required."""
import ctypes
import json
import os

import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadConfig, enable_offload, remove_dynamic_offload
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry


class InterruptibleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.SiLU())
            for _ in range(3)
        ])
        self.interruption = None

    def forward(self, value):
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index == 0 and self.interruption is not None:
                raise self.interruption("intentional forward interruption")
        return value


def bind_source(model, path):
    registry, payload = SafetensorSliceRegistry(), bytearray()
    for parameter in model.parameters():
        raw = parameter.detach().view(torch.uint8).numpy().tobytes()
        assert registry.bind(parameter, SafetensorFileSlice(path, len(payload), len(raw)))
        payload.extend(raw)
    path.write_bytes(payload)
    model._ddo_safetensor_slices = registry
    return registry, len(payload)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_interruption_fixture_cpu_only(error_type, tmp_path):
    torch.manual_seed(43)
    model = InterruptibleModel().eval()
    value = torch.randn(2, 32)
    with torch.no_grad():
        expected = model(value)
        model.interruption = error_type
        with pytest.raises(error_type, match="intentional forward interruption"):
            model(value)
        model.interruption = None
        torch.testing.assert_close(model(value), expected)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    registry, size = bind_source(model, tmp_path / "cpu.bin")
    assert size == 12672
    registry.close()


@pytest.mark.vmm
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_interrupted_forward_resumes_then_unloads_without_readback(error_type, tmp_path):
    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    dll_path = os.environ.get("DDO_VMM_LIBRARY_PATH")
    assert dll_path, "select isolated lifecycle OFF DLL"
    dll = ctypes.CDLL(dll_path)
    assert not hasattr(dll, "ddo_vmm_source_test_failures")
    assert hasattr(dll, "ddo_vmm_source_owned_events_v1")
    torch.manual_seed(43)
    model = InterruptibleModel().eval()
    value = torch.randn(2, 32)
    with torch.no_grad():
        expected = model(value)
    path = tmp_path / "cancel.bin"
    registry, size = bind_source(model, path)
    result = enable_offload(model, use_environment=False, config=DynamicOffloadConfig(
        execution_mode="vmm_fast_runtime", vmm_native_transport=True,
        vmm_module_pattern=r"^layers\.\d+$",
        vmm_resident_weight_budget_gb=0.07, vmm_staging_buffer_gb=2 / 1024,
        vmm_page_mb=2, vmm_source_cache_gb=64 / 1024,
        vmm_source_registered_gb=64 / 1024,
    ))
    observations = []
    try:
        sample = value.cuda()
        with torch.no_grad():
            actual = model(sample).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        first = result.metrics_snapshot(collect_completed=True).as_dict()
        observations.append({"phase": "warm", "report": first})
        model.interruption = error_type
        with torch.no_grad(), pytest.raises(error_type, match="intentional forward interruption"):
            model(sample)
        model.interruption = None
        with torch.no_grad():
            actual = model(sample).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        resumed = result.metrics_snapshot(collect_completed=True).as_dict()
        observations.append({"phase": "resumed", "report": resumed})
        stats = resumed["runtime"]["vmm"]
        assert stats["fast_gpu_prefetch_cancelled_units"] >= 1
        assert stats["fast_source_cache_source_read_bytes"] == first["runtime"]["vmm"]["fast_source_cache_source_read_bytes"]
        assert stats["fast_source_cache_source_read_bytes"] == size
        assert resumed["runtime"]["completion_collection"]["tickets_after"] == 0
        assert stats["fast_source_cache_transfer_owned_events"] == 0
        assert stats["fast_source_cache_pooled_events"] <= 256
        assert stats["fast_source_cache_committed_bytes"] <= 64 * 2**20
        assert stats["fast_source_cache_registered_bytes"] <= 64 * 2**20
        assert stats["fast_cpu_source_bindings"] == 0
        assert stats["fast_file_backed_bindings"] == 6
        for key in ("fast_fault_failures", "fast_temporary_tensors",
                    "fast_source_cache_admission_denials", "fast_source_cache_bypasses"):
            assert stats[key] == 0
        # Second interruption proceeds directly to unload: no output readback
        # or metrics collection between cancellation and explicit cleanup.
        model.interruption = error_type
        with torch.no_grad(), pytest.raises(error_type, match="intentional forward interruption"):
            model(sample)
    finally:
        remove_dynamic_offload(model)
    assert registry.closed and not hasattr(model, "_ddo_safetensor_slices")
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    path.unlink()
    (tmp_path / "cancel_observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
