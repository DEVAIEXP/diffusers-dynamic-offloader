"""Bounded generic-model reuse/unload gate. Explicit CUDA approval required."""
import ctypes
import json
import os
from pathlib import Path

import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadConfig, enable_offload, remove_dynamic_offload
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry


def make_model_and_source(path, seed, width):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.SiLU(),
                                torch.nn.Linear(width, width)).eval()
    slices, payload, offset = SafetensorSliceRegistry(), [], 0
    for parameter in model.parameters():
        raw = parameter.detach().view(torch.uint8).numpy().tobytes()
        assert slices.bind(parameter, SafetensorFileSlice(path, offset, len(raw)))
        payload.append(raw)
        offset += len(raw)
    path.write_bytes(b"".join(payload))
    model._ddo_safetensor_slices = slices
    return model, slices, offset


def test_model_source_fixture_is_cpu_only_and_replaces_generation(tmp_path):
    path = tmp_path / "weights.bin"
    first, slices, size = make_model_and_source(path, 43, 32)
    assert size == path.stat().st_size
    assert all(p.device.type == "cpu" for p in first.parameters())
    original = path.read_bytes()
    slices.close()
    second, other, changed = make_model_and_source(path, 44, 48)
    assert changed == path.stat().st_size and changed != size
    assert path.read_bytes() != original
    other.close()


@pytest.fixture
def runtime():
    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    path = os.environ.get("DDO_VMM_LIBRARY_PATH")
    assert path, "explicit isolated OFF lifecycle DLL required"
    dll = ctypes.CDLL(path)
    assert not hasattr(dll, "ddo_vmm_source_test_failures"), "never use fault-injection build here"
    assert hasattr(dll, "ddo_vmm_source_owned_events_v1")
    return torch.device("cuda:0")


@pytest.mark.vmm
def test_generic_model_128_forwards_with_unload_reload_and_source_replacement(runtime, tmp_path):
    path = tmp_path / "same-path-weights.bin"
    observations = []
    config = DynamicOffloadConfig(
        execution_mode="vmm_fast_runtime", vmm_resident_weight_budget_gb=0.07,
        vmm_staging_buffer_gb=2 / 1024, vmm_page_mb=2,
        # Public persistent-source geometry is 64 MiB per window. Use one
        # explicit window, not a tiny requested capacity rounded up implicitly.
        vmm_native_transport=True, vmm_source_cache_gb=64 / 1024,
        vmm_source_registered_gb=64 / 1024,
    )
    for cycle, (seed, width) in enumerate(((43, 32), (44, 48), (43, 32), (44, 48))):
        model, slices, payload_size = make_model_and_source(path, seed, width)
        value = torch.randn(2, width)
        with torch.no_grad():
            expected = model(value).detach()
        result = enable_offload(model, use_environment=False, config=config)
        try:
            before = result.metrics_snapshot().as_dict()["runtime"]["vmm"]
            assert before["fast_source_cache_process_committed_bytes"] == 0
            assert before["fast_source_cache_process_registered_bytes"] == 0
            stable = None
            device_value = value.to(runtime)
            for invocation in range(32):
                with torch.no_grad():
                    actual = model(device_value).cpu()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                report = result.metrics_snapshot(collect_completed=True).as_dict()
                stats = report["runtime"]["vmm"]
                assert report["runtime"]["completion_collection"]["tickets_after"] == 0
                assert stats["fast_source_cache_transfer_owned_events"] == 0
                assert stats["fast_source_cache_pooled_events"] <= 256
                assert stats["fast_source_cache_committed_bytes"] <= 64 * 2**20
                assert stats["fast_source_cache_registered_bytes"] <= 64 * 2**20
                assert stats["fast_cpu_source_bindings"] == 0
                assert stats["fast_file_backed_bindings"] == 4
                for name in ("fast_fault_failures", "fast_temporary_tensors", "fast_execution_plan_mismatches",
                             "fast_source_cache_admission_denials", "fast_source_cache_bypasses"):
                    assert stats[name] == 0
                key = tuple(stats[name] for name in (
                    "fast_source_cache_source_read_bytes", "fast_source_cache_committed_bytes",
                    "fast_source_cache_registered_bytes", "fast_source_cache_registrations",
                    "fast_source_cache_batch_completion_events", "fast_source_cache_pooled_events"))
                stable = key if stable is None else stable
                assert key == stable
                if invocation in (0, 31):
                    observations.append({"cycle": cycle, "invocation": invocation,
                                         "width": width, "source_payload_bytes": payload_size,
                                         "report": report})
            del device_value, actual
        finally:
            remove_dynamic_offload(model)
        assert slices.closed and not hasattr(model, "_ddo_safetensor_slices")
        assert not model._forward_pre_hooks and not model._forward_hooks
        assert all(not submodule._forward_hooks and not submodule._forward_pre_hooks for submodule in model.modules())
        # Same pathname must become writable/deletable after real DDO cleanup.
        # Next cycle changes size/content, including A->B->A, exposing stale descriptors.
        path.unlink()
        del model, result, slices, expected, value
    (tmp_path / "lifecycle_observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
