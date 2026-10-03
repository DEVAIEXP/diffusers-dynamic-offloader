"""CUDA contract tests for DDO's single-arena ``vmm_fast_runtime`` route."""

import os
from pathlib import Path

import pytest
import torch

from diffusers_dynamic_offloader import (
    DynamicOffloadConfig,
    apply_dynamic_offload,
    enable_offload,
    remove_dynamic_offload,
)
from diffusers_dynamic_offloader.profiles import load_dynamic_offload_profile
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry


def _vmm_library_path() -> Path:
    return Path(os.environ["DDO_VMM_LIBRARY_PATH"])


@pytest.mark.vmm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vmm_fast_uses_file_descriptors_without_retaining_cpu_weights(tmp_path: Path) -> None:
    torch.manual_seed(43)
    model = torch.nn.Linear(8, 8).eval()
    reference = torch.nn.Linear(8, 8).eval()
    reference.load_state_dict(model.state_dict())
    value = torch.randn(2, 8)
    expected = reference(value)
    weight = model.weight.detach()
    bias = model.bias.detach()
    weight_bytes = weight.numel() * weight.element_size()
    shard = tmp_path / "weights.bin"
    shard.write_bytes(weight.view(torch.uint8).numpy().tobytes() + bias.view(torch.uint8).numpy().tobytes())
    slices = SafetensorSliceRegistry()
    assert slices.bind(weight, SafetensorFileSlice(shard, 0, weight_bytes))
    assert slices.bind(bias, SafetensorFileSlice(shard, weight_bytes, bias.numel() * bias.element_size()))
    model._ddo_safetensor_slices = slices
    model._ddo_safetensor_slice_summary = {"bound": 2, "available": 2}

    hook = apply_dynamic_offload(
        model,
        DynamicOffloadConfig(
            execution_mode="vmm_fast_runtime",
            vmm_resident_weight_budget_gb=0.07,
            vmm_staging_buffer_gb=0.003,
        ),
    )
    try:
        actual = model(value.cuda()).cpu()
        torch.cuda.synchronize()
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
        stats = hook.vmm_runtime_stats()
        assert stats is not None
        assert stats["fast_file_backed_bindings"] == 2
        assert stats["fast_cpu_source_bindings"] == 0
        assert stats["fast_file_uploads"] == 2
        assert stats["fast_shard_read_bytes"] == weight_bytes + bias.numel() * bias.element_size()
        assert stats["fast_dma_enqueue_bytes"] == weight_bytes + bias.numel() * bias.element_size()
        assert stats["fast_compute_wait_events"] == 2
        assert hook._vmm_staging_pool is not None
        # A short forward need not visit every slot in the two-slot ring.
        assert any(event is not None for event in hook._vmm_staging_pool._ready_events)
    finally:
        remove_dynamic_offload(model)

    assert slices.closed
    assert not hasattr(model, "_ddo_safetensor_slices")


@pytest.mark.vmm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vmm_fast_public_route_uses_bounded_native_source_cache(tmp_path: Path) -> None:
    """DDO attaches the cache without a second PyTorch staging allocation."""
    torch.manual_seed(43)
    model = torch.nn.Linear(8, 8).eval()
    reference = torch.nn.Linear(8, 8).eval()
    reference.load_state_dict(model.state_dict())
    value = torch.randn(2, 8)
    expected = reference(value)
    weight = model.weight.detach()
    bias = model.bias.detach()
    weight_bytes = weight.numel() * weight.element_size()
    shard = tmp_path / "weights.bin"
    shard.write_bytes(weight.view(torch.uint8).numpy().tobytes() + bias.view(torch.uint8).numpy().tobytes())
    slices = SafetensorSliceRegistry()
    assert slices.bind(weight, SafetensorFileSlice(shard, 0, weight_bytes))
    assert slices.bind(bias, SafetensorFileSlice(shard, weight_bytes, bias.numel() * bias.element_size()))
    model._ddo_safetensor_slices = slices

    hook = apply_dynamic_offload(
        model,
        DynamicOffloadConfig(
            execution_mode="vmm_fast_runtime",
            vmm_resident_weight_budget_gb=0.07,
            vmm_staging_buffer_gb=2 / 1024,
            vmm_native_source_cache_slot_mb=2,
            vmm_native_source_cache_slots=1,
        ),
    )
    try:
        assert hook._vmm_staging_pool is None
        actual = model(value.cuda()).cpu()
        torch.cuda.synchronize()
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
        stats = hook.vmm_runtime_stats()
        assert stats is not None
        assert stats["fast_file_backed_bindings"] == 2
        assert stats["fast_cpu_source_bindings"] == 0
        assert stats["fast_source_cache_source_read_bytes"] == weight_bytes
        assert stats["fast_source_cache_registered_bytes"] == 2 * 1024**2
        assert stats["fast_source_cache_process_staging_registered_bytes"] == 2 * 1024**2
        assert stats["fast_source_cache_process_limit_bytes"] == 4 * 1024**2
    finally:
        remove_dynamic_offload(model)

    assert slices.closed


@pytest.mark.vmm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vmm_fast_uses_event_safe_temporary_fallback_when_native_fault_is_denied(tmp_path: Path) -> None:
    from ddo_vmm_allocator.native import load_native_vmm

    torch.manual_seed(43)
    model = torch.nn.Linear(8, 8).eval()
    reference = torch.nn.Linear(8, 8).eval()
    reference.load_state_dict(model.state_dict())
    value = torch.randn(2, 8)
    expected = reference(value)
    weight = model.weight.detach()
    bias = model.bias.detach()
    weight_bytes = weight.numel() * weight.element_size()
    shard = tmp_path / "weights.bin"
    shard.write_bytes(weight.view(torch.uint8).numpy().tobytes() + bias.view(torch.uint8).numpy().tobytes())
    slices = SafetensorSliceRegistry()
    assert slices.bind(weight, SafetensorFileSlice(shard, 0, weight_bytes))
    assert slices.bind(bias, SafetensorFileSlice(shard, weight_bytes, bias.numel() * bias.element_size()))
    model._ddo_safetensor_slices = slices
    model._ddo_safetensor_slice_summary = {"bound": 2, "available": 2}
    native = load_native_vmm(_vmm_library_path())
    hook = apply_dynamic_offload(
        model,
        DynamicOffloadConfig(
            execution_mode="vmm_fast_runtime",
            vmm_resident_weight_budget_gb=0.07,
            vmm_staging_buffer_gb=0.003,
        ),
    )
    try:
        assert Path(hook.state.planner_decisions["vmm_native_library"]).resolve() == _vmm_library_path().resolve()
        native.set_dxgi_budget_guard(device_ordinal=0, system_reserve_bytes=1 << 60)
        actual = model(value.cuda()).cpu()
        torch.cuda.synchronize()
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
        stats = hook.vmm_runtime_stats()
        assert stats is not None
        assert stats["fast_fault_failures"] == 1
        assert stats["fast_temporary_tensors"] == 2
        assert stats["fast_temporary_bytes"] == weight_bytes + bias.numel() * bias.element_size()
    finally:
        native.set_dxgi_budget_guard(device_ordinal=0)
        remove_dynamic_offload(model)


@pytest.mark.vmm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vmm_fast_profile_saves_native_capacity_without_recalibrating_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DDO_PROFILE_DIR", str(tmp_path / "profiles"))
    model = torch.nn.Linear(8, 8).eval()
    reference = torch.nn.Linear(8, 8).eval()
    reference.load_state_dict(model.state_dict())
    value = torch.randn(2, 8)
    expected = reference(value)
    result = enable_offload(
        model,
        use_environment=False,
        config=DynamicOffloadConfig(
            execution_mode="vmm_fast_runtime",
            vmm_resident_weight_budget_gb=0.07,
            vmm_staging_buffer_gb=0.003,
            profile_name="native-capacity",
            build_profile=True,
        ),
    )
    try:
        with result.profile_run() as report:
            actual = model(value.cuda()).cpu()
            torch.cuda.synchronize()
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
        assert report["status"] == "saved"
        assert report["vmm"]["virtual_bytes"] > 0
        assert result.settings.config.vmm_resident_weight_budget_gb == 0.07
        saved = load_dynamic_offload_profile(result._profile.context)
        assert saved is not None
        assert saved["observations"][-1]["profile_measurement_kind"] == "vmm_capacity"
    finally:
        remove_dynamic_offload(model)
