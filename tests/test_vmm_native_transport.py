"""Small public-route controls for native transport with/without retention."""

import json
import os
from pathlib import Path

import pytest
import torch
from vmm_test_helpers import primary_context_for_probe

from diffusers_dynamic_offloader import DynamicOffloadConfig, enable_offload, remove_dynamic_offload
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry


@pytest.mark.vmm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cache_enabled", [False, True], ids=["ring-only", "ring-and-cache"])
def test_public_gpu_lookahead_prepares_next_unit_and_cancels_skipped_units(tmp_path, cache_enabled):
    """Full GPU prefetch, numerical parity and branch cancellation at fixed small capacity."""

    class DenseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])

        def forward(self, value, skip_middle=False):
            for index, layer in enumerate(self.transformer_blocks):
                if index != 1 or not skip_middle:
                    value = layer(value)
            return value

    torch.manual_seed(43)
    model = DenseModel().eval()
    value = torch.randn(2, 8)
    expected = {skip: model(value, skip).detach() for skip in (False, True)}
    slices = SafetensorSliceRegistry()
    shard = tmp_path / "lookahead.bin"
    payload = bytearray()
    for parameter in model.parameters():
        source = parameter.detach()
        size = source.numel() * source.element_size()
        assert slices.bind(source, SafetensorFileSlice(shard, len(payload), size))
        payload.extend(source.view(torch.uint8).numpy().tobytes())
    shard.write_bytes(payload)
    model._ddo_safetensor_slices = slices
    result = enable_offload(
        model,
        use_environment=False,
        config=DynamicOffloadConfig(
            execution_mode="vmm_fast_runtime",
            vmm_resident_weight_budget_gb=0.07,
            vmm_staging_buffer_gb=2 / 1024,
            vmm_native_transport=True,
            vmm_native_source_cache_slot_mb=2 if cache_enabled else 0,
            vmm_native_source_cache_slots=1 if cache_enabled else 0,
        ),
    )
    observations = []

    def observe_before_consumer(module, args):
        observations.append(result.metrics_snapshot().as_dict()["runtime"]["vmm"])

    observer = model.transformer_blocks[1].register_forward_pre_hook(observe_before_consumer, prepend=True)
    try:
        sample = value.cuda()
        for skip in (False, False, True, False):
            actual = model(sample, skip).cpu()
            torch.testing.assert_close(actual, expected[skip], rtol=1e-4, atol=1e-5)
        assert observations[0]["fast_gpu_prefetch_units"] >= 1
        # Both next-unit weights were uploaded before its consuming hook.
        assert observations[0]["fast_file_uploads"] >= 4
        stats = result.metrics_snapshot().as_dict()["runtime"]["vmm"]
        assert stats["fast_gpu_prefetch_hits"] >= 6
        assert stats["fast_gpu_prefetch_cancelled_units"] >= 1
        assert stats["fast_fault_failures"] == 0
        assert stats["fast_temporary_tensors"] == 0
    finally:
        observer.remove()
        remove_dynamic_offload(model)
    assert slices.closed


@pytest.mark.vmm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cache_enabled", [False, True], ids=["ring-only", "ring-and-cache"])
def test_public_native_transport_repeated_units_and_source_aliases(tmp_path: Path, cache_enabled: bool) -> None:
    """Keep the ring fixed; one cache slot must retain an alias and bypass misses.

    Source aliases share a file range; this does not assert shared GPU storage.
    No forced GPU eviction or full-model performance claim is made by this gate.
    """
    torch.manual_seed(43)
    model = torch.nn.Sequential(*(torch.nn.Linear(8, 8) for _ in range(3))).eval()
    model[2].weight = model[0].weight
    value = torch.randn(2, 8)
    expected = model(value).detach()
    shard = tmp_path / "weights.bin"
    slices = SafetensorSliceRegistry()
    payload = bytearray()
    ranges = {}
    for layer in model:
        for parameter in (layer.weight, layer.bias):
            source = parameter.detach()
            size = source.numel() * source.element_size()
            key = (source.data_ptr(), size)
            if key not in ranges:
                ranges[key] = SafetensorFileSlice(shard, len(payload), size)
                payload.extend(source.view(torch.uint8).numpy().tobytes())
            assert slices.bind(source, ranges[key])
    shard.write_bytes(payload)
    assert slices.descriptor_count == 5  # Six bindings, one shared source range.
    model._ddo_safetensor_slices = slices

    result = enable_offload(
        model,
        use_environment=False,
        config=DynamicOffloadConfig(
            execution_mode="vmm_fast_runtime",
            vmm_resident_weight_budget_gb=0.07,
            vmm_staging_buffer_gb=2 / 1024,
            vmm_native_transport=True,
            vmm_native_source_cache_slot_mb=2 if cache_enabled else 0,
            vmm_native_source_cache_slots=1 if cache_enabled else 0,
        ),
    )
    try:
        first = None
        for _ in range(3):
            actual = model(value).cpu()
            torch.cuda.synchronize()
            assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
            report = result.metrics_snapshot().as_dict()
            assert json.loads(json.dumps(report))["schema_version"] == 1
            stats = report["runtime"]["vmm"]
            assert stats["fast_file_backed_bindings"] == 6
            assert stats["fast_cpu_source_bindings"] == 0
            assert stats["fast_fault_failures"] == 0
            assert stats["fast_temporary_tensors"] == 0
            if first is None:
                first = stats
            else:
                assert stats["fast_file_uploads"] == first["fast_file_uploads"]
                assert stats["fast_signature_hits"] > first["fast_signature_hits"]
        assert result.hook._vmm_staging_pool is None  # No duplicate PyTorch ring.
        decisions = report["runtime"]["state"]["planner_decisions"]
        assert decisions["vmm_native_transport_registered_capacity_bytes"] == 2 * 1024**2
        if cache_enabled:
            assert stats["fast_source_cache_hits"] >= 1
            assert stats["fast_source_cache_bypasses"] >= 1
            assert stats["fast_source_cache_source_read_bytes"] == 256
            assert stats["fast_source_cache_registered_bytes"] == 2 * 1024**2
            assert stats["fast_source_cache_process_staging_registered_bytes"] == 2 * 1024**2
            assert stats["fast_source_cache_process_limit_bytes"] == 4 * 1024**2
        else:
            assert not any(key.startswith("fast_source_cache_") for key in stats)
    finally:
        remove_dynamic_offload(model)

    assert slices.closed
    assert not hasattr(model, "_ddo_safetensor_slices")
    from ddo_vmm_allocator.native import load_native_vmm

    native = load_native_vmm(Path(os.environ["DDO_VMM_LIBRARY_PATH"]))
    # Closing an arena can detach its retained context from this thread.
    # Cached PyTorch allocations do not necessarily activate it again.
    from cuda.bindings import driver

    with primary_context_for_probe(driver), native.create_source_cache(slot_bytes=2 * 1024**2, slots=1) as probe:
        snapshot = probe.snapshot()
        assert snapshot["process_registered_bytes"] == 0
        assert snapshot["process_staging_registered_bytes"] == 0
