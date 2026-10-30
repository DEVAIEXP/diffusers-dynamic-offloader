"""Small CUDA gate for the public persistent-source execution-buffer route."""

import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadConfig, enable_offload, remove_dynamic_offload
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry


@pytest.mark.vmm
@pytest.mark.parametrize("unit_name", ["transformer_blocks", "layers"])
@pytest.mark.parametrize("block_depth", [1, 2])
def test_public_native_buffer_reuses_backing_without_tail_remaps(tmp_path, unit_name, block_depth):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    class DenseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.add_module(unit_name, torch.nn.ModuleList(
                [torch.nn.Linear(512, 512, bias=False) if block_depth == 1 else
                 torch.nn.Sequential(*[torch.nn.Linear(512, 512, bias=False)
                                       for _ in range(block_depth)]) for _ in range(8)]
            ))

        def forward(self, value, skip_middle=False):
            for index, layer in enumerate(getattr(self, unit_name)):
                if not (skip_middle and index == 4):
                    value = layer(value)
            return value

    torch.manual_seed(43)
    model = DenseModel().eval()
    value = torch.randn(2, 512)
    with torch.no_grad():
        expected = {skip: model(value, skip) for skip in (False, True)}
    shard = tmp_path / "weights.bin"
    slices = SafetensorSliceRegistry()
    payload = bytearray()
    for parameter in model.parameters():
        tensor = parameter.detach()
        size = tensor.numel() * tensor.element_size()
        assert slices.bind(tensor, SafetensorFileSlice(shard, len(payload), size))
        payload.extend(tensor.view(torch.uint8).numpy().tobytes())
    shard.write_bytes(payload)
    model._ddo_safetensor_slices = slices
    result = enable_offload(
        model, use_environment=False,
        config=DynamicOffloadConfig(
            execution_mode="vmm_fast_runtime", vmm_native_transport=True,
            vmm_module_pattern=rf"^{unit_name}\.\d+$",
            # Two 2-MiB-aligned slots plus one admitted 2-MiB page on Windows.
            # This is a small-contract capacity, not a production budget change.
            vmm_resident_weight_budget_gb=6 / 1024, vmm_page_mb=1,
            vmm_staging_buffer_gb=2 / 1024, vmm_source_cache_gb=64 / 1024,
        ),
    )
    try:
        sample = value.cuda()
        first = None
        with torch.no_grad():
            for skip in (False, True, False):
                actual = model(sample, skip).cpu()
                torch.testing.assert_close(actual, expected[skip], rtol=1e-4, atol=1e-5)
                stats = result.metrics_snapshot().as_dict()["runtime"]["vmm"]
                assert stats["fast_execution_buffer_uploads"] > 0
                # vmm_page_mb is a request, rounded to driver granularity.
                page = stats["fast_page_bytes"]
                expected_capacity = ((block_depth * 1024**2 + page - 1) // page) * page
                assert stats["fast_view_prepare_calls"] == 8
                assert stats["fast_view_release_calls"] == 8
                assert stats["fast_view_create_calls"] == 8 * block_depth
                assert stats["fast_execution_buffer_slots"] == 2
                assert stats["fast_execution_buffer_slot_bytes"] == expected_capacity
                assert stats["fast_execution_buffer_capacity_bytes"] == 2 * expected_capacity
                assert stats["fast_execution_buffer_resident_bytes"] == 2 * expected_capacity
                assert stats["fast_total_weight_resident_bytes"] <= 6 * 1024**2
                assert stats["fast_arena_tail_evictions"] == 0
                assert stats["fast_temporary_tensors"] == 0
                if first is not None:
                    assert stats["fast_signature_hits"] > first["fast_signature_hits"]
                    assert stats["fast_source_cache_source_read_bytes"] == first["fast_source_cache_source_read_bytes"]
                first = stats
        assert stats["fast_gpu_prefetch_hits"] > 0
        assert stats["fast_gpu_prefetch_cancelled_units"] > 0
        assert stats["fast_source_cache_batch_submissions"] > 0
        assert stats["fast_source_cache_events_reused"] > 0
    finally:
        remove_dynamic_offload(model)
    assert slices.closed
