"""Legacy NVMe arena consumes the framework staging contract without CUDA."""

from contextlib import nullcontext
from types import SimpleNamespace
import threading

import pytest
import torch

from ddo_vmm_allocator.weight_arena import VmmWeightArena, WeightArenaError, _WeightGroup
from diffusers_dynamic_offloader.vmm_weight_store import (
    PinnedStagingPool,
    PinnedStagingPoolStats,
    SafetensorFileSlice,
    SafetensorSliceRegistry,
)


@pytest.mark.parametrize("truncate", [False, True])
def test_checkpoint_views_are_counted_before_upload(tmp_path, monkeypatch, truncate):
    sources = [torch.arange(6, dtype=torch.bfloat16), torch.arange(4, dtype=torch.float32)]
    payload = b"".join(source.view(torch.uint8).numpy().tobytes() for source in sources)
    shard = tmp_path / "weights.bin"
    shard.write_bytes(payload)
    registry = SafetensorSliceRegistry()
    offset = 0
    packed = []
    for index, source in enumerate(sources):
        size = source.numel() * source.element_size()
        registry.bind(source, SafetensorFileSlice(shard, offset, size))
        packed.append((str(index), source, offset))
        offset += size

    pool = object.__new__(PinnedStagingPool)
    pool.size_bytes = len(payload)
    pool.slots = 1
    pool._buffers = [torch.empty(len(payload), dtype=torch.uint8)]
    pool._virtual_buffers = []
    pool._ready_events = [None]
    pool._next_slot = 0
    pool._active_slot = None
    pool._offset = 0
    pool._windows_file_reader = None
    pool._file_lock = threading.Lock()
    pool._checkpoint_files = {}
    pool._checkpoint_maps = {}
    pool._stats = PinnedStagingPoolStats()
    if truncate:
        stage = pool.stage_file_slices
        monkeypatch.setattr(pool, "stage_file_slices", lambda entries: stage(entries)[:-1])

    recorded = []
    event = SimpleNamespace(record=lambda stream: recorded.append(stream))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    arena = VmmWeightArena(None, resident_budget_bytes=1024)
    arena._file_slices = registry
    arena._staging_pool = pool
    destination = torch.zeros(len(payload), dtype=torch.uint8)
    group = _WeightGroup("block.0", None, tensor_view=SimpleNamespace(tensor=destination),
                         packed_sources=packed, packed_payload_bytes=len(payload))
    stream = object()
    try:
        if truncate:
            with pytest.raises(WeightArenaError, match="payload mismatch"):
                arena._upload_packed_group(group, stream)
            assert not group.payload_loaded
            assert not recorded
            assert not destination.any()
            assert arena._runtime_stats["uploads"] == 0
        else:
            arena._upload_packed_group(group, stream)
            assert bytes(destination.tolist()) == payload
            assert group.payload_loaded
            assert group.upload_event is event
            assert pool._ready_events == [event]
            assert recorded == [stream]
            assert arena._runtime_stats["file_upload_bytes"] == len(payload)
            assert arena._runtime_stats["uploads"] == 1
            arena._upload_packed_group(group, stream)
            assert recorded == [stream]
    finally:
        pool.close()
        registry.close()
