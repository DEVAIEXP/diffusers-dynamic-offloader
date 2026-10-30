import gc
import threading
import weakref
from pathlib import Path

import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadConfig, DynamicOffloadHook
from diffusers_dynamic_offloader.vmm_weight_store import (
    PinnedStagingPool,
    PinnedStagingPoolStats,
    SafetensorFileSlice,
    SafetensorSliceRegistry,
)
from diffusers_dynamic_offloader.windows_file_reader import WindowsPinnedFileReader


def test_slice_registry_retains_only_immutable_descriptor_metadata():
    registry = SafetensorSliceRegistry()
    tensor = torch.empty(16, dtype=torch.float32)
    tensor_reference = weakref.ref(tensor)
    descriptor = SafetensorFileSlice(Path("weights-00001.safetensors"), offset_bytes=128, size_bytes=64)

    assert registry.bind(tensor, descriptor)
    assert registry.lookup(tensor) == descriptor
    assert registry.descriptor_count == 1
    assert registry.descriptor_bytes == 64

    del tensor
    gc.collect()
    assert tensor_reference() is None


def test_slice_registry_close_releases_descriptors_and_rejects_reuse():
    registry = SafetensorSliceRegistry()
    tensor = torch.empty(4, dtype=torch.float32)
    registry.bind(tensor, SafetensorFileSlice(Path("weights.safetensors"), offset_bytes=0, size_bytes=16))

    registry.close()

    assert registry.closed
    assert registry.descriptor_count == 0
    assert registry.descriptor_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        registry.lookup(tensor)


def test_vmm_detach_closes_and_removes_module_slice_metadata():
    module = torch.nn.Linear(2, 2, bias=False)
    registry = SafetensorSliceRegistry()
    registry.bind(module.weight.data, SafetensorFileSlice(Path("weights.safetensors"), offset_bytes=0, size_bytes=16))
    module._ddo_safetensor_slices = registry
    module._ddo_safetensor_slice_summary = {"bound": 1, "available": 1}
    hook = DynamicOffloadHook(DynamicOffloadConfig())
    hook._vmm_runtime_active = True
    hook._vmm_file_slices = registry

    hook._close_vmm_store(module)

    assert registry.closed
    assert not hasattr(module, "_ddo_safetensor_slices")
    assert not hasattr(module, "_ddo_safetensor_slice_summary")


def test_staging_transport_reports_mapped_shard_read(tmp_path: Path):
    pool = object.__new__(PinnedStagingPool)
    pool._windows_file_reader = None
    pool._file_lock = threading.Lock()
    pool._checkpoint_files = {}
    pool._checkpoint_maps = {}
    pool._virtual_buffers = []
    pool._buffers = []
    pool._stats = PinnedStagingPoolStats()
    payload = bytes(range(16))
    shard = tmp_path / "weights.bin"
    shard.write_bytes(payload)
    destination = torch.empty(len(payload), dtype=torch.uint8)

    PinnedStagingPool._read_file_slice(pool, SafetensorFileSlice(shard, 0, len(payload)), destination)

    assert bytes(destination.tolist()) == payload
    assert pool.transport_stats.shard_read_bytes == len(payload)
    assert pool.transport_stats.shard_read_seconds >= 0
    pool.close()


def test_staging_transport_reports_wait_only_when_reusing_a_busy_slot():
    class _Event:
        def __init__(self):
            self.calls = 0

        def synchronize(self):
            self.calls += 1

    pool = object.__new__(PinnedStagingPool)
    event = _Event()
    pool._ready_events = [event]
    pool.slots = 1
    pool._next_slot = 0
    pool._active_slot = None
    pool._offset = 0
    pool._stats = PinnedStagingPoolStats()

    pool.begin_upload()

    assert event.calls == 1
    assert pool.transport_stats.slot_reuse_waits == 1
    assert pool.transport_stats.slot_reuse_wait_seconds >= 0


def test_staging_prefetch_checks_next_slot_without_waiting():
    class _Event:
        def __init__(self, ready: bool) -> None:
            self.ready = ready
            self.queries = 0

        def query(self) -> bool:
            self.queries += 1
            return self.ready

    pool = object.__new__(PinnedStagingPool)
    event = _Event(ready=False)
    pool._ready_events = [event]
    pool._next_slot = 0
    pool._active_slot = None

    assert not pool.next_slot_ready
    assert event.queries == 1
    event.ready = True
    assert pool.next_slot_ready


def test_windows_reader_async_request_delegates_without_waiting(tmp_path: Path):
    class _Requests:
        def __init__(self) -> None:
            self.call = None

        def submit(self, function, *args):
            self.call = (function, args)
            return "future"

    reader = object.__new__(WindowsPinnedFileReader)
    reader._requests = requests = _Requests()
    reader.read_at = lambda *_args: None

    assert reader.read_at_async(tmp_path / "weights.bin", 12, 34, 56) == "future"
    assert requests.call is not None
    assert requests.call[1] == (tmp_path / "weights.bin", 12, 34, 56)
