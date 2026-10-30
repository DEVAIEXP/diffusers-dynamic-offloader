"""CPU-only source preparation contracts; no CUDA context or model execution."""
import ctypes
import os
from threading import RLock
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from ddo_vmm_allocator.source_cache import NativeSourceBatch, NativeSourceCache
from ddo_vmm_allocator.model_arena_adapter import _NativeShardTransport, ModelArenaModuleAdapter
from ddo_vmm_allocator.native import NativeVmmError
from ddo_vmm_allocator.source import NativeSourceError


def batch(enabled=True, status=0):
    calls = []
    cache = NativeSourceCache.__new__(NativeSourceCache)
    cache._handle = ctypes.c_void_p(8)
    cache._lock = RLock()
    cache.source_dispatcher = enabled
    cache._dll = SimpleNamespace(
        ddo_vmm_source_batch_prepare=lambda handle: calls.append(handle.value) or status,
        ddo_vmm_source_cache_last_error=lambda: b"prepare failed")
    result = NativeSourceBatch.__new__(NativeSourceBatch)
    result._cache = cache
    result._handle = ctypes.c_void_p(17)
    result._destinations = []
    return result, calls


def test_prepare_calls_native_once_without_destination_or_dma():
    subject, calls = batch()
    assert subject.prepare_source() is True
    assert calls == [17] and subject._destinations == []
    assert subject._cache._batch_timings["batch_bridge_source_prepare_total_calls"] == 1


def test_old_or_disabled_library_does_not_prepare():
    subject, calls = batch(False)
    assert subject.prepare_source() is False and calls == []


@pytest.mark.parametrize("cache_closed", [False, True])
def test_prepare_rejects_closed_owners(cache_closed):
    subject, calls = batch()
    if cache_closed:
        subject._cache._handle.value = None
    else:
        subject._handle.value = None
    with pytest.raises(NativeVmmError, match="closed"):
        subject.prepare_source()
    assert calls == []


def test_prepare_failure_does_not_drop_batch_or_claim_success():
    subject, calls = batch(status=5)
    with pytest.raises(NativeSourceError, match="prepare failed"):
        subject.prepare_source()
    assert calls == [17] and subject._handle.value == 17
    assert subject._cache._handle.value == 8 and subject._destinations == []


def test_optional_dispatcher_window_snapshot_is_normalized_without_cuda():
    subject, _ = batch()
    cache = subject._cache
    cache._stats_version = 8
    cache._dll.ddo_vmm_source_cache_snapshot = lambda *args: 0
    calls = []

    def snapshot(handle, owned, peak, limit):
        calls.append(handle.value)
        owned._obj.value, peak._obj.value, limit._obj.value = 2, 8, 8
        return 0

    cache._dll.ddo_vmm_source_dispatcher_snapshot = snapshot
    result = cache.snapshot()
    assert calls == [8]
    assert result['source_dispatcher_owned_windows'] == 2
    assert result['source_dispatcher_peak_windows'] == result['source_dispatcher_window_limit'] == 8
    del cache._dll.ddo_vmm_source_dispatcher_snapshot
    assert 'source_dispatcher_peak_windows' not in cache.snapshot()


def test_transport_passes_unit_descriptors_without_window_policy():
    calls, prepared = [], []
    transport = _NativeShardTransport.__new__(_NativeShardTransport)
    transport.cache = SimpleNamespace(source_dispatcher=True,
        create_batch=lambda spans, ring, **kwargs: calls.append((spans, ring, kwargs))
        or SimpleNamespace(prepare_source=lambda: prepared.append(True)))
    transport._batches = {}
    transport._source = lambda path: ("immutable-file", path)
    transport.ring = "ring"
    transport.slots = 24
    transport.window_size_bytes = 64 * 2**20
    binding = SimpleNamespace(label="generic.weight", offset=512, nbytes=1024,
        file_slice=SimpleNamespace(path="shard.bin", offset_bytes=7))
    transport.prepare_sources([SimpleNamespace(tensors=[binding]), SimpleNamespace(tensors=[])])
    assert calls == [([(("immutable-file", "shard.bin"), 7, 1024, 512)],
                     "ring", {"ring_slots": 24, "ring_slot_bytes": 64 * 2**20})]
    assert prepared == [True]
    assert transport._immutable_batch([binding]) is next(iter(transport._batches.values()))
    assert len(calls) == 1


@pytest.mark.parametrize("cache", [None, SimpleNamespace(source_dispatcher=False)])
def test_transport_preserves_legacy_path_without_metadata_walk(cache):
    transport = _NativeShardTransport.__new__(_NativeShardTransport)
    transport.cache = cache
    class Forbidden:
        def __iter__(self):
            raise AssertionError("legacy path must not inspect units")
    transport.prepare_sources(Forbidden())


def test_actual_optional_exports_do_not_initialize_cuda():
    path = os.environ.get("DDO_VMM_LIBRARY_PATH")
    if not path:
        pytest.skip("explicit library path required for read-only export contract")
    dll = ctypes.CDLL(path)
    enabled = getattr(dll, "ddo_vmm_source_dispatcher_enabled", None)
    if enabled is None:
        pytest.skip("protected older DLL omits optional exports")
    enabled.argtypes, enabled.restype = [], ctypes.c_int
    assert enabled() in (0, 1)
    prepare = dll.ddo_vmm_source_batch_prepare
    prepare.argtypes, prepare.restype = [ctypes.c_void_p], ctypes.c_int
    assert prepare(None) == 1  # Rejected before context/device lookup.
    snapshot = getattr(dll, 'ddo_vmm_source_dispatcher_snapshot', None)
    if snapshot is not None:
        snapshot.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_size_t)] * 3
        snapshot.restype = ctypes.c_int
        assert snapshot(None, None, None, None) == 1


def test_source_startup_failure_cleans_up_after_view_scope_exits(monkeypatch):
    calls = []
    @contextmanager
    def setup(_spans):
        calls.append("view-enter")
        try:
            yield
        finally:
            calls.append("view-exit")
    def fail(_units):
        calls.append("source-start")
        raise RuntimeError("startup failed")
    store = SimpleNamespace(view_setup=setup,
        _native_shard_transport=SimpleNamespace(prepare_sources=fail),
        close=lambda: calls.append("store-close"))
    adapter = ModelArenaModuleAdapter(store)
    monkeypatch.setattr(adapter, "_attach_linear_modules", lambda *args, **kwargs: "report")
    monkeypatch.setattr(adapter, "close", lambda: calls.append("hooks-close"))
    with pytest.raises(RuntimeError, match="startup failed"):
        adapter.attach_linear_modules(None, ".*", _layout=((), {}, 1))
    assert calls == ["view-enter", "view-exit", "source-start", "hooks-close", "store-close"]
