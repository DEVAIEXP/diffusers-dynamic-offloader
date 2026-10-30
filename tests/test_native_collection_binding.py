"""CPU-only optional-export and ownership tests."""
import ctypes
from threading import RLock
from types import SimpleNamespace

import pytest
from ddo_vmm_allocator.source_cache import NativeSourceCache
from ddo_vmm_allocator.native import NativeVmmError
from ddo_vmm_allocator.source import NativeSourceError


def cache_with(fn=None):
    cache = NativeSourceCache.__new__(NativeSourceCache)
    cache._lock = RLock()
    cache._handle = ctypes.c_void_p(7)
    cache._dll = SimpleNamespace(ddo_vmm_source_cache_last_error=lambda: b"query failed")
    if fn is not None:
        cache._dll.ddo_vmm_source_cache_collect_completed = fn
    return cache


def test_collection_calls_native_once_without_ticket_iteration():
    class Fn:
        def __call__(self, handle):
            assert handle.value == 7
            self.calls += 1
            return 0
        calls = 0
    fn = Fn()
    cache_with(fn).collect_completed()
    assert fn.calls == 1 and fn.restype == ctypes.c_int


def test_collection_keeps_old_dll_compatible_until_explicit_request():
    with pytest.raises(NativeVmmError, match="rebuilt"):
        cache_with().collect_completed()


def test_collection_rejects_closed_owner():
    cache = cache_with()
    cache._handle.value = None
    with pytest.raises(NativeVmmError, match="closed"):
        cache.collect_completed()


def test_collection_propagates_query_failure_and_keeps_owner():
    class Fn:
        def __call__(self, handle):
            return 5
    cache = cache_with(Fn())
    with pytest.raises(NativeSourceError, match="query failed"):
        cache.collect_completed()
    assert cache._handle.value == 7
