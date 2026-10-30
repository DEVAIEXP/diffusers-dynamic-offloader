"""CPU-only optional allocator telemetry contracts."""
import ctypes
import os
from types import SimpleNamespace

import pytest
from ddo_vmm_allocator.native import NativeModelArena, NativeVmmLibrary, NativeVmmError
from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics


def owner(fn=None):
    value = NativeVmmLibrary.__new__(NativeVmmLibrary)
    value._library = SimpleNamespace()
    if fn is not None:
        value._library.ddo_vmm_torch_release_stats_v1 = fn
    return value


def test_old_dll_omits_unavailable_observations():
    assert owner().torch_release_stats() == {}
    assert owner().torch_memory_stats() == {}
    assert owner().torch_allocation_footprint() == {}
    assert owner().torch_pool_stats() == {}


def test_pool_snapshot_unique_backing_and_legacy_scope_omission():
    class Fn:
        def __call__(self, values, count):
            assert count == 15
            values[:] = [1, 4096, 513, 1024, 1536, 1024, 512, 1, 0, 2, 1024, 3, 250_000_000, 1, 1]
            return 0
    library = owner()
    fn = Fn()
    library._library.ddo_vmm_torch_pool_stats_v1 = fn
    stats = library.torch_memory_stats()
    assert stats["pool_schema_version"] == 1 and stats["pool_mapped_bytes"] == 4096
    assert stats["pool_requested_bytes"] == 513 and stats["pool_eviction_wait_seconds"] == 0.25
    assert "footprint_mapped_bytes" not in stats and "live_mapped_bytes" not in stats
    assert library.torch_release_stats() == {} and library.torch_allocation_footprint() == {}
    assert fn.argtypes == [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    report = DynamicOffloadMetrics(route="vmm", runtime={"vmm": {
        f"fast_torch_allocator_{k}": v for k, v in stats.items()}})
    assert "torch_span_pool=" in report.format_profile()
    assert "unique backing snapshot, not GPU total/peak" in report.format_profile()
    assert report.as_dict()["runtime"]["vmm"]["fast_torch_allocator_pool_mapped_bytes"] == 4096


@pytest.mark.parametrize("status", [1, 2, 0])
def test_pool_query_errors_and_unsupported_are_distinct(status):
    class Fn:
        def __call__(self, values, count):
            values[:] = [1, 8192, 4096, 1024, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
            return status
    library = owner()
    library._library.ddo_vmm_torch_pool_stats_v1 = Fn()
    if status == 2:
        assert library.torch_pool_stats() == {}
    else:
        with pytest.raises(NativeVmmError):
            library.torch_pool_stats()


def test_footprint_atomic_units_and_public_report():
    class Fn:
        calls = 0
        def __call__(self, values, count):
            self.calls += 1
            assert count == 6
            values[:] = [512, 2097152, 8388608, 2096640, 6291456, 1]
            return 0
    library = owner()
    fn = Fn()
    library._library.ddo_vmm_torch_allocation_footprint_v1 = fn
    stats = library.torch_memory_stats()
    assert stats == dict(footprint_requested_bytes=512, footprint_minimum_mapped_bytes=2097152,
                         footprint_mapped_bytes=8388608, footprint_granularity_padding_bytes=2096640,
                         footprint_reuse_excess_bytes=6291456, footprint_allocation_count=1)
    assert fn.calls == 1 and fn.restype == ctypes.c_int
    assert fn.argtypes == [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    report = DynamicOffloadMetrics(route="vmm", runtime={"vmm": {
        f"fast_torch_allocator_{k}": v for k, v in stats.items()}})
    assert "reuse_excess_gib=0.0059" in report.format_profile()
    assert "atomic live ledger, not GPU total/peak" in report.format_profile()
    assert report.as_dict()["runtime"]["vmm"]["fast_torch_allocator_footprint_requested_bytes"] == 512


@pytest.mark.parametrize("failure", [True, False])
def test_footprint_failure_or_inconsistent_output_is_not_zero(failure):
    class Fn:
        def __call__(self, values, count):
            values[:] = [4, 3, 2, 0, 0, 1]
            return int(failure)
    library = owner()
    library._library.ddo_vmm_torch_allocation_footprint_v1 = Fn()
    with pytest.raises(NativeVmmError):
        library.torch_memory_stats()


def test_memory_gauges_are_optional_read_only_size_t_exports():
    library = owner()

    class Fn:
        def __init__(self, value):
            self.value, self.calls = value, 0

        def __call__(self):
            self.calls += 1
            return self.value

    expected = dict(live_allocation_count=4, live_mapped_bytes=5 * 1024**3,
                    cached_allocation_count=2, cached_mapped_bytes=1024**3)
    functions = {}
    for name, value in expected.items():
        fn = functions[name] = Fn(value)
        setattr(library._library, f"ddo_vmm_torch_{name}", fn)
    assert library.torch_memory_stats() == expected
    assert all(fn.calls == 1 and fn.argtypes == [] and fn.restype == ctypes.c_size_t
               for fn in functions.values())


def test_memory_gauges_preserve_partial_support_and_actual_zero():
    library = owner()

    class Fn:
        def __call__(self):
            return 0

    library._library.ddo_vmm_torch_cached_mapped_bytes = Fn()
    assert library.torch_memory_stats() == {"cached_mapped_bytes": 0}


def test_memory_query_exception_is_not_reported_as_zero():
    library = owner()

    class Fn:
        def __call__(self):
            raise RuntimeError("query unavailable")

    library._library.ddo_vmm_torch_live_mapped_bytes = Fn()
    with pytest.raises(RuntimeError, match="query unavailable"):
        library.torch_memory_stats()


def test_arena_combines_release_counters_and_process_memory_gauges():
    arena = NativeModelArena.__new__(NativeModelArena)
    arena._owner = SimpleNamespace(torch_release_stats=lambda: {"reset_sync_calls": 3},
                                   torch_memory_stats=lambda: {"live_mapped_bytes": 4096})
    assert arena.torch_allocator_stats == {"reset_sync_calls": 3, "live_mapped_bytes": 4096}


def test_public_report_serializes_bytes_and_renders_gib_scope():
    report = DynamicOffloadMetrics(route="vmm", runtime={"vmm": {
        "fast_torch_allocator_live_mapped_bytes": 2 * 1024**3,
        "fast_torch_allocator_cached_mapped_bytes": 0}})
    assert "live_gib=2.0000 cached_gib=0.0000" in report.format_profile()
    assert "process ledger snapshot, not GPU total/peak" in report.format_profile()
    assert report.as_dict()["runtime"]["vmm"]["fast_torch_allocator_live_mapped_bytes"] == 2 * 1024**3


def test_public_report_does_not_invent_missing_memory_gauges():
    report = DynamicOffloadMetrics(route="vmm", runtime={"vmm": {"fast_faults": 1}})
    assert "torch_allocator=" not in report.format_profile()


def test_units_order_and_one_read_only_call():
    class Fn:
        calls = 0
        def __call__(self, values, count):
            self.calls += 1
            assert count == 6
            values[:] = [3, 250_000_000, 1, 2, 4096, 125_000_000]
            return 0
    fn = Fn()
    stats = owner(fn).torch_release_stats()
    assert fn.calls == 1 and fn.restype == ctypes.c_int
    assert stats == dict(reset_sync_calls=3, reset_sync_seconds=0.25,
                         reset_failures=1, cache_eviction_attempts=2,
                         cache_eviction_attempted_bytes=4096,
                         cache_eviction_sync_seconds=0.125)
    report = DynamicOffloadMetrics(route="vmm", runtime={
        "vmm": {f"fast_torch_allocator_{k}": v for k, v in stats.items()}})
    assert "torch_cache_eviction_sync_seconds=0.125000" in report.format_profile()
    assert "process lifetime" in report.format_profile()
    assert report.as_dict()["runtime"]["vmm"]["fast_torch_allocator_reset_failures"] == 1


def test_query_error_is_not_reported_as_zero():
    class Fn:
        def __call__(self, values, count):
            return 1
    with pytest.raises(NativeVmmError, match="query failed"):
        owner(Fn()).torch_release_stats()


@pytest.mark.skipif(not os.environ.get("DDO_VMM_LIBRARY_PATH"), reason="optional built DLL")
def test_real_export_is_cpu_only_and_checks_output_capacity():
    # Loading this export and reading atomics does not create a CUDA context.
    library = NativeVmmLibrary(os.environ["DDO_VMM_LIBRARY_PATH"])
    stats = library.torch_release_stats()
    pool = library.torch_pool_stats()
    assert (stats == {} if pool else len(stats) == 6)
    assert all(value >= 0 for value in stats.values())
    memory = library.torch_memory_stats()
    if pool:
        assert memory == pool
    else:
        assert {"live_allocation_count", "live_mapped_bytes",
                "cached_allocation_count", "cached_mapped_bytes"} <= set(memory)
    assert all(isinstance(value, (int, float)) and value >= 0 for value in memory.values())
    fn = library._library.ddo_vmm_torch_release_stats_v1
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    fn.restype = ctypes.c_int
    sentinels = (ctypes.c_uint64 * 6)(*[17] * 6)
    assert fn(sentinels, 5) != 0 and list(sentinels) == [17] * 6
    assert fn(None, 6) != 0
    footprint_fn = getattr(library._library, "ddo_vmm_torch_allocation_footprint_v1", None)
    if footprint_fn is not None:
        footprint = library.torch_allocation_footprint()
        if pool:
            assert footprint == {}
        else:
            assert len(footprint) == 6
            assert footprint["footprint_requested_bytes"] + footprint["footprint_granularity_padding_bytes"] + footprint["footprint_reuse_excess_bytes"] == footprint["footprint_mapped_bytes"]
        footprint_fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        footprint_fn.restype = ctypes.c_int
        assert footprint_fn(sentinels, 5) != 0 and list(sentinels) == [17] * 6
        assert footprint_fn(None, 6) != 0
    pool_fn = getattr(library._library, "ddo_vmm_torch_pool_stats_v1", None)
    if pool_fn is not None:
        pool_fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        pool_fn.restype = ctypes.c_int
        pool_sentinels = (ctypes.c_uint64 * 15)(*[17] * 15)
        assert pool_fn(pool_sentinels, 14) != 0 and list(pool_sentinels) == [17] * 15
        assert pool_fn(None, 15) != 0
