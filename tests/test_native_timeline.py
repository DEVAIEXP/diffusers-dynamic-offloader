"""CPU contracts: native window schema, optional exports, disabled observations."""
import ctypes
import os
from types import SimpleNamespace

import pytest
from ddo_vmm_allocator.native import NativeVmmError
from ddo_vmm_allocator.timeline import Info, Sample, NativeTimeline, decode_snapshot
from ddo_vmm_allocator.timeline import summarize_windows, union_intervals
from diffusers_dynamic_offloader import DynamicOffloadMetrics
from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadApplyResult


def info(**kwargs):
    result = Info(capacity=2, count=2, sealed=1, version=1)
    for k, v in kwargs.items():
        setattr(result, k, v)
    return result


def rows():
    return (Sample * 2)(Sample(1, 10, 4, 0, 1, 1, -500, 1000),
                        Sample(2, 10, 7, 0, 2, 1, 0, 250))


def test_signed_offsets_and_normalized_scope():
    report = decode_snapshot(info(), rows())
    assert report["status"] == "complete"
    assert report["samples"][0]["start_seconds"] == -0.5
    assert report["samples"][1]["end_seconds"] == 0.25
    assert report["samples"][1]["kind"] == "compute_upload_dependency"
    ddo = DynamicOffloadMetrics(route="vmm", runtime={"native_timeline": report})
    assert "not engine activity" in ddo.format_profile()
    assert ddo.as_dict()["runtime"]["native_timeline"]["capacity"] == 2


@pytest.mark.parametrize("state,status", [(0, "pending"), (2, "failed")])
def test_nonready_rows_have_no_fabricated_time(state, status):
    values = rows()
    values[0].state = state
    report = decode_snapshot(info(), values)
    assert report["status"] == status
    assert "start_seconds" not in report["samples"][0]


def test_truncation_and_failed_queries_are_not_complete():
    assert decode_snapshot(info(dropped=1), rows())["status"] == "truncated"
    assert decode_snapshot(info(failures=1), rows())["status"] == "failed"
    assert decode_snapshot(info(sealed=0), rows())["status"] == "pending"
    values = rows()
    values[0].end_ms = -1000
    assert decode_snapshot(info(), values)["status"] == "failed"


def test_reject_corrupt_payload_and_absent_export():
    with pytest.raises(NativeVmmError, match="schema"):
        decode_snapshot(info(version=2), rows())
    with pytest.raises(NativeVmmError, match="capacity"):
        decode_snapshot(info(count=3), rows())
    with pytest.raises(NativeVmmError, match="rebuilt"):
        NativeTimeline(SimpleNamespace(_library=SimpleNamespace()), capacity=2, stream=0)


@pytest.mark.parametrize("capacity", [0, -1, 8193])
def test_capacity_validation_before_any_native_call(capacity):
    with pytest.raises(ValueError, match="capacity"):
        NativeTimeline(None, capacity=capacity, stream=0)


def test_public_option_rejects_protected_routes_without_cuda():
    result = DynamicOffloadApplyResult(module=None, hook=None)
    with pytest.raises(ValueError, match="vmm_fast"):
        result.execution_timer(enabled=True, native_units=True)
    assert result.execution_timer().snapshot()["status"] == "disabled"


def test_window_reduction_deduplicates_and_clips_signed_intervals():
    assert union_intervals([(-2, 1), (0, 2), (3, 3), (4, 5)]) == [(-2, 2), (4, 5)]
    samples = [dict(kind=kind, status="complete", start_seconds=a, end_seconds=b)
               for kind, a, b in [("upload_batch_window", -2, 2),
                                   ("upload_batch_window", -1, 1),
                                   ("execution_unit_window", 0, 3),
                                   ("compute_upload_dependency", -1, 0)]]
    result = summarize_windows(samples, start=0, end=2)
    assert result["union_seconds"]["upload_batch_window"] == 2
    assert result["upload_execution_window_intersection_seconds"] == 2
    assert result["union_seconds"]["compute_upload_dependency"] == 0


def test_warm_window_uses_native_step_boundaries_not_host_clock():
    values = (Sample * 4)(Sample(1, 10, 4, 0, 1, 1, -1000, 3000),
                          Sample(2, 1, 7, 0, 5, 1, 1000, 1000),
                          Sample(3, 2**64-1, 7, 0, 5, 1, 2000, 2000),
                          Sample(4, 10, 7, 0, 4, 1, 1500, 2500))
    report = decode_snapshot(info(capacity=4, count=4), values)
    assert report["warm_window_summary"]["union_seconds"]["upload_batch_window"] == 1
    assert report["warm_window_summary"]["upload_execution_window_intersection_seconds"] == 0.5


def test_native_wrapper_retains_pending_and_closes_complete_without_waits():
    calls = []
    ready = [False]
    class Fn:
        def __init__(self, name):
            self.name = name
        def __call__(self, *args):
            calls.append(self.name)
            if self.name == "open":
                ctypes.cast(args[2], ctypes.POINTER(ctypes.c_void_p))[0] = 9
            elif self.name == "snapshot":
                record = ctypes.cast(args[1], ctypes.POINTER(Info))
                record[0] = info(count=1, pending=0 if ready[0] else 1)
                args[2][0] = Sample(1, 10, 0, 0, 1, 1 if ready[0] else 0, -100, 200)
            elif self.name == "close":
                if not ready[0]:
                    return 1
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = None
            return 0
    dll = SimpleNamespace(**{f"ddo_vmm_timeline_{name}_v1": Fn(name) for name in
                              ("open", "mark_begin", "mark_end", "seal", "snapshot", "close")})
    session = NativeTimeline(SimpleNamespace(_library=dll), capacity=2, stream=0)
    session.seal()
    assert session.snapshot()["status"] == "pending"
    assert "close" not in calls and session._handle.value == 9
    assert not session.close() and session._handle.value == 9
    assert "close" not in calls
    ready[0] = True
    assert session.close()
    report = session.snapshot()
    assert report["status"] == "complete" and report["event_cleanup"] == "released"
    assert not session._handle.value
    previous = list(calls)
    assert session.snapshot() is report and calls == previous


@pytest.mark.skipif(not os.environ.get("DDO_VMM_LIBRARY_PATH"), reason="built DLL optional")
def test_native_disabled_and_invalid_calls_do_not_initialize_cuda():
    loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
    dll = loader(os.environ["DDO_VMM_LIBRARY_PATH"])
    begin = dll.ddo_vmm_timeline_mark_begin_v1
    begin.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]
    begin.restype = ctypes.c_uint64
    assert begin(1, 1, 0, None) == 0
    open_fn = dll.ddo_vmm_timeline_open_v1
    open_fn.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    handle = ctypes.c_void_p()
    assert open_fn(0, None, ctypes.byref(handle)) != 0
    assert not handle.value
    assert open_fn(8193, None, ctypes.byref(handle)) != 0
