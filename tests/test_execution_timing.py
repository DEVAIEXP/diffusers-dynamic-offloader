"""CPU-only event timing and report integration contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadExecutionTimer, DynamicOffloadMetrics
from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadApplyResult, DynamicOffloadHook


@pytest.fixture
def events(monkeypatch):
    made = []
    stream = SimpleNamespace(cuda_stream=123)

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.index = len(made)
            self.ready = False
            self.waits = 0
            made.append(self)

        def record(self, actual):
            assert actual is stream

        def query(self):
            return self.ready

        def synchronize(self):
            self.waits += 1
            self.ready = True

        def elapsed_time(self, other):
            assert made[-1].ready
            timestamps = [0, 1000, 3000, 3500]
            return timestamps[other.index] - timestamps[self.index]

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return made


def test_timing_pending_is_not_zero_and_never_waits(events):
    with DynamicOffloadExecutionTimer(enabled=True) as timer:
        timer.mark_step()
        timer.mark_step()
    pending = timer.snapshot()
    assert pending["status"] == "pending"
    assert "phase_seconds" not in pending and "step_seconds" not in pending
    assert all(event.waits == 0 for event in events)
    events[-1].ready = True
    complete = timer.snapshot()
    assert complete["step_seconds"] == [1.0, 2.0]
    assert complete["phase_seconds"] == 3.5
    assert complete["tail_seconds"] == 0.5
    assert complete["steady_state_seconds_per_step"] == 2.0
    assert complete["stream"] == 123
    assert all(event.waits == 0 for event in events)


def test_only_explicit_final_wait_synchronizes(events):
    with DynamicOffloadExecutionTimer(enabled=True) as timer:
        timer.mark_step()
        timer.mark_step()
    assert timer.snapshot(wait=True)["status"] == "complete"
    assert [event.waits for event in events] == [0, 0, 0, 1]


def test_diagnostic_wait_only_touches_each_step_event_and_labels_report(events):
    with DynamicOffloadExecutionTimer(enabled=True, diagnostic_step_wait=True) as timer:
        timer.mark_step()
        timer.mark_step()
    assert [event.waits for event in events] == [0, 1, 1, 0]
    events[-1].ready = True
    report = timer.attach(DynamicOffloadMetrics(route="vmm"))
    diagnostic = report.runtime["execution_timing"]["diagnostic_step_wait"]
    assert diagnostic["calls"] == 2 and diagnostic["host_seconds"] >= 0
    assert diagnostic["benchmark_eligible"] is False
    assert "not a benchmark" in report.format_profile()


def test_diagnostic_wait_rejected_without_timing_or_native_route():
    with pytest.raises(ValueError, match="enabled"):
        DynamicOffloadExecutionTimer(diagnostic_step_wait=True)
    result = DynamicOffloadApplyResult(module=None, hook=None)
    with pytest.raises(ValueError, match="native unit"):
        result.execution_timer(enabled=True, diagnostic_step_wait=True)
    with pytest.raises(ValueError, match="vmm_fast"):
        result.execution_timer(enabled=True, native_units=True, diagnostic_step_wait=True)


def test_failed_run_is_labelled_and_does_not_wait(events):
    with pytest.raises(ValueError), DynamicOffloadExecutionTimer(enabled=True) as timer:
        raise ValueError("forward failed")
    assert timer.snapshot()["failed"] is True
    assert timer.snapshot()["status"] == "pending"
    assert all(event.waits == 0 for event in events)


def test_disabled_timer_does_not_touch_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled timing must not inspect CUDA")
    monkeypatch.setattr(torch.cuda, "Event", unexpected)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected)
    monkeypatch.setattr(torch.cuda, "device", unexpected)
    with DynamicOffloadExecutionTimer() as timer:
        timer.mark_step()
    assert timer.snapshot(wait=True)["status"] == "disabled"


def test_timer_rejects_marker_outside_scope_and_reuse(events):
    timer = DynamicOffloadExecutionTimer(enabled=True)
    with pytest.raises(RuntimeError, match="active"):
        timer.mark_step()
    with timer:
        assert timer.snapshot()["status"] == "not_finished"
    with pytest.raises(RuntimeError, match="reused"):
        with timer:
            pass


def test_timer_attaches_public_report_without_mutation(events):
    report = DynamicOffloadMetrics(route="vmm_fast_runtime", runtime={"other": 1})
    with DynamicOffloadExecutionTimer(enabled=True) as timer:
        timer.mark_step()
        timer.mark_step()
    events[-1].ready = True
    combined = timer.attach(report)
    assert "execution_timing" not in report.runtime
    assert combined.as_dict()["runtime"]["execution_timing"]["phase_seconds"] == 3.5
    assert combined.runtime["other"] == 1
    assert "cuda_phase_seconds=3.500000" in combined.format_profile()


def test_native_scope_is_sealed_detached_and_consumed_only_in_report(events):
    calls = []
    native = SimpleNamespace(mark_boundary=lambda n, stream: calls.append(("marker", n, stream)),
                             seal=lambda: calls.append("seal"),
                             snapshot=lambda: {"status": "complete", "samples": []})
    def factory(stream):
        calls.append(("open", stream))
        return native, lambda: calls.append("detach")
    with DynamicOffloadExecutionTimer(enabled=True, native_factory=factory) as timer:
        timer.mark_step()
        timer.mark_step()
    assert calls == [("open", 123), ("marker", 0, 123), ("marker", 1, 123),
                     ("marker", 2, 123), ("marker", 2**64-1, 123), "seal", "detach"]
    events[-1].ready = True
    report = timer.attach(DynamicOffloadMetrics(route="vmm"))
    assert report.runtime["native_timeline"]["status"] == "complete"


def test_native_scope_detaches_if_basic_marker_initialization_fails(events, monkeypatch):
    calls = []
    native = SimpleNamespace(mark_boundary=lambda *args: None,
                             seal=lambda: calls.append("seal"))
    def fail(**kwargs):
        raise RuntimeError("marker failed")
    monkeypatch.setattr(torch.cuda, "Event", fail)
    timer = DynamicOffloadExecutionTimer(enabled=True, native_factory=lambda stream:
                                         (native, lambda: calls.append("detach")))
    with pytest.raises(RuntimeError, match="marker failed"):
        with timer:
            pass
    assert calls == ["seal", "detach"]


def test_result_collection_is_explicit_and_reported():
    calls = []
    def collect():
        calls.append("collect")
        return {"tickets_retired": 4}
    hook = SimpleNamespace(state=SimpleNamespace(as_dict=lambda: {}),
                           vmm_runtime_stats=lambda: {}, collect_completed_runtime_resources=collect)
    result = DynamicOffloadApplyResult(module=None, hook=hook)
    assert "completion_collection" not in result.metrics_snapshot().runtime
    assert calls == []
    report = result.metrics_snapshot(collect_completed=True)
    assert report.runtime["completion_collection"]["tickets_retired"] == 4
    assert calls == ["collect"]


def test_non_native_collection_does_not_touch_cuda():
    hook = SimpleNamespace(_vmm_fast_store=None)
    assert DynamicOffloadHook.collect_completed_runtime_resources(hook)["status"] == "not_applicable"


def test_native_collection_uses_one_native_call_and_before_after_counts():
    calls = []
    snapshots = iter([{"live_tickets": 12}, {"live_tickets": 0}])
    cache = SimpleNamespace(snapshot=lambda: next(snapshots), collect_completed=lambda: calls.append("native"))
    hook = SimpleNamespace(_vmm_fast_store=SimpleNamespace(_native_shard_transport=SimpleNamespace(cache=cache)))
    result = DynamicOffloadHook.collect_completed_runtime_resources(hook)
    assert calls == ["native"]
    assert result == {"status": "collected", "blocking": False,
                      "tickets_before": 12, "tickets_after": 0, "tickets_retired": 12}
