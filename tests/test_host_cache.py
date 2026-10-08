"""CPU mocks only: explicit boundary ownership and public report contracts."""
import json
from types import SimpleNamespace

import pytest

from diffusers_dynamic_offloader import host_cache as api
from diffusers_dynamic_offloader import release_pytorch_host_cache


def install(monkeypatch, *, initialized=True, supported=True, failure=None):
    calls = []
    stats = iter([{"allocated_bytes.current": 100, "active_bytes.current": 300},
                  {"allocated_bytes.current": 20, "active_bytes.current": 300}])
    def empty():
        calls.append("empty")
        if failure:
            raise failure
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: initialized,
        synchronize=lambda device: calls.append(("sync", device)),
        host_memory_stats=lambda: next(stats)),
        _C=SimpleNamespace(**({"_accelerator_emptyHostCache": empty} if supported else {})))
    monkeypatch.setattr(api, "torch", fake)
    monkeypatch.setattr(api.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(api, "_host_snapshot", lambda: {"available_process_commit_bytes": 1024})
    return calls, fake


@pytest.mark.parametrize("enabled,initialized,supported,status", [
    (False, True, True, "disabled"),
    (True, False, True, "cuda_not_initialized"),
    (True, True, False, "unsupported"),
])
def test_noop_does_not_collect_synchronize_or_release(monkeypatch, enabled, initialized, supported, status):
    calls, _ = install(monkeypatch, initialized=initialized, supported=supported)
    report = release_pytorch_host_cache(enabled=enabled)
    assert calls == []
    assert report.setup["host_cache_release"]["status"] == status
    assert not report.setup["host_cache_release"]["released"]


def test_completed_transfers_precede_release_and_inconsistent_active_is_not_capacity(monkeypatch):
    calls, _ = install(monkeypatch)
    report = release_pytorch_host_cache(enabled=True, phase="before_transformer", device="cuda:0")
    assert calls == ["gc", ("sync", "cuda:0"), "empty"]
    result = json.loads(json.dumps(report.as_dict()))["setup"]["host_cache_release"]
    assert result["allocator_bytes_delta"] == 80
    assert result["after"]["allocated_bytes"] == 20  # Live allocations are not discarded.
    assert result["before"]["reported_active_bytes"] == 300
    assert result["before"]["active_bytes"] is None
    assert not result["before"]["active_counter_consistent"]
    assert "allocator_bytes_delta=80" in report.format_profile()


def test_missing_stats_do_not_invent_zero_reclaim(monkeypatch):
    _, fake = install(monkeypatch)
    del fake.cuda.host_memory_stats
    result = release_pytorch_host_cache(enabled=True).setup["host_cache_release"]
    assert result["released"]
    assert result["allocator_bytes_delta"] is None


@pytest.mark.parametrize("where", ["sync", "release"])
def test_failures_propagate_and_failed_sync_never_releases(monkeypatch, where):
    original = RuntimeError("boundary failed")
    calls, fake = install(monkeypatch, failure=original if where == "release" else None)
    if where == "sync":
        def fail(_):
            raise original
        fake.cuda.synchronize = fail
    with pytest.raises(RuntimeError) as captured:
        release_pytorch_host_cache(enabled=True)
    assert captured.value is original
    if where == "sync":
        assert "empty" not in calls


def test_consistent_stats_and_stat_errors(monkeypatch):
    _, fake = install(monkeypatch)
    fake.cuda.host_memory_stats = lambda: {"allocated_bytes.current": 100, "active_bytes.current": 40}
    snapshot = api._allocator_snapshot()
    assert snapshot["active_counter_consistent"] and snapshot["active_bytes"] == 40
    def fail():
        raise RuntimeError("stats unavailable")
    fake.cuda.host_memory_stats = fail
    assert api._allocator_snapshot() == {"status": "unavailable", "error": "stats unavailable"}


def test_real_cpu_process_does_not_initialize_cuda():
    import torch
    assert not torch.cuda.is_initialized()
    result = release_pytorch_host_cache(enabled=True).setup["host_cache_release"]
    assert result["status"] == "cuda_not_initialized"
    assert not torch.cuda.is_initialized()
