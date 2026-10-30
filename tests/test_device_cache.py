"""CPU contracts for explicitly opted-in native phase-boundary cleanup."""

from types import SimpleNamespace

import pytest

from diffusers_dynamic_offloader import device_cache as api


@pytest.mark.parametrize(
    "enabled,initialized,status",
    [
        (False, True, "disabled"),
        (True, False, "cuda_not_initialized"),
    ],
)
def test_noop_never_loads_or_synchronizes(monkeypatch, enabled, initialized, status):
    monkeypatch.setattr(api.torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(api, "_installed_native", lambda: pytest.fail("must not load"))
    report = api.release_native_vmm_idle_cache(enabled=enabled)
    assert report.setup["device_cache_release"]["status"] == status


def test_missing_installed_allocator_is_not_release(monkeypatch):
    monkeypatch.setattr(api.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(api, "_installed_native", lambda: None)
    result = api.release_native_vmm_idle_cache(enabled=True).setup["device_cache_release"]
    assert result["status"] == "native_allocator_not_installed"
    assert not result["trim_executed"]


@pytest.mark.parametrize("failure", [False, True])
def test_completion_precedes_trim_and_failures_propagate(monkeypatch, failure):
    calls = []

    def trim():
        calls.append("trim")
        if failure:
            raise RuntimeError("proof failed")
        return {
            "status": "completed",
            "trim_executed": True,
            "before": {"pool_mapped_bytes": 100},
            "after": {"pool_mapped_bytes": 20},
        }

    monkeypatch.setattr(api.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(api.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(api.torch.cuda, "synchronize", lambda d: calls.append(("sync", d)))
    monkeypatch.setattr(
        api, "_installed_native", lambda: SimpleNamespace(path="installed.dll", trim_idle_torch_pool=trim)
    )
    if failure:
        with pytest.raises(RuntimeError, match="proof failed"):
            api.release_native_vmm_idle_cache(enabled=True, device="cuda:0")
    else:
        result = api.release_native_vmm_idle_cache(enabled=True, device="cuda:0").setup["device_cache_release"]
        assert result["trim_executed"]
        assert result["after"]["pool_mapped_bytes"] == 20
    assert calls == ["gc", ("sync", "cuda:0"), "trim"]


@pytest.mark.parametrize(
    "missing,multiple,capturing", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_blas_cleanup_refuses_unsupported_or_unsafe_routes(monkeypatch, missing, multiple, capturing):
    monkeypatch.setattr(api.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(api, "_installed_native", lambda: SimpleNamespace())
    monkeypatch.setattr(
        api.torch._C, "_cuda_clearCublasWorkspaces", None if missing else lambda: pytest.fail("must not clear")
    )
    monkeypatch.setattr(api.torch.cuda, "device_count", lambda: 2 if multiple else 1)
    monkeypatch.setattr(api.torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(api.torch.cuda, "synchronize", lambda _: pytest.fail("must refuse before sync"))
    with pytest.raises(RuntimeError):
        api.release_native_vmm_idle_cache(enabled=True, clear_cublas_workspaces=True)


def test_blas_clear_follows_completion_and_precedes_native_trim(monkeypatch):
    calls = []
    monkeypatch.setattr(api.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(api.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(api.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(api.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(api.torch.cuda, "synchronize", lambda _: calls.append("sync"))
    monkeypatch.setattr(api.torch._C, "_cuda_clearCublasWorkspaces", lambda: calls.append("blas"))

    def trim():
        calls.append("trim")
        return {"status": "completed", "trim_executed": True}

    def snapshot():
        calls.append("snapshot")
        return {"pool_requested_bytes": 8519680}

    monkeypatch.setattr(
        api,
        "_installed_native",
        lambda: SimpleNamespace(path="installed.dll", trim_idle_torch_pool=trim, torch_pool_stats=snapshot),
    )
    result = api.release_native_vmm_idle_cache(enabled=True, clear_cublas_workspaces=True).setup["device_cache_release"]
    assert calls == ["gc", "sync", "snapshot", "blas", "trim"]
    assert result["cublas_workspaces_cleared"]
    assert result["pool_before_cublas_clear"]["pool_requested_bytes"] == 8519680
