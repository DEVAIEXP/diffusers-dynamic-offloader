"""Opt-in execution backing policy contracts without CUDA initialization."""

import pytest

from diffusers_dynamic_offloader import dynamic_offload as runtime


def test_reclaim_remains_disabled_without_explicit_configuration():
    settings = runtime.load_dynamic_offload_settings_from_env(environ={}, running_on_wsl=False)
    assert settings.config.vmm_execution_backing_reclaim is False
    settings = runtime.load_dynamic_offload_settings_from_env(
        environ={"DDO_VMM_EXECUTION_BACKING_RECLAIM": "1"}, running_on_wsl=False)
    assert settings.config.vmm_execution_backing_reclaim is True


@pytest.mark.parametrize("mode", ["linear_runtime", "vmm_runtime", "off"])
def test_explicit_reclaim_refuses_other_routes_before_attachment(mode):
    config = runtime.DynamicOffloadConfig(execution_mode=mode, vmm_execution_backing_reclaim=True)
    with pytest.raises(ValueError, match="requires vmm_fast_runtime"):
        runtime.DynamicOffloadHook(config)


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_automatic_reclaim_refuses_non_windows_before_native_setup(monkeypatch, platform):
    monkeypatch.setattr(runtime.sys, "platform", platform)
    config = runtime.DynamicOffloadConfig(
        execution_mode="vmm_fast_runtime", vmm_execution_backing_reclaim=True)
    with pytest.raises(ValueError, match="requires Windows DXGI"):
        runtime.DynamicOffloadHook(config)
