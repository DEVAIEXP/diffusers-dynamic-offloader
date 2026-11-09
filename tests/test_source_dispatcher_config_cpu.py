"""Source dispatcher selection contracts without CUDA initialization."""
import argparse

import pytest

from diffusers_dynamic_offloader import dynamic_offload as runtime
from diffusers_dynamic_offloader.backend import add_dynamic_offload_arguments, resolve_dynamic_offload_backend


def test_dispatcher_is_selected_by_default():
    assert runtime.DynamicOffloadConfig().vmm_source_dispatcher is True
    settings = runtime.load_dynamic_offload_settings_from_env(
        environ={"DDO_EXECUTION_MODE": "vmm_fast_runtime"}, running_on_wsl=False)
    assert settings.config.vmm_source_dispatcher is True
    assert settings.as_metrics()["dynamic_offload_vmm_source_dispatcher"] is True


def test_environment_disables_the_dispatcher():
    settings = runtime.load_dynamic_offload_settings_from_env(
        environ={"DDO_EXECUTION_MODE": "vmm_fast_runtime", "DDO_VMM_SOURCE_DISPATCHER": "0"},
        running_on_wsl=False)
    assert settings.config.vmm_source_dispatcher is False


@pytest.mark.parametrize("value, expected", [(None, True), ("0", False), ("1", True), ("off", False)])
def test_runner_config_honors_dispatcher_and_unit_level_environment(monkeypatch, value, expected):
    monkeypatch.delenv("DDO_VMM_SOURCE_DISPATCHER", raising=False)
    monkeypatch.delenv("DDO_VMM_EXECUTION_UNIT_LEVEL", raising=False)
    if value is not None:
        monkeypatch.setenv("DDO_VMM_SOURCE_DISPATCHER", value)
    args = add_dynamic_offload_arguments(argparse.ArgumentParser()).parse_args(["--ddo-backend", "vmm"])
    config = resolve_dynamic_offload_backend("vmm", "auto").build_config_from_args(args)
    assert config.vmm_source_dispatcher is expected
    assert config.vmm_execution_unit_level == "block_child"
    monkeypatch.setenv("DDO_VMM_EXECUTION_UNIT_LEVEL", "block")
    assert resolve_dynamic_offload_backend("vmm", "auto").build_config_from_args(args).vmm_execution_unit_level == "block"
