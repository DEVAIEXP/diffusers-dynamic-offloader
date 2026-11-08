"""Execution unit level configuration contracts without CUDA initialization."""

import pytest

from diffusers_dynamic_offloader import dynamic_offload as runtime


def test_default_execution_unit_level_is_block_child():
    assert runtime.DynamicOffloadConfig().vmm_execution_unit_level == "block_child"
    settings = runtime.load_dynamic_offload_settings_from_env(
        environ={"DDO_EXECUTION_MODE": "vmm_fast_runtime"}, running_on_wsl=False)
    assert settings.config.vmm_execution_unit_level == "block_child"
    assert settings.as_metrics()["dynamic_offload_vmm_execution_unit_level"] == "block_child"


@pytest.mark.parametrize("value, expected", [("block", "block"), (" Block_Child ", "block_child")])
def test_environment_selects_execution_unit_level(value, expected):
    settings = runtime.load_dynamic_offload_settings_from_env(
        environ={"DDO_EXECUTION_MODE": "vmm_fast_runtime", "DDO_VMM_EXECUTION_UNIT_LEVEL": value},
        running_on_wsl=False)
    assert settings.config.vmm_execution_unit_level == expected


def test_supported_levels_match_the_allocator_adapter():
    adapter = pytest.importorskip("ddo_vmm_allocator.model_arena_adapter")
    assert runtime.VMM_EXECUTION_UNIT_LEVELS == adapter._EXECUTION_UNIT_LEVELS
