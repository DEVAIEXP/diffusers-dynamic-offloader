"""Component-only routing contracts; never initialize CUDA or execute a benchmark."""

from dataclasses import replace

import pytest
from torch import nn

from diffusers_dynamic_offloader import (
    DynamicOffloadConfig,
    enable_offload,
    load_dynamic_offload_settings_from_env,
)


@pytest.mark.parametrize("preset", ["auto", "one_shot_fast", "low_ram_safe"])
def test_vae_leaf_default_does_not_change_transformer(preset):
    result = enable_offload(
        nn.Sequential(nn.Conv2d(2, 2, 1)),
        component="vae",
        preset=preset,
        use_environment=False,
        execution_device="cpu",
        apply_hook=False,
    )
    assert result.route == "diffusers_group_offload"
    assert result.event_payload["offload_type"] == "leaf_level"
    assert result.event_payload["use_stream"] is False
    transformer = enable_offload(
        nn.Linear(2, 2),
        component="transformer",
        preset=preset,
        use_environment=False,
        execution_device="cpu",
        apply_hook=False,
    )
    assert transformer.route == "dynamic_offload"
    assert transformer.settings.config.execution_mode == "linear_runtime"
    assert not transformer.settings.config.linear_convolution_offload


@pytest.mark.parametrize(
    "preset,expected",
    [
        ("vmm_fast", "diffusers_group_offload"),
        ("nvme_baseline", "none"),
        ("warm_process", "none"),
        ("wsl_compat", "none"),
        ("off", "none"),
    ],
)
def test_native_and_preserved_policies(preset, expected):
    settings = load_dynamic_offload_settings_from_env(execution_device="cpu", environ={}, default_preset="off")
    settings = replace(settings, effective_preset=preset)
    result = enable_offload(
        nn.Conv2d(2, 2, 1),
        settings=settings,
        component="vae",
        execution_device="cpu",
        use_environment=False,
        apply_hook=False,
    )
    assert result.route == expected


def test_explicit_linear_route_uses_isolated_configuration():
    result = enable_offload(
        nn.Sequential(nn.Conv2d(2, 2, 1)).eval(),
        component="vae",
        preset="one_shot_fast",
        route="linear_convolution",
        execution_device="cpu",
        use_environment=False,
        pin_cpu_memory=False,
        apply_hook=False,
    )
    assert result.route == "dynamic_offload"
    assert result.settings.config.linear_convolution_offload
    assert result.settings.config.auto_budget_policy == "off"
    assert result.settings.config.resident_module_budget_gb == 0
    assert not result.settings.config.cache_pinned_weights


def test_explicit_config_is_not_replaced_by_default_leaf():
    config = DynamicOffloadConfig(
        execution_device="cpu", execution_mode="linear_runtime", linear_convolution_offload=True
    )
    result = enable_offload(
        nn.Sequential(nn.Conv2d(2, 2, 1)).eval(),
        component="vae",
        config=config,
        execution_device="cpu",
        use_environment=False,
        apply_hook=False,
    )
    assert result.route == "dynamic_offload"
    assert result.settings.config is config


def test_policy_and_explicit_route_precedence():
    settings = load_dynamic_offload_settings_from_env(
        execution_device="cpu",
        environ={},
        default_preset="auto",
        component_policies={"vae": {"route": "linear_convolution"}},
    )
    result = enable_offload(
        nn.Sequential(nn.Conv2d(2, 2, 1)).eval(),
        settings=settings,
        component="vae",
        execution_device="cpu",
        use_environment=False,
        apply_hook=False,
    )
    assert result.settings.config.linear_convolution_offload
    resident = enable_offload(
        nn.Conv2d(2, 2, 1),
        settings=settings,
        component="vae",
        route="none",
        execution_device="cpu",
        use_environment=False,
        apply_hook=False,
    )
    assert resident.route == "none"


def test_disabled_environment_does_not_leak(monkeypatch):
    monkeypatch.setenv("DDO_VAE_ROUTE", "none")
    result = enable_offload(
        nn.Conv2d(2, 2, 1), component="vae", use_environment=False, execution_device="cpu", apply_hook=False
    )
    assert result.route == "diffusers_group_offload"
    override = enable_offload(
        nn.Conv2d(2, 2, 1), component="vae", environ={"DDO_VAE_ROUTE": "none"}, execution_device="cpu", apply_hook=False
    )
    assert override.route == "none"


def test_leaf_apply_calls_official_helper_with_accepted_options(monkeypatch):
    import diffusers.hooks

    calls = []
    monkeypatch.setattr(diffusers.hooks, "apply_group_offloading", lambda module, **kwargs: calls.append(kwargs))
    result = enable_offload(nn.Conv2d(2, 2, 1), component="vae", execution_device="cpu", use_environment=False)
    assert result.route == "diffusers_group_offload"
    assert len(calls) == 1
    assert calls[0]["offload_type"] == "leaf_level"
    assert calls[0]["use_stream"] is False
    assert calls[0]["record_stream"] is False
    assert calls[0]["low_cpu_mem_usage"] is False


def test_explicit_group_and_dynamic_enable_arguments_win():
    group = enable_offload(
        nn.Conv2d(2, 2, 1),
        component="vae",
        group_offload=True,
        use_stream=True,
        execution_device="cpu",
        use_environment=False,
        apply_hook=False,
    )
    assert group.event_payload["use_stream"] is True
    resident = enable_offload(
        nn.Conv2d(2, 2, 1),
        component="vae",
        dynamic_offload=False,
        execution_device="cpu",
        use_environment=False,
        apply_hook=False,
    )
    assert resident.route == "none"
