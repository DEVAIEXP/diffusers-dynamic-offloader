import argparse

import pytest

from diffusers_dynamic_offloader import add_dynamic_offload_arguments, resolve_dynamic_offload_backend


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_runner_config_honors_reclaim_environment(monkeypatch, value, expected):
    monkeypatch.delenv("DDO_VMM_EXECUTION_BACKING_RECLAIM", raising=False)
    if value is not None:
        monkeypatch.setenv("DDO_VMM_EXECUTION_BACKING_RECLAIM", value)
    parser = add_dynamic_offload_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--ddo-backend", "vmm"])
    route = resolve_dynamic_offload_backend("vmm", "auto")
    assert route.build_config_from_args(args).vmm_execution_backing_reclaim is expected
    assert route.build_config(vmm_execution_backing_reclaim=False).vmm_execution_backing_reclaim is False
    for backend, preset in [("linear", "auto"), ("vmm", "nvme_baseline")]:
        assert not resolve_dynamic_offload_backend(backend, preset).build_config_from_args(args).vmm_execution_backing_reclaim


def test_linear_auto_preserves_the_existing_default():
    route = resolve_dynamic_offload_backend()

    assert route.backend == "linear"
    assert route.effective_preset == "one_shot_fast"
    assert route.execution_mode == "linear_runtime"
    assert not route.requires_native_bootstrap


def test_vmm_auto_selects_native_fast_route():
    route = resolve_dynamic_offload_backend("vmm", "auto")

    assert route.effective_preset == "vmm_fast"
    assert route.execution_mode == "vmm_fast_runtime"
    assert route.requires_native_bootstrap


def test_vmm_explicit_fallback_is_protected():
    route = resolve_dynamic_offload_backend("vmm", "nvme_baseline")

    assert route.effective_preset == "nvme_baseline"
    assert route.execution_mode == "vmm_runtime"
    assert not route.requires_native_bootstrap


def test_resolved_backend_owns_the_runtime_mode_when_building_config():
    route = resolve_dynamic_offload_backend("vmm", "auto")

    config = route.build_config(vmm_resident_weight_budget_gb=4.0)

    assert config.execution_mode == "vmm_fast_runtime"
    assert config.vmm_resident_weight_budget_gb == 4.0
    with pytest.raises(ValueError, match="execution_mode"):
        route.build_config(execution_mode="plan")


def test_runner_arguments_expose_native_transport_and_source_cache_without_env():
    parser = add_dynamic_offload_arguments(argparse.ArgumentParser())
    args = parser.parse_args([
        "--ddo-backend", "vmm", "--ddo-vmm-native-transport",
        "--ddo-vmm-staging-buffer-gb", "0.5",
        "--ddo-vmm-native-source-cache-slot-mb", "64",
        "--ddo-vmm-native-source-cache-slots", "2",
    ])
    config = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).build_config_from_args(args)
    assert config.execution_mode == "vmm_fast_runtime"
    assert config.vmm_native_transport is True
    assert config.vmm_staging_buffer_gb == 0.5
    assert config.vmm_native_source_cache_slot_mb == 64
    assert config.vmm_native_source_cache_slots == 2


def test_invalid_vmm_preset_is_rejected():
    with pytest.raises(ValueError, match="Invalid VMM preset"):
        resolve_dynamic_offload_backend("vmm", "one_shot_fast")


def test_argument_helper_keeps_runner_surface_small():
    parser = add_dynamic_offload_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--ddo-backend", "vmm", "--show-metrics"])

    assert (args.ddo_backend, args.ddo_preset, args.show_metrics) == ("vmm", "auto", True)
    config = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).build_config_from_args(args)
    assert config.vmm_resident_weight_budget_gb == 4.0


def test_runner_arguments_preserve_independent_source_storage_and_registration():
    parser = add_dynamic_offload_arguments(argparse.ArgumentParser())
    args = parser.parse_args([
        "--ddo-backend", "vmm", "--ddo-vmm-source-cache-gb", "24.625",
        "--ddo-vmm-source-registered-gb", "0.5",
    ])
    config = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).build_config_from_args(args)
    assert config.vmm_source_cache_gb == 24.625
    assert config.vmm_source_registered_gb == 0.5
    assert config.vmm_resident_weight_budget_gb == 4.0


def test_registration_argument_defaults_do_not_change_protected_routes():
    parser = add_dynamic_offload_arguments(argparse.ArgumentParser())
    for arguments, mode in (([], "linear_runtime"),
                            (["--ddo-backend", "vmm", "--ddo-preset", "nvme_baseline"], "vmm_runtime")):
        args = parser.parse_args(arguments)
        config = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).build_config_from_args(args)
        assert config.execution_mode == mode
        assert config.vmm_source_cache_gb == 0.0
        assert config.vmm_source_registered_gb == 0.0


def test_existing_argument_namespaces_without_registration_remain_compatible():
    args = add_dynamic_offload_arguments(argparse.ArgumentParser()).parse_args(["--ddo-backend", "vmm"])
    del args.ddo_vmm_source_registered_gb
    assert resolve_dynamic_offload_backend("vmm").build_config_from_args(args).vmm_source_registered_gb == 0.0
