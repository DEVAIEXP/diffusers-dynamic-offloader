import argparse

import pytest

from diffusers_dynamic_offloader import add_dynamic_offload_arguments, resolve_dynamic_offload_backend


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
