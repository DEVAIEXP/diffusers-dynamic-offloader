"""VMM residency profile recommendation, identity and application (stage 2), CPU only."""
import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusers_dynamic_offloader import DynamicOffloadConfig
from diffusers_dynamic_offloader import profiles
from diffusers_dynamic_offloader.backend import add_dynamic_offload_arguments, resolve_dynamic_offload_backend
from diffusers_dynamic_offloader.profiles import (
    VMM_FLOOR_ARENA_HEADROOM_BYTES, _NamedProfile, dynamic_offload_profile_key, dynamic_offload_profile_path,
    load_dynamic_offload_profile, recommend_vmm_resident_budget,
)

GiB = 1024**3
MiB = 1024**2
SLOTS = 608174080  # LTX 2.3 block_child: 2 x 304,087,040 B.
MACHINE = {"platform": "win32", "gpu_name": "Test GPU", "device_memory_mib": 8192, "host_ram_gib": 64,
           "native_library_sha256": "f818"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DDO_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(profiles, "_vmm_machine_identity", lambda config: dict(MACHINE))


def observation(**overrides):
    values = {"device_budget_bytes": int(7.61 * GiB), "peak_usage_bytes": int(5.24 * GiB),
              "total_weight_resident_bytes": SLOTS + 40 * MiB, "execution_slots_bytes": SLOTS,
              "arena_virtual_bytes": 22 * GiB}
    values.update(overrides)
    return values


def test_recommendation_subtracts_the_non_weight_peak_and_margin():
    result = recommend_vmm_resident_budget(observation(), 256 * MiB)
    non_weight = int(5.24 * GiB) - (SLOTS + 40 * MiB)
    assert result["non_weight_peak_bytes"] == non_weight
    assert result["vmm_resident_weight_budget_bytes"] == int(7.61 * GiB) - 256 * MiB - non_weight
    assert result["fit_class"] == "fits_with_headroom"
    assert result["floor_bytes"] == SLOTS + VMM_FLOOR_ARENA_HEADROOM_BYTES


def test_recommendation_clamps_to_the_floor_and_classifies():
    floor = SLOTS + VMM_FLOOR_ARENA_HEADROOM_BYTES
    budget = int(7.61 * GiB)
    weights = SLOTS + 40 * MiB
    # Non-weight peak leaving exactly the floor after the margin.
    at_limit = recommend_vmm_resident_budget(
        observation(peak_usage_bytes=budget - floor - 256 * MiB + weights), 256 * MiB)
    assert at_limit["fit_class"] == "fits_at_limit"
    assert at_limit["vmm_resident_weight_budget_bytes"] == floor
    too_big = recommend_vmm_resident_budget(observation(peak_usage_bytes=budget + SLOTS), 256 * MiB)
    assert too_big["fit_class"] == "does_not_fit"
    assert too_big["vmm_resident_weight_budget_bytes"] == floor


def test_recommendation_is_capped_by_the_model_size():
    result = recommend_vmm_resident_budget(observation(arena_virtual_bytes=100 * MiB), 256 * MiB)
    assert result["vmm_resident_weight_budget_bytes"] == SLOTS + VMM_FLOOR_ARENA_HEADROOM_BYTES + 100 * MiB


@pytest.mark.parametrize("missing", ["device_budget_bytes", "peak_usage_bytes", "total_weight_resident_bytes",
                                     "execution_slots_bytes"])
def test_unobservable_inputs_give_no_recommendation(missing):
    assert recommend_vmm_resident_budget(observation(**{missing: None}), 256 * MiB) is None


def config(**overrides):
    values = dict(execution_mode="vmm_fast_runtime", execution_device="cuda",
                  profile_name="ltx23-video-1280x704-8s", vmm_resident_weight_budget_gb=0.0)
    values.update(overrides)
    return DynamicOffloadConfig(**values)


def build_profile(observed):
    profile = _NamedProfile(nn.Linear(2, 2), config(build_profile=True), component="transformer")
    profile.prepare()
    profile.attach_hook(SimpleNamespace(vmm_runtime_stats=lambda: {}))
    original = profiles._VmmResidencySampler.stop
    profiles._VmmResidencySampler.stop = lambda self, stats: (self.close(), observed)[1]
    try:
        with profile.measure() as report:
            pass
    finally:
        profiles._VmmResidencySampler.stop = original
    return profile, report


def test_vmm_identity_uses_its_own_prefix_and_never_collides_with_linear():
    vmm = _NamedProfile(nn.Linear(2, 2), config(), component="transformer")
    linear = _NamedProfile(nn.Linear(2, 2), DynamicOffloadConfig(execution_mode="linear_runtime",
                                                                 profile_name="ltx23-video-1280x704-8s"))
    assert vmm.context["runtime"] == "vmm" and vmm.context["identity_version"] == 4
    assert vmm.context["unit_level"] == "block_child" and vmm.context["gpu_name"] == "Test GPU"
    assert dynamic_offload_profile_key(vmm.context).startswith("ddo-vmm-residency-v1-")
    assert dynamic_offload_profile_key(linear.context).startswith("ddo-capacity-v1-")
    assert dynamic_offload_profile_path(vmm.context) != dynamic_offload_profile_path(linear.context)


def test_build_then_use_applies_the_saved_budget_and_explicit_wins():
    profile, report = build_profile(observation())
    assert report["status"] == "saved" and report["fit_class"] == "fits_with_headroom"
    saved = load_dynamic_offload_profile(profile.context)
    budget = saved["recommendation"]["vmm_resident_weight_budget_gb"]
    used = _NamedProfile(nn.Linear(2, 2), config(), component="transformer")
    prepared = used.prepare()
    assert used.report["status"] == "applied"
    assert prepared.vmm_resident_weight_budget_gb == budget and prepared._profile_budget_applied
    explicit = _NamedProfile(nn.Linear(2, 2), config(vmm_resident_weight_budget_gb=0.625), component="transformer")
    assert explicit.prepare().vmm_resident_weight_budget_gb == 0.625
    assert explicit.report["status"] == "explicit"


def test_other_machine_or_component_does_not_reuse_the_profile(monkeypatch):
    build_profile(observation())
    monkeypatch.setattr(profiles, "_vmm_machine_identity", lambda config: {**MACHINE, "host_ram_gib": 32})
    other_ram = _NamedProfile(nn.Linear(2, 2), config(), component="transformer")
    assert other_ram.prepare().vmm_resident_weight_budget_gb == 0.0
    assert other_ram.report["status"] == "missing"
    monkeypatch.setattr(profiles, "_vmm_machine_identity", lambda config: dict(MACHINE))
    other_component = _NamedProfile(nn.Linear(2, 2), config(), component="text_encoder")
    assert other_component.prepare().vmm_resident_weight_budget_gb == 0.0


def test_unavailable_observations_save_without_recommendation():
    profile, report = build_profile(observation(device_budget_bytes=None))
    assert report["fit_class"] == "unavailable"
    saved = load_dynamic_offload_profile(profile.context)
    assert not saved.get("recommendation")
    used = _NamedProfile(nn.Linear(2, 2), config(), component="transformer")
    assert used.prepare().vmm_resident_weight_budget_gb == 0.0 and used.report["status"] == "missing"


def test_generic_cli_omitted_budget_resolves_per_route():
    args = add_dynamic_offload_arguments(argparse.ArgumentParser()).parse_args(["--ddo-backend", "vmm"])
    assert args.ddo_vmm_resident_weight_budget_gb is None
    assert resolve_dynamic_offload_backend("vmm", "auto").build_config_from_args(args).vmm_resident_weight_budget_gb == 0.0
    assert resolve_dynamic_offload_backend("vmm", "nvme_baseline").build_config_from_args(
        args).vmm_resident_weight_budget_gb == 4.0
    explicit = add_dynamic_offload_arguments(argparse.ArgumentParser()).parse_args(
        ["--ddo-backend", "vmm", "--ddo-vmm-resident-weight-budget-gb", "0.625"])
    assert resolve_dynamic_offload_backend("vmm", "auto").build_config_from_args(
        explicit).vmm_resident_weight_budget_gb == 0.625
