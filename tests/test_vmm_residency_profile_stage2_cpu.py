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


def build_profile(observed, **overrides):
    """One build run whose sampler returns ``observed``; returns (profile, prepared config, report)."""
    profile = _NamedProfile(nn.Linear(2, 2), config(build_profile=True, **overrides), component="transformer")
    prepared = profile.prepare()
    profile.attach_hook(SimpleNamespace(vmm_runtime_stats=lambda: {}))
    original = profiles._VmmResidencySampler.stop
    profiles._VmmResidencySampler.stop = lambda self, stats: (self.close(), observed)[1]
    try:
        with profile.measure() as report:
            pass
    finally:
        profiles._VmmResidencySampler.stop = original
    return profile, prepared, report


SHAPES = {"args": [[1, 22000, 128]], "kwargs": None}


def timed(seconds, spread=0.05, **overrides):
    return observation(**{"steady_forward_seconds": seconds, "steady_forward_spread_seconds": spread,
                          "input_shapes": SHAPES, **overrides})


def saved(profile):
    return load_dynamic_offload_profile(profile.context)["recommendation"]


def test_vmm_identity_uses_its_own_prefix_and_never_collides_with_linear():
    vmm = _NamedProfile(nn.Linear(2, 2), config(), component="transformer")
    linear = _NamedProfile(nn.Linear(2, 2), DynamicOffloadConfig(execution_mode="linear_runtime",
                                                                 profile_name="ltx23-video-1280x704-8s"))
    assert vmm.context["runtime"] == "vmm" and vmm.context["identity_version"] == 4
    assert vmm.context["unit_level"] == "block_child" and vmm.context["gpu_name"] == "Test GPU"
    assert dynamic_offload_profile_key(vmm.context).startswith("ddo-vmm-residency-v1-")
    assert dynamic_offload_profile_key(linear.context).startswith("ddo-capacity-v1-")
    assert dynamic_offload_profile_path(vmm.context) != dynamic_offload_profile_path(linear.context)


FLOOR = SLOTS + VMM_FLOOR_ARENA_HEADROOM_BYTES


def test_build_one_measures_the_floor_and_saves_a_pending_ceiling():
    profile, prepared, report = build_profile(timed(33.0))
    assert prepared.vmm_resident_weight_budget_gb == 0.0  # DDO sizes the structural floor.
    assert report["build_stage"] == "floor_measured" and report["decision"] == "pending_ceiling_build"
    pending = saved(profile)
    assert pending["vmm_resident_weight_budget_bytes"] == FLOOR == pending["floor_budget_bytes"]
    assert pending["ceiling_budget_bytes"] > FLOOR and pending["floor_forward_seconds"] == 33.0
    used = _NamedProfile(nn.Linear(2, 2), config(), component="transformer")
    assert used.prepare().vmm_resident_weight_budget_gb == round(FLOOR / GiB, 4)  # Floor meanwhile.
    assert used.report["build_stage"] == "floor_measured"


@pytest.mark.parametrize("ceiling_seconds, spread, decision, reason", [
    (32.97, 0.05, "floor", "gain_within_timing_noise"),      # 0.03 s gain inside 0.05 s spread.
    (32.80, 0.01, "floor", "gain_below_threshold"),          # Above noise, ~0.2 %/GiB.
    (30.00, 0.05, "ceiling", "gain_at_or_above_threshold"),
    (30.00, 4.00, "floor", "gain_within_timing_noise"),      # Large gain but noisier than it.
])
def test_build_two_keeps_the_ceiling_only_above_threshold_and_noise(ceiling_seconds, spread, decision, reason):
    first, _, _ = build_profile(timed(33.0, spread=0.01))
    ceiling = saved(first)["ceiling_budget_bytes"]
    profile, prepared, report = build_profile(timed(ceiling_seconds, spread=spread))
    assert prepared.vmm_resident_weight_budget_gb == ceiling / GiB  # Measured at the ceiling.
    final = saved(profile)
    assert final["vmm_build_stage"] == "final" and (final["decision"], final["reason"]) == (decision, reason)
    assert final["vmm_resident_weight_budget_bytes"] == (ceiling if decision == "ceiling" else FLOOR)
    used = _NamedProfile(nn.Linear(2, 2), config(), component="transformer")
    assert used.prepare().vmm_resident_weight_budget_gb == final["vmm_resident_weight_budget_gb"]
    assert used.report["decision"] == decision
    # A third build restarts the calibration from the floor.
    _, restarted, report = build_profile(timed(33.0))
    assert restarted.vmm_resident_weight_budget_gb == 0.0 and report["build_stage"] == "floor_measured"


def test_ceiling_run_over_the_device_budget_trims_the_ceiling():
    build_profile(timed(33.0))
    profile, _, _ = build_profile(timed(25.0, peak_usage_bytes=int(7.6 * GiB)))
    final = saved(profile)
    overshoot = int(7.6 * GiB) - (int(7.61 * GiB) - 256 * MiB)
    assert (final["decision"], final["reason"]) == ("ceiling", "ceiling_trimmed_to_device_budget")
    assert final["ceiling_trim_bytes"] == overshoot
    assert final["vmm_resident_weight_budget_bytes"] == final["ceiling_budget_bytes"] - overshoot > FLOOR


def test_ceiling_run_checks_the_lowest_budget_of_both_builds():
    # WDDM lowered the process budget during build 2; build 1's budget alone would pass.
    build_profile(timed(33.0))
    lower = int(7.5 * GiB)
    profile, _, _ = build_profile(timed(25.0, peak_usage_bytes=int(7.3 * GiB), device_budget_bytes=lower))
    final = saved(profile)
    assert final["build_min_device_budget_bytes"] == lower
    assert final["ceiling_trim_bytes"] == int(7.3 * GiB) - (lower - 256 * MiB)
    assert final["reason"] == "ceiling_trimmed_to_device_budget"


def test_ceiling_run_with_no_room_above_the_floor_falls_back_to_the_floor():
    build_profile(timed(33.0))
    profile, _, _ = build_profile(timed(25.0, peak_usage_bytes=int(7.61 * GiB) + 3 * GiB))
    final = saved(profile)
    assert (final["decision"], final["reason"]) == ("floor", "ceiling_run_exceeded_device_budget")
    assert final["vmm_resident_weight_budget_bytes"] == FLOOR


def test_shape_mismatch_between_builds_requires_a_restart():
    build_profile(timed(33.0))
    profile, _, report = build_profile(timed(30.0, input_shapes={"args": [[1, 9, 128]], "kwargs": None}))
    assert report["shape_check"] == "mismatch"
    assert saved(profile)["vmm_build_stage"] == "restart_required"
    _, restarted, _ = build_profile(timed(33.0))
    assert restarted.vmm_resident_weight_budget_gb == 0.0  # Restarted from the floor.


def test_single_build_decisions_without_a_second_build():
    whole, _, _ = build_profile(timed(33.0, arena_virtual_bytes=100 * MiB))
    assert (saved(whole)["decision"], saved(whole)["vmm_build_stage"]) == ("whole_model", "final")
    assert saved(whole)["vmm_resident_weight_budget_bytes"] == FLOOR + 100 * MiB


def test_no_headroom_or_no_timing_keeps_the_floor(tmp_path, monkeypatch):
    profile, _, _ = build_profile(timed(33.0, peak_usage_bytes=int(7.61 * GiB) + SLOTS))
    assert (saved(profile)["decision"], saved(profile)["reason"]) == ("floor", "no_headroom")
    monkeypatch.setenv("DDO_PROFILE_DIR", str(tmp_path / "other"))
    profile, _, _ = build_profile(timed(None))
    assert saved(profile)["reason"] == "timing_unavailable_use_three_or_more_steps"
    monkeypatch.setenv("DDO_PROFILE_DIR", str(tmp_path / "two-steps"))
    profile, _, _ = build_profile(timed(33.0, spread=None))  # Two steps: one steady forward only.
    assert saved(profile)["reason"] == "timing_unavailable_use_three_or_more_steps"


def test_explicit_budget_is_refused_during_a_build_and_wins_otherwise():
    with pytest.raises(ValueError, match="measures its own budgets"):
        _NamedProfile(nn.Linear(2, 2), config(build_profile=True, vmm_resident_weight_budget_gb=0.625),
                      component="transformer").prepare()
    build_profile(timed(33.0))
    explicit = _NamedProfile(nn.Linear(2, 2), config(vmm_resident_weight_budget_gb=0.625), component="transformer")
    assert explicit.prepare().vmm_resident_weight_budget_gb == 0.625
    assert explicit.report["status"] == "explicit"


@pytest.mark.parametrize("floor_s, ceiling_s, extra_gib, spreads, decision", [
    (33.0, 32.9, 3.0, (0.01, 0.01), "floor"),      # 0.1 % per GiB < 0.5 %.
    (33.0, 32.4, 3.0, (0.05, 0.05), "ceiling"),    # 0.61 % per GiB, above noise.
    (33.0, 32.4, 3.0, (0.70, 0.05), "floor"),      # Gain 0.6 s within 0.7 s noise.
    (33.0, 32.4, 3.0, (None, 0.05), "floor"),      # Noise unknown.
    (None, 30.0, 3.0, (0.05, 0.05), "floor"),      # Missing floor timing.
    (33.0, 30.0, 0.0, (0.05, 0.05), "floor"),      # No extra memory.
])
def test_decide_budget_threshold_and_noise(floor_s, ceiling_s, extra_gib, spreads, decision):
    result = profiles.decide_vmm_resident_budget(floor_s, ceiling_s, FLOOR, FLOOR + int(extra_gib * GiB), 0.005,
                                                 floor_spread_seconds=spreads[0], ceiling_spread_seconds=spreads[1])
    assert result["decision"] == decision


def test_the_measured_10s_case_would_need_a_gain_above_its_noise():
    # 2026-11-10 10 s numbers: 44.380 s floor vs 43.405 s at +2.42 GiB (0.908 %/GiB).
    kept = profiles.decide_vmm_resident_budget(44.380, 43.405, FLOOR, FLOOR + int(2.42 * GiB), 0.005,
                                               floor_spread_seconds=0.3, ceiling_spread_seconds=0.3)
    noisy = profiles.decide_vmm_resident_budget(44.380, 43.405, FLOOR, FLOOR + int(2.42 * GiB), 0.005,
                                                floor_spread_seconds=1.2, ceiling_spread_seconds=0.3)
    assert kept["decision"] == "ceiling" and noisy["reason"] == "gain_within_timing_noise"


def test_other_machine_or_component_does_not_reuse_the_profile(monkeypatch):
    build_profile(timed(33.0))
    monkeypatch.setattr(profiles, "_vmm_machine_identity", lambda config: {**MACHINE, "host_ram_gib": 32})
    other_ram = _NamedProfile(nn.Linear(2, 2), config(), component="transformer")
    assert other_ram.prepare().vmm_resident_weight_budget_gb == 0.0
    assert other_ram.report["status"] == "missing"
    monkeypatch.setattr(profiles, "_vmm_machine_identity", lambda config: dict(MACHINE))
    other_component = _NamedProfile(nn.Linear(2, 2), config(), component="text_encoder")
    assert other_component.prepare().vmm_resident_weight_budget_gb == 0.0


def test_unavailable_observations_save_without_recommendation():
    profile, _, report = build_profile(timed(33.0, device_budget_bytes=None))
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
