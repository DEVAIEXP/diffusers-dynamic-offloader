"""VMM residency profile report and first-forward shape check (stage 3), CPU only."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusers_dynamic_offloader import DynamicOffloadConfig
from diffusers_dynamic_offloader import profiles
from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadApplyResult
from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics
from diffusers_dynamic_offloader.profiles import _NamedProfile

GiB = 1024**3
MiB = 1024**2
SLOTS = 608174080


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DDO_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(profiles, "_vmm_machine_identity", lambda config: {"platform": "win32", "gpu_name": "g"})


def config(**overrides):
    values = dict(execution_mode="vmm_fast_runtime", execution_device="cuda", profile_name="w",
                  vmm_resident_weight_budget_gb=0.0)
    values.update(overrides)
    return DynamicOffloadConfig(**values)


def build(module, peak=int(5.24 * GiB)):
    profile = _NamedProfile(module, config(build_profile=True), component="transformer")
    profile.prepare()
    profile.attach_hook(SimpleNamespace(vmm_runtime_stats=lambda: {
        "fast_execution_buffer_capacity_bytes": SLOTS, "fast_resident_bytes": 40 * MiB,
        "fast_total_weight_resident_bytes": SLOTS + 40 * MiB, "fast_virtual_bytes": 22 * GiB},
        vmm_device_memory=lambda: {"local_budget_bytes": int(7.61 * GiB), "local_usage_bytes": peak}))
    with profile.measure():
        module(torch.zeros(3, 2))
    return profile


def use(module):
    profile = _NamedProfile(module, config(), component="transformer")
    prepared = profile.prepare()
    profile.attach_hook(SimpleNamespace())
    return profile, prepared


def test_shape_check_matches_or_flags_a_different_workload(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3 * GiB, 8 * GiB))
    module = nn.Linear(2, 2)
    build(module)
    same, _ = use(module)
    assert same.report["shape_check"] == "pending"
    module(torch.zeros(3, 2))
    assert same.report["shape_check"] == "match"
    other, _ = use(module)
    module(torch.zeros(7, 2))
    assert other.report["shape_check"] == "mismatch"
    assert not module._forward_pre_hooks  # One-shot checks removed themselves.


def test_missing_profile_reports_unavailable_shape_check():
    profile, prepared = use(nn.Linear(2, 2))
    assert profile.report["status"] == "missing" and profile.report["shape_check"] == "unavailable"
    assert prepared.vmm_resident_weight_budget_gb == 0.0


def test_metrics_carry_the_residency_profile_and_format_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3 * GiB, 8 * GiB))
    module = nn.Linear(2, 2)
    build(module)
    profile, _ = use(module)
    hook = SimpleNamespace(state=SimpleNamespace(planner_decisions={"vmm_resident_weight_budget_source": "profile"},
                                                 as_dict=lambda: {}),
                           vmm_runtime_stats=lambda: None, vmm_resident_weight_budget_bytes=2 * GiB)
    result = DynamicOffloadApplyResult(module=module, hook=hook, _profile=profile)
    payload = result.metrics_snapshot().as_dict()["runtime"]["vmm_residency_profile"]
    assert payload["status"] == "applied" and payload["fit_class"] == "fits_with_headroom"
    assert payload["budget_selection"] == "profile" and payload["effective_budget_bytes"] == 2 * GiB
    assert payload["budget_source"] == "dxgi_local_budget" and payload["margin_gb"] == 0.25
    text = result.metrics_snapshot().format_profile()
    assert "vmm_residency_profile=w status=applied fit=fits_with_headroom budget_gib=2.000" in text
    assert "selection=profile" in text


def test_does_not_fit_is_explained_and_unavailable_values_are_named():
    report = DynamicOffloadMetrics(route="dynamic_offload", runtime={"vmm_residency_profile": {
        "name": "w", "status": "applied", "fit_class": "does_not_fit", "effective_budget_bytes": None}})
    text = report.format_profile()
    assert "budget_gib=unavailable" in text and "does not fit the device budget" in text


def test_use_run_reports_its_own_peak_without_synchronizing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3 * GiB, 8 * GiB))
    module = nn.Linear(2, 2)
    build(module)  # Build peak 5.24 GiB.
    synchronizations = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: synchronizations.append(device))
    profile = _NamedProfile(module, config(), component="transformer")
    profile.prepare()
    usage = iter([int(4.2 * GiB)] * 1000)
    profile.attach_hook(SimpleNamespace(vmm_device_memory=lambda: {"local_budget_bytes": int(7.61 * GiB),
                                                                   "local_usage_bytes": next(usage)}))
    with profile.measure():
        module(torch.zeros(3, 2))
    assert synchronizations == []  # Use runs never synchronize the device.
    report = profile.residency_report()
    assert report["use_peak_usage_bytes"] == int(4.2 * GiB)
    assert report["build_peak_usage_bytes"] == int(5.24 * GiB)
    assert report["use_device_budget_bytes"] == int(7.61 * GiB)
    assert not module._forward_pre_hooks and not module._forward_hooks


def test_linear_routes_have_no_residency_profile_section():
    linear = _NamedProfile(nn.Linear(2, 2), DynamicOffloadConfig(execution_mode="linear_runtime", profile_name="w"))
    assert linear.residency_report() is None
