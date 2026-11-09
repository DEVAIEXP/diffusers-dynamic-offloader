"""Use-time fit of a saved VMM residency budget to the current device budget (CPU, no CUDA)."""
from dataclasses import replace
from types import SimpleNamespace

from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadConfig, DynamicOffloadHook
from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics
from diffusers_dynamic_offloader.profiles import recheck_vmm_profile_budget

GIB = 1024**3
MIB = 1024**2
MARGIN = 256 * MIB


def test_saved_budget_fits_unchanged():
    check = recheck_vmm_profile_budget(2 * GIB, 64 * MIB, 4 * GIB, MARGIN, 7 * GIB)
    assert check["status"] == "fits" and check["applied_bytes"] == 2 * GIB


def test_shrunk_device_budget_reduces_by_the_deficit():
    # Measured LTX image case: 7.030 GiB device budget now, 4.629 GiB non-weight peak.
    saved, non_weight, now = 2369781760, 4970250240, 7548805120
    check = recheck_vmm_profile_budget(saved, 64 * MIB, non_weight, MARGIN, now)
    assert check["status"] == "reduced"
    assert check["applied_bytes"] == now - MARGIN - non_weight < saved


def test_never_below_the_floor_and_never_raised():
    floor = 629 * MIB
    check = recheck_vmm_profile_budget(3 * GIB, floor, 6 * GIB, MARGIN, 6 * GIB)
    assert check["status"] == "device_budget_below_floor" and check["applied_bytes"] == floor
    floor_decision = recheck_vmm_profile_budget(floor, floor, 6 * GIB, MARGIN, 6 * GIB)
    assert floor_decision["applied_bytes"] == floor
    roomy = recheck_vmm_profile_budget(floor, floor, 1 * GIB, MARGIN, 16 * GIB)
    assert roomy["status"] == "fits" and roomy["applied_bytes"] == floor


def test_missing_observations_keep_the_saved_budget():
    for now, non_weight in ((None, 4 * GIB), (7 * GIB, None)):
        check = recheck_vmm_profile_budget(2 * GIB, 64 * MIB, non_weight, MARGIN, now)
        assert check["status"] == "unavailable" and check["applied_bytes"] == 2 * GIB


def _hook(budget, non_weight, floor):
    config = replace(DynamicOffloadConfig(), _profile_budget_applied=True, _profile_floor_bytes=floor,
                     _profile_non_weight_peak_bytes=non_weight)
    return SimpleNamespace(config=config, vmm_resident_weight_budget_bytes=budget,
                           execution_device=SimpleNamespace(index=0),
                           state=SimpleNamespace(planner_decisions={}))


class _Library:
    def __init__(self, local_budget):
        self.local_budget = local_budget

    def windows_memory_info(self, ordinal):
        if self.local_budget is None:
            raise RuntimeError("query failed")
        return SimpleNamespace(local_budget_bytes=self.local_budget, local_usage_bytes=0)


def test_hook_applies_and_records_the_check(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    hook = _hook(2369781760, 4970250240, 64 * MIB)
    DynamicOffloadHook._recheck_vmm_profile_budget(hook, _Library(7548805120))
    check = hook.state.planner_decisions["vmm_profile_budget_check"]
    assert check["status"] == "reduced" and hook.vmm_resident_weight_budget_bytes == check["applied_bytes"]


def test_hook_keeps_the_budget_when_the_query_fails_or_off_windows(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    hook = _hook(2 * GIB, 4 * GIB, 64 * MIB)
    DynamicOffloadHook._recheck_vmm_profile_budget(hook, _Library(None))
    assert hook.vmm_resident_weight_budget_bytes == 2 * GIB
    assert hook.state.planner_decisions["vmm_profile_budget_check"]["status"] == "unavailable"
    monkeypatch.setattr("sys.platform", "linux")
    hook = _hook(2 * GIB, 4 * GIB, 64 * MIB)
    DynamicOffloadHook._recheck_vmm_profile_budget(hook, _Library(1 * GIB))
    assert hook.vmm_resident_weight_budget_bytes == 2 * GIB


def test_profile_line_shows_the_check():
    metrics = DynamicOffloadMetrics(route="dynamic_offload", configuration={}, setup={}, runtime={
        "vmm_residency_profile": {"name": "p", "status": "applied", "budget_check": {
            "status": "reduced", "saved_bytes": 2 * GIB, "applied_bytes": GIB, "device_budget_bytes": 7 * GIB}}})
    text = metrics.format_profile()
    assert "budget_check=reduced saved_gib=2.000 applied_gib=1.000 device_budget_now_gib=7.000" in text
