"""VMM residency profile observation (stage 1) without CUDA: fake device queries."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusers_dynamic_offloader import DynamicOffloadConfig
from diffusers_dynamic_offloader import profiles
from diffusers_dynamic_offloader.profiles import _NamedProfile, load_dynamic_offload_profile

GiB = 1024**3


@pytest.fixture(autouse=True)
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("DDO_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)


def stats(**overrides):
    values = {
        "fast_virtual_bytes": 16 * GiB, "fast_page_bytes": 2 * 1024**2, "fast_resident_bytes": 40 * 1024**2,
        "fast_file_backed_bindings": 2, "fast_file_backed_bytes": 64, "fast_cpu_source_bindings": 0,
        "fast_faults": 3, "fast_signature_hits": 1, "fast_uploads": 2, "fast_temporary_tensors": 0,
        "fast_temporary_bytes": 0, "fast_execution_buffer_capacity_bytes": 608174080,
        "fast_total_weight_resident_bytes": 648174080,
    }
    values.update(overrides)
    return values


def build(module, hook, *, run=None):
    config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime", execution_device="cuda",
                                  profile_name="ltx-video-1280x704-8s", build_profile=True,
                                  vmm_resident_weight_budget_gb=0.0)
    profile = _NamedProfile(module, config, component="transformer")
    assert profile.prepare() is config  # Stage 1 never changes the configured budget.
    profile.attach_hook(hook)
    with profile.measure() as report:
        if run is not None:
            run()
    assert report["status"] == "saved"
    return load_dynamic_offload_profile(profile.context)["observations"][-1]["vmm_residency"]


def test_windows_build_records_wddm_budget_peak_floor_and_first_forward_shapes(monkeypatch):
    usage = iter([2 * GiB, 5 * GiB, 4 * GiB] + [3 * GiB] * 1000)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3 * GiB, 8 * GiB))
    hook = SimpleNamespace(vmm_runtime_stats=lambda: stats(),
                           vmm_device_memory=lambda: {"local_budget_bytes": int(7.61 * GiB),
                                                      "local_usage_bytes": next(usage)})
    module = nn.Linear(4, 2)
    observation = build(module, hook, run=lambda: (module(torch.zeros(3, 4)), module(torch.zeros(5, 4))))
    assert observation["schema_version"] == profiles.VMM_RESIDENCY_OBSERVATION_SCHEMA_VERSION
    assert observation["budget_source"] == "dxgi_local_budget"
    assert observation["device_budget_bytes"] == int(7.61 * GiB)
    assert observation["peak_usage_bytes"] == observation["peak_local_usage_bytes"] == 5 * GiB
    assert observation["peak_driver_used_bytes"] == 5 * GiB
    assert observation["device_total_bytes"] == 8 * GiB
    assert observation["execution_slots_bytes"] == 608174080
    assert observation["arena_resident_bytes"] == 40 * 1024**2
    assert observation["first_forward_seen"] is True
    assert observation["input_shapes"] == {"args": [[3, 4]], "kwargs": None}  # First forward only.
    assert len(observation["forward_seconds"]) == 2
    assert observation["steady_forward_seconds"] == observation["forward_seconds"][1]  # After the first.
    assert not module._forward_pre_hooks and not module._forward_hooks  # Observation hooks removed.


def test_the_lowest_wddm_budget_of_the_region_is_recorded(monkeypatch):
    # WDDM lowers the process budget under the process's own load.
    budgets = iter([int(7.086 * GiB), int(7.030 * GiB)] + [int(7.05 * GiB)] * 1000)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3 * GiB, 8 * GiB))
    hook = SimpleNamespace(vmm_runtime_stats=lambda: stats(),
                           vmm_device_memory=lambda: {"local_budget_bytes": next(budgets),
                                                      "local_usage_bytes": 4 * GiB})
    observation = build(nn.Linear(2, 2), hook)
    assert observation["device_budget_bytes"] == int(7.030 * GiB)


def test_without_wddm_the_device_total_is_the_budget_source(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (6 * GiB, 8 * GiB))
    observation = build(nn.Linear(2, 2), SimpleNamespace(vmm_runtime_stats=lambda: stats()))
    assert observation["budget_source"] == "cuda_device_total"
    assert observation["device_budget_bytes"] == 8 * GiB
    assert observation["peak_usage_bytes"] == 2 * GiB
    assert observation["peak_local_usage_bytes"] is None
    assert observation["first_forward_seen"] is False and observation["input_shapes"] is None
    assert observation["forward_seconds"] == [] and observation["steady_forward_seconds"] is None


def test_component_root_capacity_is_observed(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device=None: (6 * GiB, 8 * GiB))
    module = nn.Linear(2, 2)
    hook = SimpleNamespace(vmm_runtime_stats=lambda: stats(fast_component_lifetime_capacity_bytes=2 * 1024**2))
    assert build(module, hook)['component_lifetime_capacity_bytes'] == 2 * 1024**2


def test_unavailable_observations_stay_null_not_zero(monkeypatch):
    def unavailable(device=None):
        raise RuntimeError("no device query")

    monkeypatch.setattr(torch.cuda, "mem_get_info", unavailable)
    minimal = {key: value for key, value in stats().items()
               if key not in ("fast_execution_buffer_capacity_bytes", "fast_total_weight_resident_bytes")}
    observation = build(nn.Linear(2, 2), SimpleNamespace(vmm_runtime_stats=lambda: minimal,
                                                         vmm_device_memory=lambda: None))
    for name in ("device_budget_bytes", "budget_source", "peak_usage_bytes", "device_total_bytes",
                 "execution_slots_bytes", "total_weight_resident_bytes"):
        assert observation[name] is None, name


def test_failed_region_removes_the_hook_and_saves_nothing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (6 * GiB, 8 * GiB))
    module = nn.Linear(2, 2)
    config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime", execution_device="cuda",
                                  profile_name="failing", build_profile=True, vmm_resident_weight_budget_gb=0.0)
    profile = _NamedProfile(module, config, component="transformer")
    profile.prepare()
    profile.attach_hook(SimpleNamespace(vmm_runtime_stats=lambda: stats()))
    with pytest.raises(RuntimeError, match="forward failed"):
        with profile.measure():
            raise RuntimeError("forward failed")
    assert profile.report["status"] == "failed"
    assert not module._forward_pre_hooks
    assert load_dynamic_offload_profile(profile.context) is None


def test_tensor_shapes_are_recursive_and_ignore_non_tensors():
    value = {"hidden": torch.zeros(1, 2), "nested": [torch.zeros(3), 7], "flag": True}
    assert profiles._tensor_shapes(value) == {"hidden": [1, 2], "nested": [[3]]}
