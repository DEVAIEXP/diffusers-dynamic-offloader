"""CPU-only SDNQ preparation contracts; pinning is simulated, never registered."""

import re
import unittest
from unittest.mock import patch

import torch
from torch import nn

from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadConfig, DynamicOffloadHook
from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics


class PackedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, dtype=torch.uint8), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(2), requires_grad=False)
        self.sdnq_dequantizer = object()
        self.forward_func = object()

    def forward(self, value):
        return value + 1


class SdnqPreparationTests(unittest.TestCase):
    def setUp(self):
        self.pinned = set()
        self.pin_calls = []
        self.fail_pin = False

        def pin(tensor):
            self.pin_calls.append(tensor.numel() * tensor.element_size())
            if self.fail_pin:
                raise RuntimeError("simulated registration failure")
            result = tensor.clone()
            self.pinned.add(result.data_ptr())
            return result

        for mock in (
            patch.object(torch.Tensor, "pin_memory", pin),
            patch.object(torch.Tensor, "is_pinned", lambda tensor: tensor.data_ptr() in self.pinned),
            patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA must not initialize")),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def prepare(self, layers=None, **settings):
        module = nn.Sequential(*(layers or [PackedLayer(), PackedLayer()]))
        hook = DynamicOffloadHook(DynamicOffloadConfig(
            execution_mode="sdnq_runtime", execution_device="cpu", offload_device="cpu",
            auto_budget_policy="off", pin_cpu_memory=True, **settings,
        ))
        hook._prepare_sdnq_runtime(module)
        self.assertEqual(module(torch.zeros(1)).item(), len(module))
        return hook

    def test_already_cpu_parameters_are_pinned_and_counted(self):
        hook = self.prepare()
        self.assertEqual(self.pin_calls, [8, 8, 8, 8])
        self.assertEqual(hook.state.sdnq_pinned_parameter_bytes, 32)
        self.assertEqual(hook.state.sdnq_pinned_parameter_count, 4)

    def test_explicit_budget_is_not_snapped_to_full(self):
        hook = self.prepare(pin_weight_budget_gb=31 / 1024**3)
        self.assertEqual(hook.state.sdnq_pinned_parameter_bytes, 24)
        self.assertEqual(hook.state.sdnq_unpinned_parameter_bytes, 8)

    def test_ratio_and_maximum_are_both_respected(self):
        hook = self.prepare(pin_weight_budget_ratio=0.5, max_pin_weight_budget_gb=8 / 1024**3)
        self.assertEqual(hook.state.sdnq_pinned_parameter_bytes, 8)
        self.assertEqual(hook.state.sdnq_unpinned_parameter_bytes, 24)

    def test_fallback_counts_remaining_parameters_without_retrying(self):
        self.fail_pin = True
        hook = self.prepare(allow_pin_memory_fallback=True)
        self.assertEqual(len(self.pin_calls), 1)
        self.assertEqual(hook.state.sdnq_unpinned_parameter_bytes, 32)
        self.assertEqual(hook.state.sdnq_unpinned_parameter_count, 4)

    def test_strict_registration_failure_is_not_silenced(self):
        self.fail_pin = True
        with self.assertRaisesRegex(RuntimeError, "simulated registration failure"):
            self.prepare(allow_pin_memory_fallback=False)

    def test_existing_pin_is_preserved_and_consumes_budget(self):
        layer = PackedLayer()
        self.pinned.add(layer.weight.data_ptr())
        hook = self.prepare(layers=[layer], pin_weight_budget_gb=4 / 1024**3)
        self.assertEqual(self.pin_calls, [])
        self.assertEqual(hook.state.sdnq_pinned_parameter_bytes, 8)
        self.assertEqual(hook.state.sdnq_unpinned_parameter_bytes, 8)
        self.assertEqual(hook.state.planner_decisions["sdnq_existing_pinned_over_budget_bytes"], 4)

    def test_shared_parameter_is_counted_once(self):
        first, second = PackedLayer(), PackedLayer()
        second.weight = first.weight
        hook = self.prepare(layers=[first, second])
        self.assertEqual(hook.state.sdnq_pinned_parameter_bytes, 24)
        self.assertEqual(hook.state.sdnq_pinned_parameter_count, 3)

    def test_disabled_pinning_still_accounts_for_cpu_parameters(self):
        module = nn.Sequential(PackedLayer())
        hook = DynamicOffloadHook(DynamicOffloadConfig(
            execution_mode="sdnq_runtime", execution_device="cpu", offload_device="cpu",
            auto_budget_policy="off", pin_cpu_memory=False,
        ))
        hook._prepare_sdnq_runtime(module)
        self.assertEqual(self.pin_calls, [])
        self.assertEqual(hook.state.sdnq_unpinned_parameter_bytes, 16)

    def test_ram_denial_does_not_mean_unlimited_pinning(self):
        module = nn.Sequential(PackedLayer())
        hook = DynamicOffloadHook(DynamicOffloadConfig(pin_cpu_memory=True))
        hook._skip_pin_weights = True
        hook._select_sdnq_parameters_to_pin(module, ())
        hook._move_sdnq_parameters_to_offload_device(module[0])
        self.assertEqual(self.pin_calls, [])
        self.assertEqual(hook.state.sdnq_unpinned_parameter_bytes, 16)

    def test_available_ram_headroom_limits_registration(self):
        hook = self.prepare(available_system_ram_gb=24 / 1024**3,
                            system_ram_headroom_gb=16 / 1024**3)
        self.assertEqual(hook.state.sdnq_pinned_parameter_bytes, 8)

    def test_skipped_and_resident_parameters_are_not_candidates(self):
        module = nn.Sequential(PackedLayer(), PackedLayer(), PackedLayer())
        hook = DynamicOffloadHook(DynamicOffloadConfig(pin_cpu_memory=True))
        hook._resident_module_names.add("1")
        hook._select_sdnq_parameters_to_pin(module, (re.compile("^0$"),))
        self.assertEqual(hook.state.planner_decisions["sdnq_candidate_parameter_bytes"], 16)
        self.assertEqual(hook._sdnq_parameters_to_pin,
                         {id(parameter) for parameter in module[2].parameters()})

    def auto_budget(self, mode="sdnq_runtime", **settings):
        module = nn.Sequential(PackedLayer(), PackedLayer(), nn.Linear(2, 2))
        hook = DynamicOffloadHook(DynamicOffloadConfig(
            execution_mode=mode, execution_device="cpu", offload_device="cpu",
            auto_budget_policy="balanced", pin_cpu_memory=True,
            system_ram_headroom_gb=0, **settings,
        ))
        with patch("diffusers_dynamic_offloader.dynamic_offload.get_cuda_total_vram_gb", return_value=8):
            hook._apply_auto_budget_policy(module, (), (), ())
        return module, hook

    def test_auto_pin_budget_uses_packed_parameters_not_dense_linears(self):
        module, hook = self.auto_budget(available_system_ram_gb=1)
        self.assertEqual(hook.pin_weight_budget_bytes, 32)
        hook._select_sdnq_parameters_to_pin(module, ())
        self.assertEqual(hook.state.planner_decisions["sdnq_candidate_parameter_bytes"], 32)
        self.assertEqual(hook.state.planner_decisions["sdnq_pin_parameter_budget_bytes"], 32)

    def test_auto_ram_admission_includes_all_packed_parameters(self):
        _, hook = self.auto_budget(available_system_ram_gb=24 / 1024**3)
        self.assertTrue(hook._skip_pin_weights)
        self.assertEqual(hook.pin_weight_budget_bytes, 0)
        self.assertEqual(hook.state.planner_decisions["auto_pin_weight_decision"], "skip_insufficient_ram")

    def test_auto_packed_budget_retains_explicit_maximum(self):
        _, hook = self.auto_budget(available_system_ram_gb=1, max_pin_weight_budget_gb=8 / 1024**3)
        self.assertEqual(hook.pin_weight_budget_bytes, 8)

    def test_dense_auto_budget_population_is_unchanged(self):
        _, hook = self.auto_budget(mode="linear_runtime", available_system_ram_gb=1)
        self.assertEqual(hook.pin_weight_budget_bytes, 16)

    def test_explicit_packed_pin_budget_is_not_recalculated(self):
        _, hook = self.auto_budget(available_system_ram_gb=1, pin_weight_budget_gb=8 / 1024**3)
        self.assertEqual(hook.pin_weight_budget_bytes, 8)

    def test_public_report_renders_actual_pin_accounting_and_ceiling(self):
        hook = self.prepare(pin_weight_budget_gb=8 / 1024**3)
        report = DynamicOffloadMetrics(route="dynamic_offload", runtime={"state": hook.state.as_dict()})
        self.assertIn("sdnq_cpu_parameters=", report.format_profile())
        self.assertIn("pin_ceiling_gib=", report.format_profile())
        self.assertIn("not GPU residency", report.format_profile())
        detached = report.as_dict()
        self.assertEqual(detached["runtime"]["state"]["sdnq_parameter_pinning"]["pinned_count"], 1)

    def test_historical_report_does_not_invent_a_pin_ceiling(self):
        report = DynamicOffloadMetrics(route="dynamic_offload", runtime={"state": {"module_count": 1}})
        self.assertNotIn("sdnq_cpu_parameters=", report.format_profile())

    def test_calibration_retains_zero_residents_with_corrected_pin_capacity(self):
        module = nn.Sequential(PackedLayer(), PackedLayer())
        hook = DynamicOffloadHook(DynamicOffloadConfig(
            execution_mode="sdnq_runtime", execution_device="cpu", offload_device="cpu",
            auto_budget_policy="balanced", pin_cpu_memory=True,
            available_system_ram_gb=1, system_ram_headroom_gb=0,
            resident_module_patterns=(r"\d+",), build_profile=True,
        ))
        with patch("diffusers_dynamic_offloader.dynamic_offload.get_cuda_total_vram_gb", return_value=8):
            hook._prepare_sdnq_runtime(module)
        self.assertEqual(hook.resident_module_budget_bytes, 0)
        self.assertEqual(hook.state.selected_resident_modules, [])
        self.assertEqual(hook.state.sdnq_pinned_parameter_bytes, 32)
        self.assertEqual(hook.state.planner_decisions["auto_resident_module_decision"], "profile_calibration")


if __name__ == "__main__":
    unittest.main()
