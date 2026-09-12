import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from diffusers_dynamic_offloader.dynamic_offload import (
    DDO_PRESETS,
    DynamicOffloadConfig,
    detect_quantized_backend_modules,
    enable_diffusers_group_offload,
    enable_dynamic_offload,
    enable_offload,
    enable_pipeline_offload,
    format_dynamic_offload_presets,
    get_dynamic_offload_presets,
    load_dynamic_offload_settings_from_env,
    maybe_purge_windows_standby_cache,
    purge_windows_standby_cache,
    purge_windows_standby_cache_event,
    remove_dynamic_offload,
    resolve_dynamic_offload_preset,
)


class DynamicOffloadPresetTests(unittest.TestCase):
    def test_auto_resolves_to_one_shot_fast(self):
        self.assertEqual(resolve_dynamic_offload_preset("auto", running_on_wsl=False), "one_shot_fast")
        self.assertEqual(resolve_dynamic_offload_preset("auto", running_on_wsl=True), "one_shot_fast")

    def test_invalid_preset_reports_canonical_env_name(self):
        with self.assertRaisesRegex(ValueError, "DDO_PRESET"):
            resolve_dynamic_offload_preset("windows_fast", running_on_wsl=False)

    def test_presets_do_not_include_removed_legacy_names(self):
        removed_aliases = {
            "compat",
            "windows_fast",
            "linux_native_fast",
            "linux_safe",
            "planner_balanced",
            "planner_slim_resident",
            "warm_server",
            "low_ram",
            "diffusers_group_offload",
            "group_offload_compat",
            "diffusers_leaf_group_offload",
            "leaf_offload_compat",
        }
        self.assertFalse(removed_aliases.intersection(DDO_PRESETS))

    def test_env_override_wins_over_preset_value(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={
                "DDO_PRESET": "one_shot_fast",
                "DDO_PIN_CPU_WORKERS": "9",
                "DDO_AVAILABLE_SYSTEM_RAM_GB": "64",
            },
        )
        self.assertEqual(settings.effective_preset, "one_shot_fast")
        self.assertEqual(settings.config.pin_cpu_workers, 9)
        self.assertEqual(settings.config.available_system_ram_gb, 64.0)

    def test_profile_headroom_is_a_regular_settings_value(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PROFILE_VRAM_HEADROOM_GB": "1.25"},
        )
        self.assertEqual(settings.config.profile_vram_headroom_gb, 1.25)
        self.assertEqual(settings.as_metrics()["dynamic_offload_profile_vram_headroom_gb"], 1.25)
        self.assertFalse(hasattr(settings.config, "auto_full_pin_resident_budget_gb"))

    def test_wsl_disables_pinned_memory_by_default(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=True,
            environ={
                "DDO_PRESET": "one_shot_fast",
                "DDO_AVAILABLE_SYSTEM_RAM_GB": "64",
            },
        )
        self.assertTrue(settings.requested_pin_cpu_memory)
        self.assertFalse(settings.effective_pin_cpu_memory)

    def test_preset_listing_is_a_copy(self):
        presets = get_dynamic_offload_presets()
        presets["one_shot_fast"]["DDO_PIN_CPU_WORKERS"] = "99"
        self.assertNotEqual(
            DDO_PRESETS["one_shot_fast"]["DDO_PIN_CPU_WORKERS"],
            "99",
        )

    def test_enable_offload_routes_by_component_preset(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={
                "DDO_PRESET": "one_shot_fast",
                "DDO_AVAILABLE_SYSTEM_RAM_GB": "64",
            },
        )
        transformer_result = enable_offload(
            nn.Linear(2, 2),
            settings=settings,
            component="transformer",
            apply_hook=False,
        )
        text_encoder_result = enable_offload(
            nn.Linear(2, 2),
            settings=settings,
            component="text_encoder",
            apply_hook=False,
        )
        self.assertEqual(transformer_result.route, "dynamic_offload")
        self.assertEqual(text_encoder_result.route, "diffusers_group_offload")

    def test_enable_offload_accepts_component_policy_overrides(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PRESET": "one_shot_fast"},
            component_policies={
                "transformer": {
                    "route": "diffusers_group_offload",
                    "offload_type": "block_level",
                    "num_blocks_per_group": 2,
                    "offload_stream": False,
                    "offload_record_stream": True,
                }
            },
        )
        result = enable_offload(
            nn.Linear(2, 2),
            settings=settings,
            component="transformer",
            apply_hook=False,
        )
        self.assertEqual(result.route, "diffusers_group_offload")
        self.assertEqual(result.event_payload["offload_type"], "block_level")
        self.assertEqual(result.event_payload["num_blocks_per_group"], 2)
        self.assertFalse(result.event_payload["use_stream"])
        self.assertTrue(result.event_payload["record_stream"])

    def test_enable_offload_explicit_route_overrides_component_policy(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PRESET": "one_shot_fast"},
            component_policies={"transformer": {"route": "diffusers_group_offload"}},
        )
        result = enable_offload(
            nn.Linear(2, 2),
            settings=settings,
            component="transformer",
            route="none",
            apply_hook=False,
        )
        self.assertEqual(result.route, "none")
        self.assertTrue(result.should_move_to_execution_device)

    def test_quantized_component_auto_routes_to_diffusers_group_offload(self):
        class FakeSdnqLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.sdnq_dequantizer = object()

        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PRESET": "one_shot_fast"},
        )
        result = enable_offload(
            FakeSdnqLayer(),
            settings=settings,
            component="transformer",
            apply_hook=False,
        )
        self.assertEqual(result.route, "diffusers_group_offload")
        self.assertEqual(
            result.event_payload["quantized_backend"]["sdnq_status"],
            "detected_forward_preserved",
        )

    def test_quantized_component_respects_off_preset(self):
        class FakeSdnqLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.sdnq_dequantizer = object()

        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PRESET": "off"},
        )
        result = enable_offload(
            FakeSdnqLayer(),
            settings=settings,
            component="transformer",
            apply_hook=False,
        )
        self.assertEqual(result.route, "none")
        self.assertTrue(result.should_move_to_execution_device)

    def test_enable_pipeline_offload_reuses_one_settings_object(self):
        class FakePipeline:
            def __init__(self):
                self.components = {
                    "text_encoder": nn.Linear(2, 2),
                    "transformer": nn.Linear(2, 2),
                    "scheduler": object(),
                }
                self.text_encoder = self.components["text_encoder"]
                self.transformer = self.components["transformer"]

        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PRESET": "one_shot_fast"},
        )
        results = enable_pipeline_offload(
            FakePipeline(),
            settings=settings,
            apply_hook=False,
        )
        self.assertEqual(set(results), {"text_encoder", "transformer"})
        self.assertEqual(results["text_encoder"].route, "diffusers_group_offload")
        self.assertEqual(results["transformer"].route, "dynamic_offload")
        self.assertIs(results["text_encoder"].settings, settings)
        self.assertIs(results["transformer"].settings, settings)

    def test_enable_offload_routes_diffusers_compat_transformer_to_group_offload(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PRESET": "diffusers_offload_compat"},
        )
        result = enable_offload(
            nn.Linear(2, 2),
            settings=settings,
            component="transformer",
            apply_hook=False,
        )
        self.assertEqual(result.route, "diffusers_group_offload")
        self.assertEqual(result.event_payload["offload_type"], "block_level")

    def test_warm_process_routes_text_encoder_to_group_offload(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PRESET": "warm_process"},
        )
        result = enable_offload(
            nn.Linear(2, 2),
            settings=settings,
            component="text_encoder",
            apply_hook=False,
        )
        self.assertEqual(result.route, "diffusers_group_offload")
        self.assertTrue(result.event_payload["use_stream"])


    def test_low_ram_safe_routes_text_encoder_to_non_streaming_group_offload(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=True,
            environ={"DDO_PRESET": "low_ram_safe"},
        )
        result = enable_offload(
            nn.Linear(2, 2),
            settings=settings,
            component="text_encoder",
            apply_hook=False,
        )
        self.assertEqual(result.route, "diffusers_group_offload")
        self.assertFalse(result.event_payload["use_stream"])
        self.assertFalse(result.event_payload["record_stream"])

    def test_enable_offload_disables_group_stream_on_wsl(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=True,
            environ={"DDO_PRESET": "one_shot_fast"},
        )
        result = enable_offload(
            nn.Linear(2, 2),
            settings=settings,
            component="text_encoder",
            apply_hook=False,
        )
        self.assertEqual(result.route, "diffusers_group_offload")
        self.assertFalse(result.event_payload["use_stream"])
        self.assertFalse(result.event_payload["record_stream"])


    def test_enable_diffusers_group_offload_reports_payload_without_applying_hook(self):
        events = []
        module = nn.Linear(2, 2)
        result = enable_diffusers_group_offload(
            module,
            offload_type="block_level",
            use_stream=False,
            record_stream=True,
            num_blocks_per_group=2,
            apply_hook=False,
            record_event=lambda name, seconds, **payload: events.append((name, seconds, payload)),
        )
        self.assertIs(result.module, module)
        self.assertEqual(result.event_payload["offload_type"], "block_level")
        self.assertEqual(result.event_payload["num_blocks_per_group"], 2)
        self.assertEqual(events[0][0], "setup_diffusers_group_offload")
        self.assertFalse(events[0][2]["use_stream"])

    def test_enable_dynamic_offload_resolves_preset_without_applying_hook(self):
        module = nn.Linear(2, 2)
        result = enable_dynamic_offload(
            module,
            preset="one_shot_fast",
            use_environment=False,
            apply_hook=False,
            available_system_ram_gb=64.0,
            show_profile=True,
        )
        self.assertIs(result.module, module)
        self.assertIsNone(result.hook)
        self.assertEqual(result.settings.effective_preset, "one_shot_fast")
        self.assertTrue(result.settings.config.show_profile)
        self.assertEqual(result.settings.config.available_system_ram_gb, 64.0)

    def test_enable_dynamic_offload_explicit_config_wins(self):
        events = []
        module = nn.Linear(2, 2)
        config = DynamicOffloadConfig(execution_mode="plan", pin_cpu_memory=False)
        result = enable_dynamic_offload(
            module,
            preset="off",
            config=config,
            use_environment=False,
            record_event=lambda name, seconds, **payload: events.append((name, seconds, payload)),
        )
        self.assertIsNotNone(result.hook)
        self.assertTrue(result.should_move_to_execution_device)
        self.assertEqual(events[0][0], "setup_dynamic_offload")
        self.assertEqual(events[0][2]["execution_mode"], "plan")

    def test_balanced_auto_prefers_full_pin_without_resident_modules_when_ram_fits(self):
        module = nn.Sequential(*[nn.Linear(512, 512, bias=False) for _ in range(4)])
        config = DynamicOffloadConfig(
            execution_device="cpu",
            offload_device="cpu",
            execution_mode="linear_runtime",
            pin_cpu_memory=True,
            auto_budget_policy="balanced",
            available_system_ram_gb=1.0,
            system_ram_headroom_gb=0.0,
            resident_module_patterns=(r"\d+",),
        )
        with patch("diffusers_dynamic_offloader.dynamic_offload.get_cuda_total_vram_gb", return_value=0.0001):
            result = enable_dynamic_offload(module, config=config, use_environment=False)

        decisions = result.hook.state.planner_decisions
        self.assertEqual(decisions["auto_pin_weight_decision"], "full_pin_zero_resident")
        self.assertEqual(decisions["auto_resident_module_decision"], "zero_for_full_pin")
        self.assertEqual(decisions["resolved_pin_weight_budget_mode"], "snap_to_full_pin")
        self.assertFalse(result.hook.state.selected_resident_modules)
        remove_dynamic_offload(module)

    def test_packed_linear_keeps_original_forward(self):
        class PackedLinear(nn.Linear):
            def __init__(self):
                nn.Module.__init__(self)
                self.weight = nn.Parameter(torch.ones(1, 2, 2))
                self.bias = None
                self.forward_called = False

            def forward(self, input):
                self.forward_called = True
                return input + 1

        class Packed2DLinear(nn.Linear):
            def __init__(self):
                super().__init__(2, 2)
                self.weight = nn.Parameter(torch.ones(9, 16))
                self.forward_called = False

            def forward(self, input):
                self.forward_called = True
                return input + 2

        module = nn.Sequential(PackedLinear(), Packed2DLinear())
        config = DynamicOffloadConfig(
            execution_device="cpu",
            offload_device="cpu",
            execution_mode="linear_runtime",
            pin_cpu_memory=False,
            auto_budget_policy="off",
        )
        result = enable_dynamic_offload(module, config=config, use_environment=False)
        output = module(torch.zeros(1, 2))

        self.assertTrue(module[0].forward_called)
        self.assertTrue(module[1].forward_called)
        self.assertEqual(output.tolist(), [[3.0, 3.0]])
        self.assertEqual(result.hook.state.patched_module_count, 0)
        self.assertEqual(result.hook.state.planner_decisions["unsupported_linear_module_count"], 2)
        remove_dynamic_offload(module)


    def test_sdnq_detection_reports_forward_preserved(self):
        class FakeSdnqLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.sdnq_dequantizer = object()

        result = detect_quantized_backend_modules(FakeSdnqLayer())
        self.assertEqual(result["sdnq_status"], "detected_forward_preserved")
        self.assertEqual(result["sdnq_module_count"], 1)


    def test_windows_standby_helpers_are_exported(self):
        self.assertTrue(callable(purge_windows_standby_cache))
        self.assertTrue(callable(purge_windows_standby_cache_event))
        self.assertTrue(callable(maybe_purge_windows_standby_cache))

    def test_maybe_purge_accepts_preset_without_settings(self):
        result = maybe_purge_windows_standby_cache(
            "before_run",
            preset="off",
            print_message=False,
        )
        self.assertEqual(result["reason"], "disabled")

    def test_maybe_purge_is_disabled_by_default_for_off_preset(self):
        settings = load_dynamic_offload_settings_from_env(
            running_on_wsl=False,
            environ={"DDO_PRESET": "off"},
        )
        result = maybe_purge_windows_standby_cache(settings, "before_run", print_message=False)
        self.assertEqual(result["reason"], "disabled")

    def test_format_presets_lists_default_mapping(self):
        formatted = format_dynamic_offload_presets(default_preset="auto", running_on_wsl=False)
        self.assertIn("auto -> one_shot_fast", formatted)
        self.assertIn("one_shot_fast", formatted)
        self.assertIn("diffusers_offload_compat", formatted)


if __name__ == "__main__":
    unittest.main()
