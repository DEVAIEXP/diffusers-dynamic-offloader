import unittest

import torch.nn as nn

from diffusers_dynamic_offloader.dynamic_offload import (
    DDO_PRESETS,
    DynamicOffloadConfig,
    enable_diffusers_group_offload,
    enable_dynamic_offload,
    format_dynamic_offload_presets,
    get_dynamic_offload_presets,
    load_dynamic_offload_settings_from_env,
    purge_windows_standby_cache,
    purge_windows_standby_cache_event,
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

    def test_windows_standby_helpers_are_exported(self):
        self.assertTrue(callable(purge_windows_standby_cache))
        self.assertTrue(callable(purge_windows_standby_cache_event))

    def test_format_presets_lists_default_mapping(self):
        formatted = format_dynamic_offload_presets(default_preset="auto", running_on_wsl=False)
        self.assertIn("auto -> one_shot_fast", formatted)
        self.assertIn("one_shot_fast", formatted)
        self.assertIn("diffusers_offload_compat", formatted)


if __name__ == "__main__":
    unittest.main()
