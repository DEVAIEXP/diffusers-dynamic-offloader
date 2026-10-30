import os
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from diffusers_dynamic_offloader import (
    DynamicOffloadConfig,
    enable_offload,
    from_pretrained_with_dynamic_offload,
)
from diffusers_dynamic_offloader.profiles import _NamedProfile, load_dynamic_offload_profile


class NamedProfileTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.dict(os.environ, {"DDO_PROFILE_DIR": directory}))
        self.directory = directory
        self.stack.enter_context(patch("torch.cuda.synchronize"))
        self.stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
        self.stack.enter_context(patch("torch.cuda.memory_reserved", return_value=int(0.5 * 1024**3)))
        self.stack.enter_context(patch("torch.cuda.max_memory_reserved", return_value=int(5.5 * 1024**3)))
        self.stack.enter_context(patch("torch.cuda.mem_get_info", return_value=(7 * 1024**3, 8 * 1024**3)))
        # Capacity profiles are CPU contracts. An unmocked device-property
        # query would silently initialize CUDA on GPU-equipped machines.
        self.stack.enter_context(patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=8 * 1024**3)))

    def enable(self, **kwargs):
        return enable_offload(nn.Linear(2, 2), apply_hook=False, use_environment=False,
                              profile_name="example", **kwargs)

    def test_calibration_saves_and_next_setup_applies(self):
        result = self.enable(build_profile=True, resident_module_budget_gb=4)
        self.assertEqual(result.settings.config.resident_module_budget_gb, 0)
        with result.profile_run() as report:
            pass
        self.assertEqual(report["status"], "saved")
        self.assertEqual(report["resident_budget_gb"], 1.0)
        reused = self.enable()
        self.assertEqual(reused.settings.config.profile_resident_module_budget_gb, 1.0)
        manual = self.enable(resident_module_budget_gb=2)
        self.assertEqual(manual.settings.config.resident_module_budget_gb, 2)
        self.assertFalse(manual.settings.config._profile_budget_applied)

    def test_failure_does_not_write_profile(self):
        result = self.enable(build_profile=True)
        with self.assertRaisesRegex(RuntimeError, "inference failed"), result.profile_run():
            raise RuntimeError("inference failed")
        self.assertEqual(os.listdir(self.directory), [])

    def test_sdnq_low_baseline_saves_a_bounded_probe_then_refines_it(self):
        config = DynamicOffloadConfig(
            execution_mode="sdnq_runtime",
            execution_device="cuda",
            profile_name="sdnq-example",
            build_profile=True,
            # Match one_shot_fast: zero means no caller cap, so SDNQ uses
            # its safe library ceiling for a low-baseline probe.
            max_resident_module_budget_gb=0,
            profile_vram_headroom_gb=1,
        )
        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(total_memory=8 * 1024**3),
        ), patch("torch.cuda.max_memory_reserved", return_value=4 * 1024**3):
            profile = _NamedProfile(nn.Linear(2, 2), config)
            prepared = profile.prepare()
            with profile.measure():
                pass
        self.assertEqual(prepared.resident_module_budget_gb, 0)
        saved = load_dynamic_offload_profile(profile.context)
        self.assertEqual(saved["recommendation"]["profile_resident_module_budget_gb"], 6)
        self.assertEqual(saved["recommendation"]["sdnq_probe_resident_budget_gb"], 6)

        probe_config = DynamicOffloadConfig(
            execution_mode="sdnq_runtime",
            execution_device="cuda",
            profile_name="sdnq-example",
            profile_vram_headroom_gb=1,
        )
        with patch("torch.cuda.max_memory_reserved", return_value=int(7.7 * 1024**3)):
            probe = _NamedProfile(nn.Linear(2, 2), probe_config)
            prepared_probe = probe.prepare()
            with probe.measure():
                pass
        self.assertEqual(prepared_probe.resident_module_budget_gb, 6)
        refined = load_dynamic_offload_profile(probe.context)
        self.assertAlmostEqual(refined["recommendation"]["profile_resident_module_budget_gb"], 4.8, places=3)
        self.assertNotIn("sdnq_probe_resident_budget_gb", refined["recommendation"])

    def test_sdnq_high_baseline_uses_only_the_safe_residual_budget(self):
        config = DynamicOffloadConfig(
            execution_mode="sdnq_runtime",
            execution_device="cuda",
            profile_name="sdnq-high-baseline",
            build_profile=True,
            profile_vram_headroom_gb=1,
        )
        with patch("torch.cuda.max_memory_reserved", return_value=int(5.67 * 1024**3)):
            profile = _NamedProfile(nn.Linear(2, 2), config)
            profile.prepare()
            with profile.measure():
                pass
        saved = load_dynamic_offload_profile(profile.context)
        self.assertAlmostEqual(saved["recommendation"]["profile_resident_module_budget_gb"], 0.83, places=2)
        self.assertNotIn("sdnq_probe_resident_budget_gb", saved["recommendation"])

    def test_vmm_profile_records_capacity_without_changing_the_vmm_budget(self):
        config = DynamicOffloadConfig(
            execution_mode="vmm_fast_runtime",
            execution_device="cuda",
            profile_name="vmm-example",
            build_profile=True,
            vmm_resident_weight_budget_gb=4,
        )
        profile = _NamedProfile(nn.Linear(2, 2), config)
        prepared = profile.prepare()
        profile.attach_hook(
            SimpleNamespace(
                vmm_runtime_stats=lambda: {
                    "fast_virtual_bytes": 16 * 1024**2,
                    "fast_page_bytes": 2 * 1024**2,
                    "fast_resident_bytes": 4 * 1024**2,
                    "fast_file_backed_bindings": 2,
                    "fast_file_backed_bytes": 64,
                    "fast_cpu_source_bindings": 0,
                    "fast_faults": 3,
                    "fast_signature_hits": 1,
                    "fast_uploads": 2,
                    "fast_temporary_tensors": 0,
                    "fast_temporary_bytes": 0,
                }
            )
        )

        with profile.measure() as report:
            pass

        self.assertEqual(prepared.vmm_resident_weight_budget_gb, 4)
        self.assertEqual(report["status"], "saved")
        saved = load_dynamic_offload_profile(profile.context)
        observation = saved["observations"][-1]
        self.assertEqual(observation["profile_measurement_kind"], "vmm_capacity")
        self.assertEqual(observation["vmm"]["virtual_bytes"], 16 * 1024**2)
        self.assertEqual(observation["vmm"]["cpu_source_bindings"], 0)

    def test_different_names_are_independent(self):
        with self.enable(build_profile=True).profile_run():
            pass
        other = enable_offload(nn.Linear(2, 2), apply_hook=False, use_environment=False,
                               profile_name="other")
        self.assertFalse(other.settings.config._profile_budget_applied)

    def test_vmm_components_have_independent_capacity_observations(self):
        config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime",
                                     execution_device="cuda", profile_name="shared-name",
                                     build_profile=True, vmm_resident_weight_budget_gb=4)
        profiles = []
        for component, size in (("transformer", 1024), ("text_encoder", 2048)):
            profile = _NamedProfile(nn.Linear(2, 2), config, component=component)
            self.assertIs(profile.prepare(), config)
            profile.attach_hook(SimpleNamespace(vmm_runtime_stats=lambda size=size: {
                "fast_virtual_bytes": size, "fast_page_bytes": 512,
                "fast_file_backed_bindings": 2, "fast_cpu_source_bindings": 0,
            }))
            with profile.measure():
                pass
            profiles.append(profile)
        self.assertNotEqual(profiles[0].context, profiles[1].context)
        for profile, size in zip(profiles, (1024, 2048)):
            saved = load_dynamic_offload_profile(profile.context)
            self.assertEqual(saved["observations"][-1]["vmm"]["virtual_bytes"], size)
            self.assertEqual(profile.context["identity_version"], 3)
            self.assertEqual(config.vmm_resident_weight_budget_gb, 4)

    def test_generic_vmm_and_linear_profile_identities_are_preserved(self):
        for mode in ("vmm_fast_runtime", "linear_runtime", "sdnq_runtime"):
            config = DynamicOffloadConfig(execution_mode=mode, profile_name="identity")
            generic = _NamedProfile(nn.Linear(2, 2), config)
            self.assertEqual(generic.context["identity_version"], 2)
            self.assertNotIn("component", generic.context)
            if mode != "vmm_fast_runtime":
                component = _NamedProfile(nn.Linear(2, 2), config, component="text_encoder")
                self.assertEqual(component.context, generic.context)

    def test_enable_offload_passes_component_to_vmm_profile(self):
        config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime", profile_name="component")
        with patch("diffusers_dynamic_offloader.profiles._NamedProfile", wraps=_NamedProfile) as factory:
            enable_offload(nn.Linear(2, 2), component="text_encoder", route="dynamic_offload",
                           config=config, apply_hook=False, use_environment=False)
        self.assertEqual(factory.call_args.kwargs["component"], "text_encoder")

    def test_vmm_component_profile_refuses_blank_identity(self):
        config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            _NamedProfile(nn.Linear(2, 2), config, component=" ")

    def test_loader_consumes_profile_options(self):
        calls = []

        def loader(path, **kwargs):
            calls.append(kwargs)
            return nn.Linear(2, 2)

        with patch("diffusers_dynamic_offloader.dynamic_offload.apply_dynamic_offload"):
            loaded = from_pretrained_with_dynamic_offload(
                "example", model_loader=loader, apply_dynamic=True,
                dynamic_offload_config=DynamicOffloadConfig(execution_mode="linear_runtime"),
                profile_name="example", build_profile=True,
            )
        self.assertEqual(calls, [{}])
        with loaded.profile_run() as report:
            pass
        self.assertEqual(report["status"], "saved")
