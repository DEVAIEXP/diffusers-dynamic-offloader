import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch.nn as nn

from diffusers_dynamic_offloader import (
    DynamicOffloadConfig,
    enable_offload,
    from_pretrained_with_dynamic_offload,
)


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

    def test_different_names_are_independent(self):
        with self.enable(build_profile=True).profile_run():
            pass
        other = enable_offload(nn.Linear(2, 2), apply_hook=False, use_environment=False,
                               profile_name="other")
        self.assertFalse(other.settings.config._profile_budget_applied)

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
