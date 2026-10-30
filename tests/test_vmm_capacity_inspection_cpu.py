import gc
import json
import unittest
import weakref
from unittest.mock import patch

import torch

from diffusers_dynamic_offloader import inspect_vmm_component
from ddo_vmm_allocator.model_arena_adapter import _discover_linear_layout


class CapacityInspectionTests(unittest.TestCase):
    def model(self):
        with torch.device("meta"):
            root = torch.nn.Module()
            root.embedding = torch.nn.Embedding(11, 7, dtype=torch.bfloat16)
            root.blocks = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(7, 7, dtype=torch.bfloat16),
                                    torch.nn.LayerNorm(7, dtype=torch.bfloat16))])
            root.head = torch.nn.Linear(7, 11, bias=False, dtype=torch.bfloat16)
            root.head.weight = root.embedding.weight
            root.register_buffer("positions", torch.arange(5))
        return root

    def test_exact_selector_alignment_shared_owners_and_buffers(self):
        root = self.model()
        with patch("torch.cuda.init", side_effect=AssertionError("CUDA")), \
             patch("ddo_vmm_allocator.model_arena_adapter.NativeVmmLibrary", side_effect=AssertionError("native load")), \
             patch("torch.cuda.get_device_properties", side_effect=AssertionError("CUDA")), \
             patch("torch.Tensor.data_ptr", side_effect=AssertionError("payload access")), \
             patch("torch.Tensor.to", side_effect=AssertionError("copy")):
            report = inspect_vmm_component(root, component="text_encoder")
        layout = report.as_dict()["setup"]["capacity_inspection"]
        self.assertEqual(layout["selected_aligned_payload_bytes"], _discover_linear_layout(root, ".*")[2])
        self.assertEqual(layout["selected_module_count"], 2)
        self.assertEqual(layout["unique_parameter_bytes"], 294)
        self.assertEqual(layout["selected_unique_parameter_bytes"], 266)
        self.assertEqual(layout["nonselected_unique_parameter_bytes"], 182)
        self.assertEqual(layout["shared_selected_nonselected_bytes"], 154)
        self.assertEqual(layout["unique_buffer_bytes"], 40)
        tied = next(group for group in layout["parameters"] if "head.weight" in group["names"])
        self.assertEqual(tied["nonselected_names"], ["embedding.weight"])
        self.assertEqual(tied["selected_names"], ["head.weight"])
        self.assertEqual(layout["selected_modules"][0]["execution_unit"], "blocks.0")
        self.assertIn("not execution support", report.format_profile())
        json.dumps(report.as_dict())

    def test_report_does_not_retain_module_or_parameters(self):
        root = self.model()
        ref = weakref.ref(root)
        weight = weakref.ref(root.head.weight)
        report = inspect_vmm_component(root)
        del root
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(weight())
        first = report.as_dict()
        first["setup"]["capacity_inspection"]["parameters"].clear()
        self.assertTrue(report.as_dict()["setup"]["capacity_inspection"]["parameters"])

    def test_pattern_matches_attachment_and_unselected_head_remains_unselected(self):
        root = self.model()
        report = inspect_vmm_component(root, module_pattern=r"^blocks$").as_dict()
        layout = report["setup"]["capacity_inspection"]
        self.assertEqual(layout["selected_module_count"], 1)
        self.assertEqual(layout["selected_aligned_payload_bytes"], _discover_linear_layout(root, r"^blocks$")[2])
        self.assertEqual(layout["shared_selected_nonselected_bytes"], 0)

    def test_empty_selection_and_root_linear(self):
        layout = inspect_vmm_component(self.model(), module_pattern="no_match").as_dict()["setup"]["capacity_inspection"]
        self.assertEqual(layout["selected_aligned_payload_bytes"], 0)
        with torch.device("meta"):
            linear = torch.nn.Linear(3, 2)
        group = inspect_vmm_component(linear).as_dict()["setup"]["capacity_inspection"]["parameters"][0]
        self.assertEqual(group["selected_names"], ["weight"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            inspect_vmm_component(self.model(), component=" ")
        with self.assertRaises(TypeError):
            inspect_vmm_component(object())

    def test_parameter_identity_is_not_storage_deduplication(self):
        root = torch.nn.Module()
        storage = torch.zeros(6)
        root.first = torch.nn.Parameter(storage)
        root.second = torch.nn.Parameter(storage)
        layout = inspect_vmm_component(root).as_dict()["setup"]["capacity_inspection"]
        self.assertEqual(layout["unique_parameter_bytes"], 48)
        self.assertEqual(len(layout["parameters"]), 2)

    def test_shared_module_alias_is_not_falsely_selected(self):
        with torch.device("meta"):
            root = torch.nn.Module()
            root.first = torch.nn.Linear(3, 2, bias=False)
            root.second = root.first
        layout = inspect_vmm_component(root).as_dict()["setup"]["capacity_inspection"]
        self.assertEqual(layout["selected_module_count"], 1)
        self.assertEqual(layout["parameters"][0]["nonselected_names"], ["second.weight"])
        self.assertEqual(layout["shared_selected_nonselected_bytes"], 24)
