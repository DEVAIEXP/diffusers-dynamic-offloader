import json
import tempfile
import unittest
from pathlib import Path

from diffusers_dynamic_offloader import (
    dynamic_offload_profile_key,
    get_dynamic_offload_profile_capacity,
    load_dynamic_offload_profile,
    record_dynamic_offload_profile,
)


class DynamicOffloadProfileTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "model": {"identifier": "example/model", "revision": "abc"},
            "runtime": {"dtype": "bfloat16", "preset": "auto"},
            "hardware": {"vram_gb": 8.0},
        }

    def test_key_is_stable_across_mapping_order(self):
        reordered = {
            "hardware": {"vram_gb": 8.0},
            "runtime": {"preset": "auto", "dtype": "bfloat16"},
            "model": {"revision": "abc", "identifier": "example/model"},
        }
        self.assertEqual(dynamic_offload_profile_key(self.context), dynamic_offload_profile_key(reordered))

    def test_record_merges_capacity_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = record_dynamic_offload_profile(
                self.context,
                {"status": "success", "capacity": {"metric": "tokens", "value": 22000}},
                directory,
            )
            record_dynamic_offload_profile(
                self.context,
                {"status": "success", "capacity": {"metric": "tokens", "value": 20000}},
                directory,
            )
            record_dynamic_offload_profile(
                self.context,
                {"status": "failure", "capacity": {"metric": "tokens", "value": 24640}},
                directory,
            )
            profile = load_dynamic_offload_profile(self.context, directory)

            self.assertEqual(path.parent, Path(directory))
            self.assertEqual(get_dynamic_offload_profile_capacity(profile, "tokens"), {"max_success": 22000.0, "min_failure": 24640.0})
            self.assertEqual(len(profile["observations"]), 3)

    def test_invalid_or_wrong_context_profile_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{dynamic_offload_profile_key(self.context)}.json"
            path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
            self.assertIsNone(load_dynamic_offload_profile(self.context, directory))
