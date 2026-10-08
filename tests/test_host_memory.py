"""CPU-only contracts for detached, read-only host ownership observations."""
import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from diffusers_dynamic_offloader import inspect_host_memory


@pytest.fixture(autouse=True)
def forbid_cuda():
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
         patch.object(torch.cuda, "synchronize", side_effect=AssertionError("wait forbidden")):
        yield


def test_host_only_snapshot_and_profile():
    with patch("diffusers_dynamic_offloader.host_memory._host_snapshot", return_value={"process_private_bytes": 42}):
        report = inspect_host_memory(phase="before_load")
    assert report.setup["host_memory"]["host"]["process_private_bytes"] == 42
    assert "cpu_storages" not in report.setup["host_memory"]
    assert "phase=before_load" in report.format_profile()
    json.dumps(report.as_dict())


def test_small_views_report_full_storage_without_duplicate_counting():
    base = torch.arange(1024, dtype=torch.float32)
    model = torch.nn.Module()
    model.register_parameter("small", torch.nn.Parameter(base[:2]))
    model.register_buffer("alias", base[4:7])
    model.register_buffer("meta", torch.empty(3, device="meta"))
    report = inspect_host_memory(model, phase="loaded")
    census = report.setup["host_memory"]["cpu_storages"]
    assert census["unique_storage_count"] == 1
    assert census["unique_storage_bytes"] == 4096
    assert census["cpu_tensor_reference_bytes"] == 20
    assert census["cpu_tensor_references"] == 2
    assert census["non_cpu_tensor_references"] == 1
    assert census["storage_samples"][0]["references"] == 2
    json.dumps(report.as_dict())


def test_shared_parameters_count_references_not_double_storage():
    model = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.ones(4))
    model.register_parameter("one", parameter)
    model.register_parameter("two", parameter)
    census = inspect_host_memory(model).setup["host_memory"]["cpu_storages"]
    assert census["cpu_tensor_references"] == 2
    assert census["unique_storage_bytes"] == 16


def test_source_descriptors_are_detached_and_no_weight_owner_is_retained():
    model = torch.nn.Linear(2, 3)
    parameter = weakref.ref(model.weight)
    model._ddo_safetensor_slices = SimpleNamespace(
        lookup=lambda tensor: SimpleNamespace(path=Path("shard.safetensors")))
    report = inspect_host_memory(model)
    assert report.setup["host_memory"]["cpu_storages"]["storage_samples"][0]["shards"] == ["shard.safetensors"]
    del model
    gc.collect()
    assert parameter() is None
    json.dumps(report.as_dict())


def test_bounded_sorted_storage_samples():
    model = torch.nn.Module()
    for count in (2, 8, 4):
        model.register_buffer(f"b{count}", torch.ones(count))
    census = inspect_host_memory(model, max_storage_samples=1).setup["host_memory"]["cpu_storages"]
    assert census["unique_storage_bytes"] == 56
    assert census["omitted_storage_count"] == 2
    assert census["storage_samples"][0]["storage_bytes"] == 32


def test_registry_errors_are_explicit_not_empty_success():
    def fail(tensor):
        raise RuntimeError("closed registry")
    model = torch.nn.Linear(2, 3)
    model._ddo_safetensor_slices = SimpleNamespace(lookup=fail)
    observation = inspect_host_memory(model, max_storage_samples=0).setup["host_memory"]
    assert observation["status"] == "partial"
    assert observation["cpu_storages"]["inspection_errors"] == 2
    assert observation["cpu_storages"]["errors"] == []


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_sample_limit(limit):
    with pytest.raises(ValueError):
        inspect_host_memory(max_storage_samples=limit)
