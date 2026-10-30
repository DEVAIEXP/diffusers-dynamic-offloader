"""CPU contracts for failure-only, nonrecovering memory explanations."""
from unittest.mock import patch
import os

import pytest
import torch

from diffusers_dynamic_offloader.memory_failure import diagnose_memory_failure


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
         patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA forbidden")):
        yield


def test_low_commit_is_correlation_not_cuda_oom():
    error = RuntimeError("CUDA illegal memory access")
    report = diagnose_memory_failure(error, observation={
        "status": "observed", "available_bytes": 1437696})
    assert report["classification"] == "failure_with_low_system_commit_headroom"
    assert "does not prove" in report["message"]
    assert "1.37 MiB" in report["message"]
    assert str(error) == "CUDA illegal memory access"


def test_adequate_or_unknown_headroom_does_not_blame_commit():
    for observation in ({"status": "observed", "available_bytes": 7 * 1024**3},
                        {"status": "unavailable"}, {"status": "observed"},
                        {"status": "observed", "available_bytes": -1}):
        report = diagnose_memory_failure(RuntimeError("kernel bug"), observation=observation)
        assert report["classification"] == "unclassified"
        assert report["message"] is None


def test_explicit_commit_failure_in_chain_and_cycle():
    original = OSError("allocation refused")
    original.winerror = 1455
    wrapper = RuntimeError("cleanup also failed")
    wrapper.__cause__ = original
    original.__cause__ = wrapper
    report = diagnose_memory_failure(wrapper, observation={"status": "unavailable"})
    assert report["classification"] == "windows_commit_allocation_failed"


def test_failed_observation_preserves_host_oom():
    with patch("diffusers_dynamic_offloader.memory_failure._windows_commit_snapshot",
               side_effect=OSError("query failed")):
        report = diagnose_memory_failure(MemoryError("original"))
    assert report["classification"] == "host_allocation_failed"
    assert report["host"]["status"] == "unavailable"
    assert "specific memory limit is not established" in report["message"]


@pytest.mark.skipif(os.name != "nt", reason="Windows API contract")
def test_real_windows_query_is_host_only_and_consistent():
    report = diagnose_memory_failure(RuntimeError("injected failure"))
    host = report["host"]
    assert host["status"] == "observed"
    assert host["available_bytes"] == host["limit_bytes"] - host["commit_bytes"]
    assert host["physical_available_bytes"] >= 0


def test_invalid_observation_does_not_mask_explicit_allocation_failure():
    report = diagnose_memory_failure(MemoryError(), observation={"status": "observed"})
    assert report["classification"] == "host_allocation_failed"
    assert report["host"]["status"] == "unavailable"
