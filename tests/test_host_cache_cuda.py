"""Explicit small CUDA boundary gate; not part of CPU validation."""
import pytest
import torch

from diffusers_dynamic_offloader import release_pytorch_host_cache

pytestmark = pytest.mark.vmm


def test_host_cache_release_preserves_live_pinned_storage_and_pending_copy():
    torch.cuda.init()
    live = torch.full((2**20,), 37, dtype=torch.uint8, pin_memory=True)
    unused = torch.empty(2**20, dtype=torch.uint8, pin_memory=True)
    device_copy = live.to("cuda:0", non_blocking=True)
    del unused
    report = release_pytorch_host_cache(enabled=True, phase="small_cuda_gate", device="cuda:0")
    observation = report.as_dict()["setup"]["host_cache_release"]
    assert observation["status"] == "released"
    assert observation["allocator_bytes_delta"] >= 2**20
    assert observation["after"]["allocated_bytes"] >= 2**20
    assert live.is_pinned()
    assert torch.equal(device_copy.cpu(), live)
    live.fill_(23)
    assert torch.all(live == 23).item()
