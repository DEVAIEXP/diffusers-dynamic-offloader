"""Explicitly approved small gate using the normal PyTorch CUDA allocator."""

import copy
import os

import pytest
import torch
from torch import nn

from diffusers_dynamic_offloader import DynamicOffloadConfig, enable_offload, remove_dynamic_offload

pytestmark = pytest.mark.skipif(
    os.environ.get("DDO_RUN_LINEAR_CONV_CUDA_GATE") != "1", reason="Requires explicit CUDA gate approval"
)


@pytest.mark.parametrize(
    "kind,shape", [(nn.Conv1d, (1, 2, 16)), (nn.Conv2d, (1, 2, 16, 16)), (nn.Conv3d, (1, 2, 4, 8, 8))]
)
def test_convolution_exact_and_retired(kind, shape):
    assert torch.cuda.get_allocator_backend() == "native"
    torch.manual_seed(42)
    root = nn.Sequential(kind(2, 3, 3, padding=1), nn.GroupNorm(1, 3)).to(dtype=torch.bfloat16).eval()
    reference = copy.deepcopy(root).cuda()
    result = enable_offload(
        root,
        component="test_component",
        route="dynamic_offload",
        use_environment=False,
        config=DynamicOffloadConfig(
            execution_device="cuda:0",
            execution_mode="linear_runtime",
            linear_convolution_offload=True,
            pin_cpu_memory=True,
        ),
    )
    try:
        with result.cleanup_on_error(), torch.inference_mode():
            for _ in range(3):
                value = torch.randn(shape, dtype=torch.bfloat16, device="cuda:0")
                expected = reference(value).cpu()
                actual = root(value).cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert root[0].weight.device.type == "cpu"
    finally:
        remove_dynamic_offload(root)


def test_custom_forward_cancellation_restores_cpu_weight():
    class Causal(nn.Conv2d):
        def forward(self, x, *, cancel=False):
            output = super().forward(x[:, :, 0]).unsqueeze(2)
            if cancel:
                raise KeyboardInterrupt()
            return output

    assert torch.cuda.get_allocator_backend() == "native"
    root = nn.Sequential(Causal(2, 3, 1)).to(dtype=torch.bfloat16).eval()
    reference = copy.deepcopy(root).cuda()
    result = enable_offload(
        root,
        component="test_component",
        route="dynamic_offload",
        use_environment=False,
        config=DynamicOffloadConfig(
            execution_device="cuda:0",
            execution_mode="linear_runtime",
            linear_convolution_offload=True,
            pin_cpu_memory=True,
        ),
    )
    with result.cleanup_on_error(), torch.inference_mode():
        value = torch.randn(1, 2, 1, 8, 8, dtype=torch.bfloat16, device="cuda:0")
        torch.testing.assert_close(root(value).cpu(), reference(value).cpu(), rtol=0, atol=0)
        with pytest.raises(KeyboardInterrupt):
            root[0](value, cancel=True)
        assert root[0].weight.device.type == "cpu"
        torch.cuda.synchronize()
    remove_dynamic_offload(root)
