"""CPU ownership contracts; these do not establish CUDA performance."""

import pytest
import torch
from torch import nn

from diffusers_dynamic_offloader import enable_offload, remove_dynamic_offload
from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadConfig, DynamicOffloadHook
from diffusers_dynamic_offloader.linear_convolution import patch_convolution, validate_convolutions


def config(**kwargs):
    return DynamicOffloadConfig(
        execution_mode="linear_runtime", execution_device="cpu", linear_convolution_offload=True, **kwargs
    )


@pytest.mark.parametrize(
    "kind,shape", [(nn.Conv1d, (1, 2, 8)), (nn.Conv2d, (1, 2, 8, 8)), (nn.Conv3d, (1, 2, 4, 4, 4))]
)
def test_native_math_and_restore(kind, shape):
    root = nn.Sequential(kind(2, 3, 3, padding=1)).eval()
    validate_convolutions(root, config())
    layer = root[0]
    original = layer.forward
    pointers = [p.data_ptr() for p in layer.parameters()]
    value = torch.randn(shape)
    with torch.inference_mode():
        expected = root(value)
        hook = DynamicOffloadHook(config())
        patch_convolution(hook, layer)
        torch.testing.assert_close(root(value), expected, rtol=0, atol=0)
    assert [p.data_ptr() for p in layer.parameters()] == pointers
    hook._restore_patches()
    assert layer.forward == original


def test_custom_forward_kwargs_and_interrupt_restore():
    class Causal(nn.Conv2d):
        def forward(self, x, *, fail=False):
            if fail:
                raise KeyboardInterrupt()
            return super().forward(x[:, :, 0]).unsqueeze(2)

    root = nn.Sequential(Causal(2, 3, 1)).eval()
    hook = DynamicOffloadHook(config())
    validate_convolutions(root, hook.config)
    patch_convolution(hook, root[0])
    before = root[0].weight.data_ptr()
    with torch.inference_mode():
        assert root[0](torch.randn(1, 2, 1, 4, 4)).shape == (1, 3, 1, 4, 4)
        with pytest.raises(KeyboardInterrupt):
            root[0](torch.randn(1, 2, 1, 4, 4), fail=True)
    assert root[0].weight.data_ptr() == before
    assert hook._linear_convolution_active == 0


def test_partial_copy_failure_restores_all_sources():
    root = nn.Sequential(nn.Conv2d(2, 3, 1)).eval()
    hook = DynamicOffloadHook(config())
    patch_convolution(hook, root[0])
    pointers = [p.data_ptr() for p in root[0].parameters()]
    count = 0

    def copy(parameter, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("copy failed")
        return parameter.detach().clone()

    hook._to_input_device = copy
    with torch.inference_mode(), pytest.raises(RuntimeError, match="copy failed"):
        root(torch.randn(1, 2, 4, 4))
    assert [p.data_ptr() for p in root[0].parameters()] == pointers
    assert hook._linear_convolution_active == 0


def test_default_and_route_isolation():
    assert not DynamicOffloadConfig().linear_convolution_offload
    assert nn.Conv2d not in DynamicOffloadConfig().target_module_classes
    assert nn.Conv2d in config().target_module_classes
    with pytest.raises(ValueError, match="only available"):
        DynamicOffloadConfig(linear_convolution_offload=True)


def test_shared_storage_refused_before_mutation():
    root = nn.Sequential(nn.Conv2d(2, 2, 1), nn.Conv2d(2, 2, 1)).eval()
    root[1].weight = nn.Parameter(root[0].weight.detach())
    with pytest.raises(ValueError, match="Shared"):
        validate_convolutions(root, config())


def test_grad_mode_refused():
    layer = nn.Conv2d(2, 2, 1).eval()
    patch_convolution(DynamicOffloadHook(config()), layer)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference"):
        layer(torch.randn(1, 2, 4, 4))


def test_full_hook_initialization_and_detach():
    root = nn.Sequential(nn.Conv2d(2, 2, 1), nn.GroupNorm(1, 2)).eval()
    hook = DynamicOffloadHook(config())
    original = root[0].forward
    hook.initialize_hook(root)
    assert hook.state.patched_module_count == 1
    assert hook.state.planner_decisions["linear_convolution_offload"] is True
    with torch.inference_mode():
        assert root(torch.randn(1, 2, 4, 4)).shape == (1, 2, 4, 4)
    hook.detach_hook(root)
    assert root[0].forward == original


def test_public_component_api():
    root = nn.Sequential(nn.Conv2d(2, 2, 1)).eval()
    result = enable_offload(root, component="vae", route="dynamic_offload", config=config(), use_environment=False)
    with result.cleanup_on_error(), torch.inference_mode():
        root(torch.randn(1, 2, 4, 4))
    assert result.hook.config.linear_convolution_offload
    remove_dynamic_offload(root)


def test_borrowed_output_and_active_detach_are_refused():
    class Borrow(nn.Conv2d):
        def forward(self, x):
            with pytest.raises(RuntimeError, match="active"):
                hook.detach_hook(root)
            return self.weight.view(-1)

    root = nn.Sequential(Borrow(2, 2, 1)).eval()
    hook = DynamicOffloadHook(config())
    patch_convolution(hook, root[0])
    before = root[0].weight.data_ptr()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="borrow"):
        root(torch.randn(1, 2, 4, 4))
    assert root[0].weight.data_ptr() == before
