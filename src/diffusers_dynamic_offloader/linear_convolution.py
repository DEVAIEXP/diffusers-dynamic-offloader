"""Explicit eager convolution offload preserving the component's native math."""

import torch
from torch import nn

CONVOLUTIONS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def validate_convolutions(root, config):
    """Refuse unsupported ownership before moving tensors or replacing forwards."""
    if config.execution_mode != "linear_runtime" or torch.device(config.offload_device).type != "cpu":
        raise ValueError("Convolution offload requires linear_runtime with CPU offload")
    if config.auto_budget_policy != "off" or config.resident_module_budget_gb or config.always_resident_modules_pattern:
        raise ValueError("Initial convolution offload requires explicit zero residency and no auto budget")
    if config.build_profile or config.cache_pinned_weights:
        raise ValueError("Convolution calibration and persistent pinned cache are not supported")
    if torch.device(config.execution_device).type == "cuda" and torch.cuda.get_allocator_backend() != "native":
        raise ValueError("Initial convolution offload requires the normal PyTorch CUDA allocator")
    owners = {}
    storage_owners = {}
    for name, module in root.named_modules():
        for key, parameter in module.named_parameters(recurse=False):
            owners.setdefault(id(parameter), []).append((name, key))
            if parameter.device.type == "cpu" and parameter.numel():
                storage_owners.setdefault(parameter.untyped_storage().data_ptr(), []).append((name, key))
    for name, module in root.named_modules():
        if not isinstance(module, CONVOLUTIONS):
            continue
        if not name or module.training or list(module.children()) or list(module.buffers(recurse=False)):
            raise ValueError("Convolution offload requires an eval child with no child modules or buffers")
        if (
            module._forward_hooks
            or module._forward_pre_hooks
            or hasattr(module, "_diffusers_hook")
            or hasattr(module, "_hf_hook")
        ):
            raise ValueError("Convolution already has execution hooks")
        for parameter in module.parameters(recurse=False):
            if type(parameter) is not nn.Parameter or parameter.device.type != "cpu" or not parameter.is_contiguous():
                raise ValueError("Convolution offload requires plain contiguous CPU parameters")
            if parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
                raise ValueError("Convolution offload requires dense floating-point parameters")
            if len(owners[id(parameter)]) != 1 or (
                parameter.numel() and len(storage_owners[parameter.untyped_storage().data_ptr()]) != 1
            ):
                raise ValueError("Shared convolution parameter ownership is not supported")


def patch_convolution(hook, module):
    """Scope temporary weights to eager, current-stream native forward execution.

    PyTorch owns copies and stream-ordered retirement. No stream, global sync,
    residency cache or convolution kernel is introduced. Custom forwards must
    not mutate parameters, escape weight views or dispatch on another stream.
    """
    original = module.forward
    active = False

    def forward(_module, *args, **kwargs):
        nonlocal active
        if active or _module.training or torch.is_grad_enabled():
            raise RuntimeError("Convolution offload requires non-reentrant inference")
        if torch.compiler.is_compiling():
            raise RuntimeError("Convolution offload does not support compiled execution")
        value = args[0] if args else kwargs.get("input", kwargs.get("x"))
        if not isinstance(value, torch.Tensor) or value.device != hook.execution_device:
            raise ValueError("Convolution input must be on the configured execution device")
        if value.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Convolution offload does not support CUDA graph capture")
        parameters = tuple(_module.parameters(recurse=False))
        sources = tuple(parameter.data for parameter in parameters)
        active = True
        hook._linear_convolution_active = getattr(hook, "_linear_convolution_active", 0) + 1
        try:
            for parameter in parameters:
                parameter.data = hook._to_input_device(
                    parameter, value, "convolution_parameter", cast_to_input_dtype=False
                )
            output = original(*args, **kwargs)

            def borrowed(item):
                if isinstance(item, torch.Tensor):
                    return any(
                        item.device == parameter.device
                        and item.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()
                        for parameter in parameters
                    )
                if isinstance(item, (tuple, list)):
                    return any(borrowed(child) for child in item)
                if isinstance(item, dict):
                    return any(borrowed(child) for child in item.values())
                return False

            if borrowed(output):
                raise RuntimeError("Convolution output must not borrow temporary parameter storage")
            return output
        finally:
            for parameter, source in zip(parameters, sources, strict=True):
                parameter.data = source
            hook._linear_convolution_active -= 1
            active = False

    hook._patched_modules.append((module, original))
    hook.state.patched_module_count += 1
    module.forward = forward.__get__(module, type(module))
