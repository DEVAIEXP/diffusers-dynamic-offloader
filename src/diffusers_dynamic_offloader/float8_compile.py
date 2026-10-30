"""Explicit experimental FP8 Linear compute compilation, not residency policy."""

from dataclasses import dataclass
from functools import partial
from types import FunctionType


def _linear_compute(input, weight, bias):
    import torch
    return torch.nn.functional.linear(input, weight, bias)


def _dispatch_linear(module, compiled, input):
    # Read the current native-published views on every invocation.
    return compiled(input, module.weight, module.bias)


@dataclass
class Float8CompileHandle:
    bindings: list

    def remove(self):
        """Remove compute wrappers only after native wrappers have retired."""
        if any(module.forward is not compiled for module, compiled in self.bindings):
            raise RuntimeError("Retire native forward boundaries before removing FP8 compilation")
        for module, _ in self.bindings:
            del module.forward
        self.bindings.clear()


def configure_float8_linear_compile(model):
    """Compile supported standard Linear computation before native attachment.

    Lazy CUDA compilation errors propagate. No eager fallback, dtype changes,
    parameter replacement, source retention, or custom math implementation.
    """
    import torch
    from ddo_vmm_allocator.quantized_payload import _supported_backend

    cls = _supported_backend()
    if not isinstance(model, torch.nn.Module) or any(module.training for module in model.modules()):
        raise ValueError("FP8 compilation requires an eval component")
    modules = [module for module in model.modules()
               if isinstance(module, torch.nn.Linear) and type(module.weight) is cls]
    if not modules:
        raise ValueError("No supported FP8 Linear weights found")
    if any(type(module) is not torch.nn.Linear for module in modules):
        raise ValueError("FP8 compilation requires standard Linear forward semantics")
    if any("forward" in vars(module) or module._compiled_call_impl is not None for module in modules):
        raise ValueError("Configure FP8 compilation before native forward wrappers or module compilation")
    options = {"triton.cudagraphs": False, "emulate_precision_casts": True,
               "pattern_matcher": False, "epilogue_fusion": False}
    handle = Float8CompileHandle([])
    geometries = {}
    try:
        for module in modules:
            key = (tuple(module.weight.shape), module.weight.dtype,
                   None if module.bias is None else (tuple(module.bias.shape), module.bias.dtype))
            if key not in geometries:
                # Dynamo caches by code object. Separate geometries without
                # increasing global limits or compiling the native boundaries.
                compute = FunctionType(_linear_compute.__code__.replace(),
                                       _linear_compute.__globals__, _linear_compute.__name__)
                geometries[key] = torch.compile(compute, backend="inductor", fullgraph=True,
                                                dynamic=False, options=options.copy())
            dispatch = partial(_dispatch_linear, module, geometries[key])
            handle.bindings.append((module, dispatch))
            module.forward = dispatch
    except BaseException:
        handle.remove()
        raise
    return {"requested": True, "configuration_applied": True, "denoise_completed": False,
            "compiled_linear_count": len(modules), "backend": "inductor", "fullgraph": True,
            "compiled_geometry_count": len(geometries), "cache_scope": "linear_weight_geometry",
            "dynamic": False, "options": options, "boundary": "linear_compute_before_native_attach",
            "module_call_compilation": False,
            "timing_note": "Lazy compilation is included in the first denoise interval"}, handle
