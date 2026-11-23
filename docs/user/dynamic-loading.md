# Dynamic loading

`from_pretrained_with_dynamic_offload(...)` is DDO's controlled component-loading entry point. Its purpose is to let a host load a memory-heavy module, such as a transformer, encoder or VAE, with loading options such as `device_map="cpu"` and an optional safetensors backend override before attaching the module to a full or staged pipeline.

This can reduce setup-time cost or peak memory pressure when the selected loading options are better for the host and model. The helper is not an unconditional accelerator: with the default empty safetensors-backend setting and ordinary `from_pretrained` arguments, it is intentionally close to a transparent wrapper around the selected loader.

```python
import torch
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_offload,
    from_pretrained_with_dynamic_offload,
)

settings = DynamicOffloadSettings.from_env(
    execution_device="cuda:0",
    offload_device="cpu",
    default_preset="auto",
)

loaded = from_pretrained_with_dynamic_offload(
    "your/model",
    model_loader=YourTransformerClass,  # omit when Diffusers AutoModel can resolve it
    dynamic_offload_config=settings.config,
    apply_dynamic=False,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
transformer = loaded.module

result = enable_offload(
    transformer,
    settings=settings,
    component="transformer",
    execution_device="cuda:0",
    offload_device="cpu",
)
transformer = result.module
```

## Loading options

The helper forwards the supplied arguments, including `device_map`, to the selected loader's `from_pretrained` method. While loading, it temporarily selects the configured safetensors backend (`DynamicOffloadConfig.load_safetensors_backend`), then restores the normal backend. The supported overrides are `mmap` and `pread`; configure one with `DDO_LOAD_SAFETENSORS_BACKEND` only after measuring it on the target host.

This lets a host use the same loading configuration whether it builds a complete pipeline or loads a component on its own, while avoiding an unnecessary full-pipeline transformer load before the host has chosen its placement and offload path.

`model_loader` can be a class with `from_pretrained` or an equivalent callable. If omitted, DDO uses Diffusers `AutoModel`.

## Loading and attachment

With `apply_dynamic=False`, the helper only loads and returns `DynamicOffloadLoadResult(module=...)`; it does not install a dynamic-offload hook or choose a compatibility route. Call `enable_offload(...)` afterwards so preset resolution, component policy, quantized-backend detection, and Diffusers group-offload fallback occur once.

Set `apply_dynamic=True` only when the caller intentionally wants to apply DDO's dynamic hook during loading. In that case the result also includes `hook` and `should_move_to_execution_device`. Do not follow it with a second dynamic-offload application to the same module.

## Measurement scope

For comparisons where transformer setup is measured, use this loading path consistently for every runner that owns transformer construction. Record loading separately from `enable_offload(...)` setup, and keep `apply_dynamic=False` when the latter call owns routing. Compare the selected `device_map` and safetensors backend as part of the setup configuration; it avoids attributing a loader-path difference to an offload policy.

## VMM Fast source association

Use the same helper with the VMM configuration from
[integration](integration.md). For file-backed native transport, resolve Hub
weights to a local immutable safetensors snapshot first, then load the component
directory or its subfolder. Keep that checkpoint unchanged during execution.

The helper associates supported weights with checkpoint slices. Native code owns
source bytes, registration and uploads; retaining the loader's CPU tensors is
not the source-cache strategy. Loading and attachment can still allocate
temporary memory, so measure their peaks separately.

A native encoder or VAE also needs explicit whole-component ownership, supported
sources and capacity setup. Selecting the transformer runtime alone does not
configure either component. See [component strategies](component-strategies.md)
and [configuration](configuration.md).

The loader's dtype argument remains backend-owned. It does not authorize native
transport to reinterpret arbitrary stored bytes or universally convert quantized
checkpoints. Saved TorchAO payloads have their own
[FP8 weight-only contract](fp8-weight-only.md).
