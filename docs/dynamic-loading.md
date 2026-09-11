# Dynamic Module Loading

`from_pretrained_with_dynamic_offload(...)` is DDO's controlled component-loading entry point. Its purpose is to let a host load a memory-heavy module, usually a transformer, with loading options such as `device_map="cpu"` and an optional safetensors backend override before attaching the module to a full or staged pipeline.

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

## What The Helper Does

The helper forwards the supplied arguments, including `device_map`, to the selected loader's `from_pretrained` method. While loading, it temporarily selects the configured safetensors backend (`DynamicOffloadConfig.load_safetensors_backend`), then restores the normal backend. The supported overrides are `mmap` and `pread`; configure one with `DDO_LOAD_SAFETENSORS_BACKEND` only after measuring it on the target host.

This lets a host use the same loading configuration whether it builds a complete pipeline or loads a component on its own, while avoiding an unnecessary full-pipeline transformer load before the host has chosen its placement and offload path.

`model_loader` can be a class with `from_pretrained` or an equivalent callable. If omitted, DDO uses Diffusers `AutoModel`.

## Loading And Offload Are Separate

With `apply_dynamic=False`, the helper only loads and returns `DynamicOffloadLoadResult(module=...)`; it does not install a dynamic-offload hook or choose a compatibility route. Call `enable_offload(...)` afterwards so preset resolution, component policy, quantized-backend detection, and Diffusers group-offload fallback occur once.

Set `apply_dynamic=True` only when the caller intentionally wants to apply DDO's dynamic hook during loading. In that case the result also includes `hook` and `should_move_to_execution_device`. Do not follow it with a second dynamic-offload application to the same module.

## Benchmark Guidance

For comparisons where transformer setup is measured, use this loading path consistently for every runner that owns transformer construction. Record loading separately from `enable_offload(...)` setup, and keep `apply_dynamic=False` when the latter call owns routing. Compare the selected `device_map` and safetensors backend as part of the setup configuration; it avoids attributing a loader-path difference to an offload policy.
