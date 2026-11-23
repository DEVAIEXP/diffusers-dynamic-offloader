# Integration

Choose the execution family first, then use the same DDO loading, attachment,
reporting and cleanup APIs. The host still owns the model class, inference
inputs and pipeline phases. See [runtimes and presets](presets-and-platforms.md)
for the available selections.

## Common flow

1. Install DDO and the dependencies needed by the model.
2. Select a backend and preset. VMM Fast must bootstrap before CUDA-owning imports.
3. Load the component on CPU. For native file-backed transport, retain its
   immutable checkpoint sources through DDO's loading helper.
4. Attach offload once and pass the returned module to the pipeline.
5. Run inference, collect the public report and retire the component at the
   application's phase boundary.

Linear uses ordinary PyTorch allocation and needs no native allocator.
VMM Fast additionally needs a matching packaged allocator and a fresh-process
bootstrap. A GPU address can remain unchanged while native code safely reuses
the physical memory behind it; this lets the runtime deliver weights as needed.

## Select either family in one runner

Place this at the beginning of a standalone runner, before Torch, Diffusers or
model imports:

```python
from ddo_vmm_bootstrap import ensure_vmm_fast_process, vmm_fast_requested

if vmm_fast_requested():
    ensure_vmm_fast_process()
```

It relaunches the same command in a fresh child for `vmm/auto`. Linear and
`vmm/nvme_baseline` do not use that process bootstrap. Packaged libraries are
discovered automatically; an ordinary installation needs no DLL/.so path.

Then register the common selector and build the configuration:

```python
import argparse
from dataclasses import replace

import torch
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    add_dynamic_offload_arguments,
    enable_offload,
    from_pretrained_with_dynamic_offload,
    resolve_dynamic_offload_backend,
)

parser = argparse.ArgumentParser()
add_dynamic_offload_arguments(parser)
args = parser.parse_args()

backend = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset)
settings = DynamicOffloadSettings.from_env(
    default_preset=args.ddo_preset if backend.backend == "linear" else "one_shot_fast",
    execution_device="cuda:0",
    offload_device="cpu",
    environ={},  # Keep this example's selection explicit.
)
config = (
    settings.config if backend.backend == "linear"
    else backend.build_config_from_args(args)
)
if backend.requires_native_bootstrap:
    config = replace(config, vmm_native_transport=True)

settings = replace(
    settings,
    requested_preset=backend.requested_preset,
    effective_preset=backend.effective_preset,
    config=config,
)

def attach_component(model_path, model_class):
    loaded = from_pretrained_with_dynamic_offload(
        model_path,
        model_loader=model_class,
        dynamic_offload_config=config,
        apply_dynamic=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    return enable_offload(
        loaded.module,
        settings=settings,
        config=config,
        component="transformer",
        execution_device="cuda:0",
        offload_device="cpu",
        use_environment=False,
    )
```

The application calls `attach_component()` with the supported model class and
component directory. For native file-backed weights, use a resolved local
safetensors snapshot. If the component is inside a pipeline repository, pass its
subfolder to the loader; see [dynamic loading](dynamic-loading.md).

The example disables inherited settings in `from_env()` and component routing.
The native CLI configuration builder still recognizes selected native environment
controls. Applications should document that precedence; see
[configuration](configuration.md). Do not bulk-move the attached module to CUDA
or apply another set of offload hooks to it.

### Common command

The example runner accepts the same selectors on every platform:

```text
your_runner.py --ddo-backend linear --ddo-preset one_shot_fast
your_runner.py --ddo-backend vmm --ddo-preset auto
```

### Windows PowerShell

Use the interpreter from your Windows environment:

```powershell
& ./.venv-ddo/Scripts/python.exe your_runner.py --ddo-backend linear --ddo-preset one_shot_fast
& ./.venv-ddo/Scripts/python.exe your_runner.py --ddo-backend vmm --ddo-preset auto
```

### Native Linux and WSL Bash

Use the interpreter from that platform's environment:

```bash
./.venv-ddo/bin/python your_runner.py --ddo-backend linear --ddo-preset one_shot_fast
./.venv-ddo/bin/python your_runner.py --ddo-backend vmm --ddo-preset auto
```

These are integration patterns, not a model recipe or fixed memory budget.
Real model commands and their platform capacities are in
[DDO examples](https://devaiexp.github.io/ddo-examples/docs/user/).

## Run, report and retire

```python
from diffusers_dynamic_offloader import remove_dynamic_offload

# result is returned by attach_component(...).
with result.cleanup_on_error():
    output = run_component(result.module)  # Application-owned inference.
    report = result.metrics_snapshot(collect_completed=True)
print(report.format_profile())

# Successful-path retirement is explicit, after consuming the required outputs.
remove_dynamic_offload(result.module)
del result
```

The error boundary attempts DDO cleanup and preserves the original exception if
cleanup also fails. It does not unload on success. Native retirement waits on
the runtime's owned work; unsafe completion retains ownership rather than
force-freeing it. Native unload can replace weights with meta placeholders:
reload before using that component again.

Remove pipeline, loader and state references that still retain the component.
Official Diffusers group hooks remain owned by Diffusers; removing a DDO hook
does not remove those hooks. See [staged execution](staged-pipeline.md) for
phase boundaries and [metrics](metrics.md) for collection scope.

## Choose the pipeline structure

| Structure | Use |
| --- | --- |
| [Full pipeline](full-pipeline.md) | The existing pipeline orchestrates encoding, denoising and decode |
| [Staged pipeline](staged-pipeline.md) | The host loads and retires components in separate phases |
| [Modular Diffusers](modular-pipeline.md) | Build only the blocks needed by each phase |

An attached transformer does not choose the encoder or VAE strategy.
Native whole-component ownership is an explicit option with its own source,
buffer and capacity checks. See [component strategies](component-strategies.md),
[capacity profiles](capacity-profiles.md) and [configuration](configuration.md).
