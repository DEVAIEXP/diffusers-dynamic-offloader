# Native VMM usage and support

`linear/auto` remains the default portable PyTorch route. Choose native VMM
explicitly; it is not silently enabled by a model class or platform.

| Host selection | Route |
| --- | --- |
| `--ddo-backend linear --ddo-preset auto` | Existing `one_shot_fast` |
| `--ddo-backend vmm --ddo-preset auto` | Experimental `vmm_fast_runtime` |
| `--ddo-backend vmm --ddo-preset nvme_baseline` | Protected `vmm_runtime` |

## Installation and lifecycle

Initialize the pinned `ddo-vmm-allocator` Git submodule, install its Python
adapter, build the native library for the platform and select that library.
Python wheels do not currently bundle a DLL/.so. Development extras are optional:
Windows `.[vmm-dev]`, Linux `.[vmm-dev-linux]`. Ordinary installs should not
acquire native development tools. See [Linux setup](linux-environment.md) and
the allocator README for native build recipes.

Bootstrap discovers the conventional checkout build, then a platform binary
inside the installed adapter: `ddo_vmm_allocator/native/win32/ddo_vmm.dll` or
`ddo_vmm_allocator/native/linux/libddo_vmm.so`. A host need not specify a path
when its installed distribution includes that binary. Current wheels do not
yet include it; binary delivery remains [release work](release-readiness.md).
Explicit `library_path` takes precedence over `DDO_VMM_LIBRARY_PATH`, which
takes precedence over automatic discovery. These are development/diagnostic
overrides, not normal end-user requirements. Missing/invalid binaries fail
clearly; no runtime compilation/download, build-directory scan or silent switch
to an experimental binary is performed. Discovery imports neither Torch nor
the allocator package in the parent. The child receives the resolved library
path so bootstrap and DDO use the same artifact.

The runner must bootstrap before any CUDA-owning imports:

```python
from ddo_vmm_bootstrap import ensure_vmm_fast_process, vmm_fast_requested

if vmm_fast_requested():
    ensure_vmm_fast_process()

# Only now import Torch, Diffusers and the application pipeline.
```

Then add DDO's standard arguments, resolve the route and pass its configuration.
For native weights, load with `from_pretrained_with_dynamic_offload(...,
apply_dynamic=False)` before `enable_offload(...)`, so DDO can associate immutable
checkpoint slices. The host provides the model loader and pipeline phases;
DDO/native code owns descriptors, source bytes, transfers, events and cleanup.
The checkpoint must stay immutable throughout execution.

Active GPU capacity, retained host storage and CUDA registration have separate
ceilings. A virtual address reservation is not physical VRAM consumption.
The benchmark used full-source coverage; smaller host capacity can introduce
rereads or fallback and does not promise the same throughput. VMM capacity
profiles are observations, not Python residency calibration.

Generic hosts can pass `--ddo-vmm-source-cache-gb` for retained storage and
`--ddo-vmm-source-registered-gb` for the separate registration ceiling through
`add_dynamic_offload_arguments()` / `build_config_from_args()`. Registration
zero retains the existing storage-ceiling default. Values are GiB; explicit
registration must fit retained storage and whole native segments. Adding this
argument does not alter the selected benchmark's capacities or native policy.

## Phase-staged execution

For tight memory, encode/read back outputs, collect the component's completed
metrics and unload/delete its model and pipeline references before loading
transformer weights. Decode after transformer outputs are consumed and native
cleanup completes. Cleanup belongs in `finally`; preserve the original forward
error if cleanup also fails. No per-step global synchronization is required.
Native unload invalidates weights to meta placeholders; reload before reuse.

An attached DDO result offers an error-only boundary:

```python
with result.cleanup_on_error():
    output = run_component(result.module)
    report = result.metrics_snapshot(collect_completed=True)
# Successful-path unload remains explicit at the application's phase boundary.
```

Success performs no automatic unload/wait. On an exception or interruption,
DDO attempts the existing hook cleanup once and re-raises the original error;
a second cleanup error is attached as a diagnostic. Failed native retirement
retains ownership rather than force-releasing it. This boundary does not cover
attachment before the result exists, nor promise poisoned-driver recovery.
The host wrapper is CPU-tested; no new CUDA/performance run has been made for it.

Generic full-pipeline linear/compatibility examples do not prove simultaneous
native encoder and transformer fit the configured ceilings. Native whole-component
ownership is opt-in; supported immutable tensors, bounded buffers and shared
aliases must pass preflight. Quantized/custom forwards normally remain on
compatibility routes. The explicit, model-agnostic saved TorchAO recipe is the
bounded experimental [FP8 exception](fp8-weight-only.md), not automatic support
for all quantized tensors. Do not replace backend math with dense Linear execution.

## Public reporting

```python
import json

report = result.metrics_snapshot(collect_completed=True)
if args.show_profile:
    print(report.format_profile())
if args.show_metrics:
    print(json.dumps(report.as_dict(), indent=2))
# Pass report.as_dict() to the application's normal JSON writer.
```

Completion collection retires finished source tickets; it is not module unload
or a per-step synchronization. Hosts consume this versioned object, not hook
state/native attribute names. The sibling tracker accepts
`tracker.record_ddo_metrics(report, component="transformer")`.
See the [metric catalog](metrics.md) for units, collection points and diagnostics.

## Validated scope

Selected Windows and native Linux BF16 text-only staged workloads completed.
Windows additionally has selected encoder numerical/reuse/robustness evidence;
Linux has selected CPU/CUDA contracts and one complete staged generation.
Coverage is not identical. Other models remain user-driven validation, not a
blanket support claim; multi-GPU, concurrent forwards/unload, unmanaged streams,
hidden escaping weight views, general compiled forwards, multimodal inputs and
poisoned-driver recovery are not validated. No WSL/macOS native VMM promise.
Windows standby purge is optional and safely no-ops on Linux.

[Final results](../experiments/dynamic_offload_results.md) and the
[benchmark manifest](benchmark-reproduction.md) state what was measured and
what remains necessary for an independent Diffusers comparison.

For user-driven validation on other models/environments, use the short
[manual checklist](manual-validation.md); it is not an exhaustive test matrix
or an automatic publication gate.

For an explicitly completed phase, DDO optionally exposes a
[PyTorch pinned host-cache release helper](pytorch-host-cache-boundary.md).
It defaults to disabled, returns a public report, and never clears native source
storage or changes admission. Its opt-in video integration passed the selected
small CUDA contract and matched video gate, with an observed improvement versus
the pressure control. This is not a default preset or a general performance
guarantee; commit pressure remained in the selected run.
