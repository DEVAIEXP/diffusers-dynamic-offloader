# Native VMM usage and support

## Optional execution-slot backing reclaim

`DynamicOffloadConfig(vmm_execution_backing_reclaim=True)` enables the experimental
Windows DXGI policy only on `vmm_fast_runtime`. The default is `False`; DDO-owned
environment configuration also recognizes `DDO_VMM_EXECUTION_BACKING_RECLAIM=1`.
The standard backend `build_config_from_args` path honors this override for
`vmm/auto` as well; explicit `build_config` overrides remain authoritative.
An explicitly enabled policy requires a compatible allocator and native execution
slots. Linux refuses automatic enablement because DXGI is Windows-specific.
No equivalent Linux pressure policy or cross-platform performance is implied.

Before new Torch backing, native code may release at most one completed idle
execution slot when reported local budget headroom is below the requested bytes
plus 256 MiB. This margin is experimental. The virtual address remains reserved;
the next slot acquisition recreates physical backing and the normal upload path
reloads the weights. Leased/prefetched slots and pending consumers remain protected.
Completion or partial release failures retain unsafe ownership and refuse reuse
and close. There is no activation eviction, compression, retry or forced cleanup.

Explicit native release/recreation is shared by Windows and Linux; this does not
enable a Linux automatic pressure policy. The local rc3 Linux wheel has selected
WSL explicit-reclaim and Linux/WSL image functional validation, with automatic
reclaim disabled. These image runs do not validate the Windows DXGI trigger or
establish a speed improvement. Earlier isolated Windows evidence does not certify
a different release artifact. See [selected validation](../developer/validation.md#local-rc3-linux-wheel-pair)
and [metrics and collection controls](metrics.md). Capacity counters do not prove
physical local-VRAM placement.

`linear/auto` remains the default portable PyTorch route. Choose native VMM
explicitly; it is not silently enabled by a model class or platform.

| Host selection | Route |
| --- | --- |
| `--ddo-backend linear --ddo-preset auto` | Existing `one_shot_fast` |
| `--ddo-backend vmm --ddo-preset auto` | Experimental `vmm_fast_runtime` |
| `--ddo-backend vmm --ddo-preset nvme_baseline` | Protected `vmm_runtime` |

## VMM residency profile

The residency budget is the shared ceiling of the two execution slots and the
model arena. Too large a ceiling lets resident weights fill device memory that
activations need; on Windows the working set then exceeds the WDDM local budget
and kernels slow down. A VMM residency profile measures this once and saves the
budget for later runs.

```python
offload = enable_offload(model, config=config, profile_name="my-video-1280x704-8s",
                         build_profile=calibrate)
with offload.profile_run():
    run_denoise(steps=3 if calibrate else steps)
```

- **Build, two runs of three steps each** (`build_profile=True`; run the same
  build twice). Forwards are timed with a device synchronization (build runs
  only); the steady time is the mean of the forwards after the first, which
  includes the initial weight load, and their spread (max - min) is the run's
  own timing noise. At least two steady forwards are needed.
  1. **Floor build.** The arena runs at the structural floor. DDO records the
     device budget (Windows: the lowest DXGI local budget seen during the
     region, since WDDM lowers it under the process's own load; otherwise the
     device total),
     the peak usage (Windows: this process's local usage; otherwise
     driver-visible usage), the resident weight bytes and the first forward's
     tensor input shapes, and derives the hardware ceiling
     `device budget - margin - (peak - resident weights)`, clamped to
     [floor, floor + model size], with a fit class (`fits_with_headroom`,
     `fits_at_limit`, `does_not_fit`). It saves a pending record whose budget is
     the floor, so ordinary runs already use the floor meanwhile.
     It saves a final decision directly when there is no headroom (`floor`), no
     steady timing or spread (`floor`; use at least three steps) or when the
     whole model fits under the ceiling (`whole_model`).
  2. **Ceiling build.** The arena runs at that ceiling. DDO keeps the ceiling
     only when `(floor time - ceiling time) / floor time / extra GiB` reaches
     `vmm_profile_min_gain_per_gib` (default 0.005, that is 0.5 % per GiB) and
     the time saved exceeds the larger steady spread of the two builds;
     otherwise it saves the floor and leaves the device memory free. The
     ceiling run's peak must fit the lowest device budget of both builds minus
     the margin: a kept ceiling is trimmed by the overshoot
     (`ceiling_trimmed_to_device_budget`, `ceiling_trim_bytes`), and the floor
     is saved only if nothing remains above it
     (`ceiling_run_exceeded_device_budget`). Different input shapes between the
     two builds require a restart.
  A further build restarts the calibration. `vmm_profile_margin_gb` defaults to
  0.25 GiB. An explicit residency budget is refused during a build. A future
  native "raise the ceiling during a run" operation could measure both in one
  build.
- **Use** (same name, no explicit budget): the native arena is created with the
  saved budget once, at attachment. An explicit positive budget always wins.
  On Windows the saved budget is first fitted to the WDDM local budget observed
  at attachment (other applications can shrink it after the build): it is
  reduced to `device budget now - margin - build non-weight peak`, never below
  the structural floor and never raised. The report shows `budget_check`
  (`fits`, `reduced`, `device_budget_below_floor` or `unavailable`); an
  unavailable observation keeps the saved budget.
  Without a saved profile the component runs at the structural floor (two
  execution slots plus 64 MiB arena headroom) and reports `missing`.
- **Identity**: name, model class, component, execution unit level, platform,
  GPU name, device memory (256 MiB steps), host RAM (1 GiB steps) and the native
  library SHA256. Files use the `ddo-vmm-residency-v1-` prefix, so linear and
  VMM profiles never share a file even with the same name. DDO does not infer
  resolution or duration: put the workload shape in the name. A use run whose
  first-forward input shapes differ from the build reports `shape_check:
  mismatch` and asks for a rebuild.
- **Does not fit**: the component keeps running at the floor. On Windows WDDM
  may demote memory (slower, no out-of-memory); on Linux there is no demotion
  and no activation offload, so activations that alone exceed the device are
  reported, not hidden.

Reports: `runtime.vmm_residency_profile` in `metrics_snapshot().as_dict()` and
one line in `format_profile()` (see [metrics](metrics.md)). This profile is
Windows-measured on one LTX 2.3 video workload so far; Linux behaviour and other
models need their own validation.

## Proposed resident-weight tuning for larger GPUs

The VMM residency profile above derives the budget automatically from a
measured step; the manual procedure below remains useful for comparisons.
Start with the resident-weight budget from a validated recipe. For the same
workload on a larger GPU, try assigning a fraction of the additional VRAM to
weights:

```text
trial budget = baseline budget + fraction × (target VRAM − baseline VRAM)
```

For example, the LTX 2.3 video VMM fast recipe uses a 0.625 GiB resident-weight
budget on an 8 GiB GPU with `block_child` execution units. The following values
are proposed trials for a 16 GiB GPU and still need validation on that hardware.

| Fraction of additional VRAM | `--ddo-vmm-resident-weight-budget-gb` |
| --- | ---: |
| 0% (same-budget control) | 0.625 |
| 25% (first trial) | 2.625 |
| 50% (second trial) | 4.625 |

Keep resolution, duration, attention backend and all other settings fixed.
Change only the resident-weight budget and compare completed runs using denoising
time and absolute peak VRAM. Leave room for activations and temporary buffers.
Host staging, source cache and source registration have separate budgets.
For the linear runtime, use [capacity profiles](capacity-profiles.md) on the
target GPU with the intended workload.

## Installation and lifecycle

Install a matching platform allocator/DDO wheel pair using the
[installation guide](installation.md). Native wheels provide the DLL/.so;
an editable source install alone does not compile or supply it.

Maintainers can initialize the pinned allocator submodule and build a reviewed
library using the [build guide](../developer/builds.md). Development extras
`.[vmm-dev]` and `.[vmm-dev-linux]` are optional maintainer tools, not
ordinary installation requirements.

Bootstrap can discover a conventional checkout build or the packaged binary:
`ddo_vmm_allocator/native/win32/ddo_vmm.dll` or
`ddo_vmm_allocator/native/linux/libddo_vmm.so`.
A compatible packaged installation requires no manual library path.
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

The child allocator is configured before CUDA initialization with the native
route defaults: segmented pool owners, one 8 GiB growable shared owner reservation
and a 4096 MiB idle cache (see the bootstrap controls in
[configuration](configuration.md)). These require a native binary built with
the segmented pool features; an older binary is refused rather than silently run
without them. Pass `segmented_pool=False, growable_reserve_gib=0` to
`ensure_vmm_fast_process` to use such a binary with the previous pool behavior.
DDO also selects `block_child` execution units by default
(`vmm_execution_unit_level`). These defaults were selected from Windows LTX 2.3
video measurements; they are not a general acceptance claim for other models or
platforms.

Then add DDO's standard arguments, resolve the route and pass its configuration.
For native weights, load with `from_pretrained_with_dynamic_offload(...,
apply_dynamic=False)` before `enable_offload(...)`, so DDO can associate immutable
checkpoint slices. The host provides the model loader and pipeline phases;
DDO/native code owns descriptors, source bytes, transfers, events and cleanup.
The checkpoint must stay immutable throughout execution.

The [native loading and transport architecture](../developer/architecture.md#native-weight-loading-and-transport)
shows positioned parallel reads, retained CPU storage, registration, staging and
GPU execution buffers, including the uncached fallback. These resources have
separate lifetimes and limits; cache hits do not necessarily eliminate H2D.

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

## Support and validation

Selected Windows, native-Linux and WSL workloads are documented in
[support](../developer/support.md) and [validation](../developer/validation.md).
Installed allocator smoke tests are separate from full-model validation.

Multi-GPU, concurrent forwards/unload, unmanaged streams, hidden escaping
weight views, arbitrary compiled forwards and poisoned-driver recovery are
not established by selected acceptance. Windows standby purge is optional
and safely no-ops elsewhere.

Model recipes and comparisons belong to
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).
For another model/environment, use the [manual checklist](../developer/manual-validation.md).

At a completed phase, the optional
[PyTorch host-cache helper](../developer/pytorch-host-cache-boundary.md)
can release unused pinned blocks. It defaults to disabled and never clears
native source storage or changes admission; it is not a general performance guarantee.
