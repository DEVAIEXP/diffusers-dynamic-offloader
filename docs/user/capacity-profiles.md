# Capacity profiles

Pass a profile name when enabling offload. DDO owns lookup, budget application,
measurement and persistence; no context builder or profile session is needed.

```python
offload = enable_offload(
    model, preset="auto", profile_name="my-workload", build_profile=calibrate,
)
with offload.profile_run():
    output = pipeline(**inputs)
```

Set `calibrate=True` for the calibration run and `False` for later runs.
The normal run applies the stored recommendation during setup; its
`profile_run()` context is a no-op. Omit the name to disable profile lookup.
The model may be a transformer, UNet or another supported module: the profile
API does not inspect model-specific inference arguments.

## Native structural minimum

The native fast route derives its initial minimum from the selected adapter's
execution layout. Linear ownership uses its configured Linear grouping;
component ownership, including saved FP8 payloads, uses the component layout.
The minimum accounts for two page-aligned execution slots, persistent root
parameter capacity and arena headroom. Observed persistent capacity remains
reserved when calculating the final profile recommendation.

This calculation does not change the native arena, backing segmentation,
eviction or completion fences. Explicit user budgets are not increased to
accommodate a layout: native admission can refuse an insufficient budget.

## Linear and SDNQ profile identity

Identity is the user-supplied name plus the model class qualified name.
DDO stores JSON under `Path.home() / ".ddo" / "profiles"`, on Windows and Linux.
`DDO_PROFILE_DIR` overrides the directory. Filenames are hashes of the identity;
the readable name is stored inside the file.

Use different names for workloads that need different budgets, for example
`video-5s` and `video-8s`. Resolution, duration, dtype, weights, adapters and GPU
are deliberately not inferred. Recalibrate or choose another name when those
change. Reusing a name and class replaces that recommendation. Profiles made
with the earlier host-context API are not automatically mapped to names.

## Calibration and measurement

Linear, SDNQ and VMM Fast share the named-profile API but use different
measurement and identity rules. This section describes linear and SDNQ;
[VMM residency](#vmm-residency-profile) is described below.

Linear/SDNQ capacity calibration requires the dynamic CUDA linear or SDNQ runtime. It forces the resident
module budget to zero. The host chooses the number of inference steps and wraps
the region to measure in `profile_run()`. DDO saves only when that region exits
successfully; an exception leaves an existing recommendation untouched.

DDO synchronizes at the region boundaries, resets CUDA peak counters, and uses
the maximum of observed driver usage and PyTorch peak reserved memory plus
initial non-PyTorch usage. This conservatively includes the allocator cache.
External allocations that change during inference are not fully captured by
this estimate. Other code must not reset CUDA peak counters inside the region,
and concurrent calibration regions on one device are not supported.

The recommendation is `max(0, total VRAM - measured peak - headroom)`.
`DynamicOffloadConfig.profile_vram_headroom_gb` defaults to 1 GiB; the regular
settings loader also accepts `DDO_PROFILE_VRAM_HEADROOM_GB`.
This is a measured estimate, not a guarantee for changed inputs or later steps.
For linear presets, a positive budget supplied through `resident_module_budget_gb`
or `DDO_RESIDENT_MODULE_BUDGET_GB` takes precedence. Otherwise a compatible named
profile supplies the budget, including a recommendation of zero. Without a usable
profile, the preset's budget policy applies. A positive
`max_resident_module_budget_gb` caps the profile recommendation.

For `sdnq_runtime`, automatic pin/RAM admission counts direct packed parameters,
scales and backend auxiliaries, not dense Linear shape eligibility. Dense fallback
Linears are kept on the execution device and are not SDNQ pin candidates. The
resolved byte ceiling, ratio, maximum and known available-RAM headroom bound new
pins; existing loader-owned pins are preserved and any excess is reported.
This accounts for ordinary packed tensors, not arbitrary tensor-subclass backing
storage or GPU dequantization temporaries. A pin budget is not a resident budget.

An SDNQ zero-resident calibration can save a pending resident probe when measured
headroom permits it. The report's `profile_measurement_kind` distinguishes
`sdnq_baseline_safe`, `sdnq_baseline_probe_pending` and `sdnq_probe`; a pending
recommendation is not a validated resident run. Reusing that profile may perform
the probe in a later inference. Do not infer video capacity from an image profile.

## Staged pipelines and loading

Wrap the denoising stage when that is the workload being calibrated. The host
still prepares inputs, chooses steps, releases stages and controls decode.

```python
# Encode inputs and release encoders before attaching the denoising model.
offload = enable_offload(model, profile_name="my-workload", build_profile=calibrate)
with offload.profile_run() as profile_report:
    latents = pipeline(**inputs, output_type="latent")
# profile_report is optional JSON-safe information for the host's metrics.
# Release the denoising model, then run decoders when appropriate.
```

The loader supports the same lifecycle with explicit runtime configuration:

```python
loaded = from_pretrained_with_dynamic_offload(
    model_path,
    model_loader=ModelClass,
    dynamic_offload_config=settings.config,
    apply_dynamic=True,
    profile_name="my-workload",
    build_profile=calibrate,
)
with loaded.profile_run():
    output = loaded.module(**inputs)
```

For deferred attachment (`apply_dynamic=False`), pass the profile options to
the later `enable_offload` call, at the actual execution stage. No model-specific
profile data belongs in loading arguments.

The older context/persistence helpers remain available for existing integrations,
but are not required by the named-profile API.

## VMM residency profile

VMM Fast measures how much weight memory is useful for the workload.
Its budget covers the execution slots, model arena and any component-lifetime
state accounted by the selected adapter. Activations and other GPU allocations
still need room outside that weight budget.

Use the same named-profile API with a `vmm_fast_runtime` configuration prepared
as in [integration](integration.md):

```python
result = enable_offload(
    model, config=config, component="transformer",
    profile_name="my-model-1280x704-8s", build_profile=calibrate,
)
with result.profile_run():
    run_denoise(steps=3 if calibrate else steps)
```

### Build and reuse

1. **Floor build:** run three denoising steps at the structural minimum.
   DDO observes device budget, peak usage, resident weight capacity, forward
   times and first-forward input shapes. It derives a possible larger ceiling
   while leaving a configurable margin.
2. **Ceiling build, when needed:** repeat the same workload with
   `build_profile=True`. DDO keeps the larger budget only when the gain exceeds
   both timing noise and the configured gain-per-GiB threshold. Otherwise it
   saves the floor. If the floor build already reaches a final decision, skip
   this second build.
3. **Use:** omit `build_profile` or set it to `False` and keep the same name.
   DDO creates the native arena with the saved recommendation at attachment.
   It does not retune residency from Python during generation.

Build timing synchronizes forwards; it is calibration overhead, not ordinary
inference latency. At least two steady forwards after the first are needed.
The profile report identifies a pending or final decision. A further build after
a final decision starts a new calibration. Keep inputs, attention, component
routes and transport capacities identical between the two builds.

Defaults are a 0.25 GiB margin and a 2% gain per extra GiB threshold.
An explicit positive residency budget takes precedence during use and is
refused during calibration. Without a usable profile, VMM Fast starts at the
structural floor. The protected `nvme_baseline` uses a fixed budget, not this
calibration.

On Windows, DDO can reduce a saved budget to the current local GPU budget at
attachment, without raising it or going below the structural minimum.
A profile does not guarantee that a changed workload or system state fits;
Windows paging can slow execution and allocation can still fail.

### Identity and report

VMM profiles use the separate `ddo-vmm-residency-v1-` file prefix and identity
version 5. Identity includes the name, model class, component, execution-unit
level, platform, GPU/device memory, host RAM, native library hash and weight
transport capacities. First-forward shapes are checked separately against the
saved observation. Changed shapes or transport require rebuilding.

DDO does not infer every model input from the profile name. Include resolution,
duration and relevant feature options in the name and keep model/adapters and
precision matched. Linear and VMM profiles never share a file, even with the
same name.

Read `runtime.vmm_residency_profile` through
`result.metrics_snapshot().as_dict()`, or use `format_profile()`.
See [metrics](metrics.md) and the selected platform/model
[benchmarks](https://devaiexp.github.io/ddo-examples/docs/benchmarks/).
