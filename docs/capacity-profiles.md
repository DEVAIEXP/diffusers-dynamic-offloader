# Named workload profiles

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

## Identity and storage

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

Calibration requires the dynamic CUDA linear runtime. It forces the resident
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
A manual positive `resident_module_budget_gb` overrides profile reuse, and
`max_resident_module_budget_gb` caps the automatic recommendation when positive.

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
