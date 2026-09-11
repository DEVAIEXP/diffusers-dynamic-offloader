# Capacity profiles

DDO capacity profiles record the result of a real workload so later runs can
recognize a previously validated range. They are intentionally observations,
not a model-specific execution engine: the host still owns the actual prompt,
denoise step, decoder order, and cleanup boundaries.

## Storage and matching

By default profiles are saved under `~/.ddo/profiles`. Set `DDO_PROFILE_DIR`
to use another directory, for example in a benchmark or CI environment.

The host supplies a JSON-safe identity context. A good context includes the
model/revision or signature, GPU and VRAM, effective DDO configuration, dtype,
and the workload family (resolution, attention/modality options, LoRA set,
and inference settings). Keep the varying capacity metric out of that identity:
for video, frame/token count should be an observation rather than a key field.
That lets a profile validated at one duration warn about a larger duration.

DDO hashes the normalized context for the filename, but keeps the readable
context inside the JSON for inspection. Invalid, mismatched, or old-schema
files are ignored safely.

## Host integration

Load automatically before the sensitive phase:

```python
from diffusers_dynamic_offloader import (
    get_dynamic_offload_profile_capacity,
    load_dynamic_offload_profile,
    record_dynamic_offload_profile,
)

profile = load_dynamic_offload_profile(context)
limits = get_dynamic_offload_profile_capacity(profile, "tokens") if profile else None
if limits and requested_tokens > limits["max_success"]:
    print("Requested workload exceeds the largest validated profile; it may OOM.")
```

After a successful calibration or generation, record it explicitly:

```python
record_dynamic_offload_profile(
    context,
    {
        "status": "success",
        "capacity": {"metric": "tokens", "value": requested_tokens},
        "result": {"elapsed_seconds": elapsed_seconds},
    },
)
```

For each generic metric, DDO keeps the largest successful value and the
smallest failed value, plus the last 20 observations. Hosts may also record a
failure if they catch an OOM and can clean up safely.

An observation may carry a JSON-safe `recommendation` mapping. DDO persists
and returns it through `get_dynamic_offload_profile_recommendation(...)`; the
host decides which values are safe to apply before it attaches offload hooks.
This keeps the library model-agnostic while allowing a staged runner to store
an exact-workload preset separately from its wider capacity history.
