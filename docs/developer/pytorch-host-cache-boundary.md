# Explicit PyTorch host-cache boundary

## Public API and scope

```python
from diffusers_dynamic_offloader import release_pytorch_host_cache

# Detach completed prior components and drop caller-owned references first.
report = release_pytorch_host_cache(
    enabled=True, phase="before_transformer", device="cuda:0"
)
tracker.record_ddo_metrics(report, component="host_cache_before_transformer")
```

The default `enabled=False` is a no-op. CUDA not initialized or an unsupported
Torch host-cache API also produces an explicit no-op report. This helper never
initializes CUDA. An executed boundary collects Python garbage, waits once for
the selected device's transfers, then asks PyTorch to release unused pinned host
blocks. Live blocks and pending blocks on other devices remain allocator-owned.
It never clears native source storage, changes budgets or runs inside denoise
steps. Synchronization/release failures propagate; they are not reported as
successful cleanup. Do not use concurrently with component loading or inference.

The operation currently uses Torch's optional private
`_accelerator_emptyHostCache` API because the selected build exposes it. Missing
support is not a request to mutate a different allocator or simulate release.
PyTorch CUDA host memory is distinct from DDO's native registered source/staging.
Windows memory observations are best-effort and do not affect policy.

## Public report

Route: `pytorch_host_cache_release`; main DDO report schema remains version 1.
Fields under `setup.host_cache_release`:

| Field | Unit / collection | Interpretation |
| --- | --- | --- |
| `phase`, `status`, `released` | label / boundary | Requested phase and executed/no-op status; released means the release call completed, not that a byte count was proven. |
| `elapsed_seconds` | seconds / operation | Includes GC, one completion boundary, sampling and release; not denoise compute time. |
| `before`, `after` | allocator snapshot / before-after release | Status, allocated bytes, reported active bytes, validated active bytes and consistency flag. Errors/unsupported counters are unavailable, not zero. |
| `allocated_bytes` | bytes / allocator snapshot | PyTorch-owned pinned blocks including cached and live; not native host storage or total process memory. |
| `reported_active_bytes`, `active_bytes`, `active_counter_consistent` | bytes / allocator snapshot | Preserve reported active counter; validated active is null if negative/noninteger/greater than allocated or absent. Never use inconsistent values for admission. |
| `allocator_bytes_delta` | bytes / before minus after | Signed observed change, null if unavailable. Not guaranteed OS commit recovery; concurrent allocation can invalidate attribution. |
| `host_before`, `host_after` | CPU observations / before-after release | Process RSS; Windows additionally process private bytes, physical available and process-available commit. Missing fields/errors remain unavailable. Available commit uses GlobalMemoryStatusEx, not pagefile free disk space. |

Host observations are snapshots, not continuous peaks. Changes in available
commit can include other processes/driver activity; compare with the allocator
delta rather than claiming causal attribution from headroom alone.

## Integration

Call only at a quiescent phase boundary after dropping completed component
references. The result is a public DDO report; hosts can print or serialize it
without reading allocator internals. The helper does not replace native idle
trim or Windows standby purge. See [metrics](../user/metrics.md).

## Bounded auxiliary source detachment

`DynamicOffloadApplyResult.detach_auxiliary_cpu_sources(enabled=True)` is a
separate diagnostic opt-in, disabled by default. It detaches small residual
CPU loader views before the first native forward; it is not a host-cache flush.

It requires an attached `vmm_fast_runtime` component and no started execution.
If native CPU sources exist, the adapter must report every retained payload
interval, matching the native binding count. Selected residual views must not
overlap those intervals; unavailable or incomplete observations are refused.
Older adapters without this observation still support the zero-CPU-source case.
Whole-component ownership is outside this helper's scope. The default copy
ceiling is 16 MiB. Registered parameter
identity, exact aliases, values, device and dtype are preserved; unsupported
or partial aliases are refused.

Exclusive component ownership is required: escaped tensor references cannot
be discovered or safely freed. The report records copied bytes and host
observations but does not prove OS commitment recovery. Neither this helper
nor host-cache release changes GPU residency budgets or backend computation.

The implementation is in `src/diffusers_dynamic_offloader/auxiliary_sources.py`;
host-cache release is in `src/diffusers_dynamic_offloader/host_cache.py`.
