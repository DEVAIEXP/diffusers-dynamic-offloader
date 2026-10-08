# Explicit PyTorch host-cache boundary

Status (2026-10-05): implementation, CPU contracts, selected small CUDA gate and
approved matched video passed. Observed improvement versus the pressure control;
not repeat-validated causality or a default promotion.

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
delta rather than claiming causal attribution from headroom alone. Raw historical
`active_gb` evidence is unchanged; only new reports flag inconsistency.

## Integration and CPU evidence

The sibling video host opts in with
`--release-pytorch-host-cache-before-transformer`. It is disabled by default and
rejects a non-native-VMM route. The single call runs after prepared VAE/audio
conditions are released and before transformer DDO setup. The runner delegates
cleanup/reporting to DDO; existing late decode cleanup is preserved.

No image host, stable linear preset, protected NVMe path, native DLL, Diffusers
forward or OS memory policy changed for this candidate.

- DDO: 60 CPU tests plus two subtests passed (host-cache, pressure reporting,
  cleanup-on-error, presets and bootstrap). The first combined invocation's
  bootstrap fixtures hit the known inaccessible pytest default temp directory;
  the unchanged contracts passed with a fresh workspace base directory.
- Video host: 13 CPU tests plus 12 subtests passed, including opt-in scope,
  public report consumption and the single pre-transformer boundary placement.
- CPU-only Windows sampler returned RSS/private/headroom observations without
  initializing CUDA. This does not validate pinned allocation release on GPU.

## Approved consolidated gate

First, from DDO using its `.venv-vmm`:

```powershell
.\.venv-vmm\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_host_cache_cuda.py -q
```

Hypothesis: one explicit completion/release boundary frees an unused 1 MiB
PyTorch pinned block while preserving another live 1 MiB block and its pending
CPU-to-GPU copy. Require one pass, correct payload, live storage still valid,
reported allocation reduction and no CUDA error. Stop at any failure.

Only if it passes, from `E:\LTX-2`, use the existing pressure-observation DLL:

```powershell
$previousBoundaryAudio = $env:LTX_USE_AUDIO_CONDITION
$previousBoundaryBlocking = $env:CUDA_LAUNCH_BLOCKING
try {
    $env:LTX_USE_AUDIO_CONDITION = '0'
    $env:CUDA_LAUNCH_BLOCKING = $null
    .\run_ddo_vmm_video.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-pressure-observations-20261005\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 5 --steps 8 --ddo-vmm-resident-weight-budget-gb 2 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer
} finally {
    $env:LTX_USE_AUDIO_CONDITION = $previousBoundaryAudio
    $env:CUDA_LAUNCH_BLOCKING = $previousBoundaryBlocking
}
```

The only selected host behavior change is the opt-in boundary. Preserve BF16,
seed 43, 121 frames at 24 fps, no input audio, generation monitoring off,
1.5 GiB staging, 24.625 GiB cache/registration ceilings and local lifetime-patched
Diffusers. DLL SHA256:
`15D349814F4E6D19C3F055D91BB91F018125E2306ACE8FB60E484FCD00976323`.

Hypothesis: earlier idle pinned-cache release increases usable commitment and
reduces native source discard. Compare the public boundary before/after values,
241 commit-pressure entries, retained bytes 16979244336, bypass bytes
140737816120, 28.60 s/step, 229.4 s generation and 303.0063 s total from the
previous matched run. Require generation/cleanup/save, complete report, zero
pool failures/quarantine/temporary tensors/plan mismatches and no memory anomaly.
The best pool result remains 26.09 s/step, not this control. Do not promote on
negligible recovery or a slower/unsafe result; no automatic budget changes,
additional repetitions or pagefile adjustments are authorized.

## Executed evidence

The user approved the sequence. The small CUDA pinned-block/transfer contract
passed in 4.67 seconds using DDO's `.venv-vmm`. The conditional video then exited
zero with the documented opt-in flag. Scoped environment values were restored.
DLL and local Diffusers forward hashes matched their recorded values. No other
CUDA cases, workload/budget changes or system configuration changes ran.

Artifacts in the sibling host:
`outputs/ddo_native_video_20261005_223421_903/`, with MP4 and `metrics/*.json`.

| Measurement | Pressure-observation control | Early host-cache release |
| --- | ---: | ---: |
| Final progress average, s/step | 28.60 | 26.23 |
| Generation console seconds | 229.4 | 210.4 |
| Full run JSON seconds | 303.0063 | 283.1354 |
| Retained native source bytes | 16979244336 | 20322541640 |
| Peak committed native source bytes | 17112760320 | 20535312384 |
| Commit-limited pressure entries/recoveries | 241/241 | 206/206 |
| Pressure evictions | 476 | 413 |
| Pressure-released bytes, cumulative | 31943819264 | 27715960832 |
| Pressure-blocked requests | 14706 | 12416 |
| Bypass/source-read route bytes | 140737816120 | 117728656640 |
| Native batch-submit wall seconds | 80.6325841 | 57.9781631 |

The public boundary report recorded 3487499715 allocator bytes before release
and zero afterward: 3.248 GiB relinquished in 0.4058957 seconds. Process private
bytes decreased from 49128091648 to 45635809280; available process commitment
increased from 23854436352 to 27352002560 bytes at the adjacent samples. RSS fell
from 6675820544 to 3190366208 bytes. These CPU observations agree in scale with
the allocator reduction, but headroom also includes external activity. Raw
active bytes remained 40244138246 and were explicitly flagged inconsistent/null
for validated interpretation, not treated as retained storage.

Generation improved by 19.0 seconds (8.3%); total by 19.8709 seconds (6.6%).
Final retained native cache grew from 15.8132 to 18.9268 GiB, bypass-route bytes
fell about 16.3%, and pressure episodes decreased. The selected workload result
supports the usefulness of this boundary. It is one matched sequential run,
not an isolated or repeat-validated causal estimate. The earlier retaining-
forward pool result remains slightly faster: 26.09 s/step, 209.4 s generation,
282.7551 s total. Do not claim a new overall performance record.

All 206 entries were still commit-limited, with zero RAM/both/unknown entries or
failed observations. The cache ceiling and low/high thresholds are unchanged.
The remaining pressure must not be hidden by raising budgets or disabling guards.
Post-release private bytes were still about 42.5 GiB before native transformer
setup, with approximately 3 GiB RSS: virtual commitment and resident host pages
are distinct. This observation does not identify whether loader storage/mapping,
driver backing or other process allocations account for that remaining private
commitment. Investigate ownership before proposing another policy change.

Safety observations matched the prior case: 192 materializations, 512 execution
buffer uploads, zero temporary fallback tensors, execution-plan mismatches,
source-read failures, pool failures and quarantine bytes. Tickets collected from
two to zero. Pool mapped/pending bytes were 2017460224/1877633536. Driver usage
after generation was 4.95 GiB absolute. Tracker RSS observations reached
25.7254 GiB including encoder execution; generation monitoring was disabled, so
these observations are not continuous denoise peaks. Encoder still executed
Diffusers group offload, not native ownership. CPU media inspection decoded all
121 frames at 1280x704/24 fps, found one audio stream and nonconstant first-frame
pixels. Full-model numerical equivalence and perceptual quality were not assessed.

Keep this helper available as an explicit video opt-in. No image/frozen-runtime
regression is claimed from the selected video gate, and no default was changed.

## Follow-up ownership audit (CPU only)

The boundary is before native attachment, not before model loading: the host
loads the transformer in Step 2, prepares auxiliary conditions in Step 3b and
then invokes the helper before Step 3c. Its 42.5 GiB private-byte observation
therefore includes the already-loaded CPU transformer. Do not classify all of
that value as leaked prior-phase storage or as persistent denoise overhead.

The recorded loader backend was unset. Installed safetensors 0.8.0 defaults to
`mmap`; its tagged PyTorch binding uses `UntypedStorage.from_file(shared=False)`.
On Windows, a copy-on-write file view can charge commitment for the entire view
without resident copies of every page, as documented by
[Microsoft](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffile).
See the tagged
[safetensors binding](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/src/lib.rs).
This is a plausible contributor, not an allocation census of the video process.

A bounded CPU-only probe in the video virtual environment created a temporary
32 MiB file and mapped it with `torch.UntypedStorage.from_file(shared=False)`.
Process private bytes increased by 33619968 bytes (32 MiB plus 64 KiB), while RSS
increased by only 40960 bytes. Deleting the storage and collecting garbage
returned private bytes exactly to the pre-mapping sample. CUDA remained
uninitialized throughout; the temporary file/directory were removed. No model,
native DLL or benchmark was executed, and no runtime policy was changed.

The native adapter replaces eligible CPU parameters with stable CUDA views and
retains immutable file descriptors rather than those CPU tensors. However, a
single remaining tensor can keep an entire shard storage alive. Current video
evidence lacks adjacent private/headroom observations after attachment, so it
does not establish whether these loader mappings were released or retained into
denoise. The next targeted diagnostic should compare before-load, loaded,
post-attachment and generation boundaries and identify surviving CPU storages
by shard before changing loading or cache policy. Do not switch to shared
writable checkpoint mappings, force-release live storage, or infer that `pread`
reduces commitment without a bounded validation of its copy/memory costs.

## Prepared phase diagnostic

DDO now exposes the read-only `inspect_host_memory` report documented in the
[metric catalog](metrics.md). The sibling host enables it only through the new,
default-off `--inspect-host-memory-phases` native-video diagnostic. Five reports
are saved under `dynamic_offload_reports.host_memory_<phase>`:
`before_transformer_load`, `transformer_loaded`, `before_transformer_attach`,
`transformer_attached`, and `after_generation`. The before-attach report follows
the existing opt-in host-cache release. All inspection logic resides in DDO;
the host only passes the component and consumes the public report.

CPU validation: 20 DDO tests passed for host memory, host-cache release and
pressure reporting; 16 host tests passed (including their existing subtests).
The new census contracts cover shared storage, tiny views retaining a larger
backing, detached provenance/no retained tensor owner, bounded output, unavailable
provenance, and no CUDA initialization/waits. The older host test was adjusted
to allow the read-only observation between cleanup and setup while preserving
their order. These results do not validate a CUDA run of the new diagnostic.

Proposed single selected run, requiring fresh user approval: repeat the executed
eight-step command above with **both** opt-in flags:
`--release-pytorch-host-cache-before-transformer --inspect-host-memory-phases`.
Keep the exact no-input-audio scope, pressure DLL, local forward, resolution,
frames, seed, steps, chunks and budgets. No additional test cases or parameter
changes are proposed. The diagnostic incurs phase observation overhead; it is
not an instrumentation-free speed record.

Hypothesis: a substantial part of pre-attach private commitment belongs to
loader mappings. Compare private bytes/headroom and unique CPU storage extents
after attachment and generation, and examine the largest surviving shard
owners. The hypothesis may be rejected without declaring a failed run. Require
all five public reports, successful generation/cleanup/save, zero fault/pool/
source failures, temporary fallbacks, quarantine and plan mismatches, and no
memory anomaly. Stop rather than extend the sequence if a safety criterion
fails. Do not infer causality solely from available-commit changes or assume an
unregistered/closure-owned tensor is covered by this census.

## Approved phase diagnostic result

The user approved the single eight-step sequence with both flags. It exited zero
and produced sibling-host artifact `outputs/ddo_native_video_20261005_225817_983/`.
DLL and local forward SHA256 values matched the earlier run. Scoped no-input-
audio and launch-blocking environment values were restored. No additional CUDA
cases, budget changes or default promotions ran.

| Phase | Private GiB | RSS GiB | Available process commit GiB | Registered CPU storage bytes |
| --- | ---: | ---: | ---: | ---: |
| Before transformer load | 8.1216 | 5.1470 | 59.5332 | Not inspected |
| Transformer loaded | 43.5684 | 5.1608 | 24.0877 | 37977676288 |
| Before attach, after early cache release | 42.4982 | 2.9679 | 25.1697 | 37977676288 |
| Transformer attached | 44.0165 | 4.4899 | 23.6934 | 8257536 |
| After generation | 65.0445 | 23.2540 | 3.1875 | 8257536 |

All five reports were observed, without storage inspection errors. Their total
reported sampling/inspection time was 0.0662413 seconds. The loaded-to-attached
CPU extent count fell from 4186 to 192, but private commitment did not fall by
the transformer payload. The remaining registered CPU parameters include
`scale_shift_table` values of 73728 bytes; sampled examples resolve to loader
shards. These exceed the existing small-tensor placement threshold and are not
managed Linear parameters. The fast native store reported 3320 file-backed
bindings and zero retained CPU-source bindings: the suspected surviving owners
are outside those managed bindings, not evidence of a native CPU weight cache.

Final progress averaged 26.35 s/step, generation took 211.4 seconds and JSON total
was 283.9525 seconds, versus 26.23 / 210.4 / 283.1354 in the preceding early-
release case. This near-matched diagnostic is not an instrumentation-free speed
record or broad regression proof. Native retained source bytes were 19999143216
(18.6257 GiB), peak committed bytes 20333985792, bypass-route bytes 118439786536
and native batch-submit time 59.3932947 seconds. All 213 pressure entries and
recoveries were commit-limited; valid/failed observations were 31272/0. Minimum
sampled available physical/commit bytes were 15070859264/902270976. Guard and
cache ceilings remained unchanged.

Safety passed: zero fault/guard/CUDA/capacity/source-read/pool failures, temporary
tensors, plan mismatches, quarantine bytes and failed reclaims. Materializations
and execution-buffer uploads remained 192/512; completion tickets collected
2 to 0. Generation, native cleanup, decode and save completed. CPU media
inspection decoded all 121 frames at 1280x704/24 fps, found one generated audio
stream and nonconstant first-frame pixels. Quality/full-model equivalence were
not assessed. Global tracker RSS observations reached 25.7260 GiB including the
encoder; generation monitoring remained off, so this is not a continuous peak.

### Bounded CPU mapping follow-up

The exact video environment's safetensors bulk `load_file` was checked using a
temporary synthetic file of 33620136 bytes with one large payload and one
65536-byte tensor. After discarding the loader dictionary except for the small
tensor, its visible storage was 65536 bytes while the file remained mapped.
After cloning the small tensor and dropping its mapped view, the process file-
mapping enumeration no longer contained that file. CUDA stayed uninitialized.
This confirms hidden parent-mapping retention is not reflected by slice
`nbytes()`. It does not prove that all four real-model shard mappings survive or
that cloning the real residual parameters will recover their full commitment.

The same CPU probe's adjacent private-byte samples did **not** show immediate
commit recovery: 1316999168 before load, 1350692864 with the small mapped view,
1351888896 with only the independent clone, and 1351909376 after its release.
Earlier direct `get_tensor` checks likewise did not prove commit recovery. Do not
hide that limitation or interpret disappearing mappings as measured recovered
commit. One exploratory mapping enumeration used an unsupported `size` field
and failed; its temporary file/directory were removed after that process exited.
All synthetic probe artifacts are removed, and no checkpoint was modified.

Next correction candidate: detach bounded, source-backed residual CPU parameters
from loader ownership after successful fast-native attachment, without changing
GPU placement/residency or the protected routes. Preserve Parameter identity,
alias/value/dtype semantics and immutable managed-source provenance, and fail
closed on unsupported ownership. Establish the selected residual byte/shard
scope and CPU lifetime/value contracts before proposing another CUDA gate.
No such correction has been implemented or performance-validated by this run.

### Residual ownership correction candidate (2026-10-06)

The subsequent CPU-only correction is now implemented as the explicit
`DynamicOffloadApplyResult.detach_auxiliary_cpu_sources(enabled=True)` boundary.
Default behavior is unchanged. It requires an attached fast-native component,
zero native CPU-source bindings/faults/execution passes, eval mode and exclusive
caller ownership. It clones only source-backed registered CPU tensor spans,
once, up to 16 MiB. All replacements are staged before mutation; admitted exact
aliases preserve tensor identity, values, CPU device and dtype. Unsupported
partial/strided/pinned/subclassed aliases and whole-component ownership refuse
the operation. Metadata relocation is atomic and mutation rolls back on failure.
Managed arena weights, native descriptors, GPU placement and cache/residency
ceilings are unchanged. Arbitrary escaped storage users remain outside scope.

The host consumes the public report and records adjacent CPU host samples. The
video runner adds default-off `--detach-transformer-auxiliary-cpu-sources` inside
its native cleanup-on-error context before the attachment observation and first
forward. Existing phase observations remain in their five-phase sequence.
`format_profile()`/JSON expose status, copied payload/alias/shard counts and
host samples; copied bytes are not proof of recovered commitment.

Validation: 66 DDO CPU tests and two subtests passed across residual ownership,
host observations, host cache and presets; 19 host CPU contracts passed.
BF16/FP32 values, exact aliases, identity, unsupported layouts, bounded copies,
readiness refusal, metadata rollback and disabled/idempotent behavior are covered.
No CUDA run, DLL rebuild, budget/default change or image benchmark was performed.
The selected video evidence remains 26.35 s/step / 211.4 seconds generation;
this candidate has no accepted real-model performance result yet.

Proposed next gate, requiring separate user confirmation: repeat exactly the
selected 1280x704, five-second, eight-step, seed-43, no-input-audio workload with
the pressure-observation DLL, 2 GiB resident weight ceiling, existing 1024-token
chunking and early host-cache release, adding only the ownership opt-in flag.
Expected selection from the prior census is 8257536 bytes across 192 extents;
report actual counts/shards, adjacent host commitment, pressure/cache retention,
generation and total times. Stop on refusal or any native failure/memory anomaly;
do not expand ceilings or run additional cases. Mapping/commit recovery and any
speed improvement must be measured, not inferred from successful CPU copies.

### Approved residual ownership video gate: passed (2026-10-06)

The user confirmed the single proposed matched run. Command, from the sibling
video host, with `LTX_USE_AUDIO_CONDITION=0` scoped/restored by the launcher shell:

```powershell
.\run_ddo_vmm_video.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-pressure-observations-20261005\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 5 --steps 8 --ddo-vmm-resident-weight-budget-gb 2 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources
```

Artifact: `E:\LTX-2\outputs\ddo_native_video_20261006_091222_765\`, including
the saved MP4 and its public JSON under `metrics/`. DLL SHA256 remained
`15D349814F4E6D19C3F055D91BB91F018125E2306ACE8FB60E484FCD00976323`;
local Diffusers transformer SHA256 remained
`581703C58AF60D1D124C087265465E13C585D2B8CFB486C510A1A5B082F265E8`.

The boundary selected exactly 8257536 bytes across 192 CPU owners/extents,
from all four source shards, with zero unbound CPU groups. It took 0.0710587
seconds. Private commitment fell from 47266713600 to 9218457600 bytes in adjacent
samples: 35.4352 GiB recovered. Available commit rose by 38049525760 bytes.
After-generation private bytes were 38267658240. CPU payload/identity remains
unchanged; recovering parent commitment does not mean 35 GiB of CPU payload was
copied. Earlier synthetic probes and limitations remain recorded unchanged.
The operation's generic report continues to say `commit_recovery_proven=false`;
the selected host samples supply recovery evidence separately.

Final progress averaged 24.87 s/step (not the last EMA, 24.58). Generation was
199.7 seconds and JSON total 276.9484, against 26.35 / 211.4 / 283.9525 in the
preceding phase diagnostic. Native cache retained 26362339400 bytes (24.5518 GiB)
within the unchanged ceiling, bypass bytes fell to 92894603200, and batch-submit
time fell to 20.4131390 seconds. Pressure entries/recoveries fell from 213/213 to
0/0 with 30841 valid observations and no failed observations. Minimum available
RAM/commit observations were 10427731968/31595470848 bytes.

Safety criteria passed: required fault/guard/CUDA/capacity/source-read/pool
failures, temporary tensors, mismatches, quarantine and failed reclaims were
zero; completion tickets were 0 to 0 and materializations/buffer uploads 192/512.
Cleanup, decode and save succeeded. CPU media inspection decoded 121 frames at
1280x704/24 fps, one generated audio stream, and first-frame pixel standard
deviation 51.4805808. No quality/full-model numerical equivalence claim.

Global sampled tracker RSS increased to 29.1745 GiB as more cache data stayed
resident; this is not a continuous generation peak. Driver usage after pipeline
return was about 4.95 GiB absolute / 3.93 GiB over baseline. The encoder still
executed through Diffusers group-offload. The single sequential improvement is
not repeat-validated causality, general model support or image regression
validation. Defaults, ceilings, DLL, protected runtimes and the image runner
remain unchanged. Keep the new boundary opt-in. This approved sequence is
complete; do not run further CUDA cases or promote defaults automatically.
