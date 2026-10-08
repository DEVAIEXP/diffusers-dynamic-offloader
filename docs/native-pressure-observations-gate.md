# Native cache pressure observations candidate

Status (2026-10-05): isolated build, CPU contracts, two approved small CUDA
contracts and one approved video completed. No default promotion or budget change.

## Scope and evidence

The preceding video candidate recorded 29.44 s/step versus 26.09, with more
native pressure evictions despite the same configured cache capacity. Existing
artifacts could not distinguish RAM from commit pressure. This change records
bounded native observations only; it does not assert a fix or a causal explanation.

- DLL: `ddo-vmm-allocator/build-pressure-observations-20261005/Release/ddo_vmm.dll`
- SHA256: `15D349814F4E6D19C3F055D91BB91F018125E2306ACE8FB60E484FCD00976323`
- MSVC 19.44, CUDA toolkit 12.8.93, Release, span pool ON, dispatcher and test hooks OFF.
- Eight isolated C++ CPU contracts passed, including synthetic RAM/commit/both/
  unavailable transitions, reclaim accounting and bounded storage.
- 59 Python tests and two subtests passed with the candidate selected: pressure
  binding, lifecycle, platform, presets and public report contracts. The initial
  invocation omitted the required DLL environment setting and failed one setup
  assertion; the corrected invocation passed all contracts.
- Against the accepted old DLL: six pressure/report CPU tests passed and the
  optional-export ABI test skipped as unavailable. No CUDA initialization.
- Existing cache stats ABI/version preserved; new export is optional. Old metrics
  stay absent rather than becoming invented zeros. Catalog: [metrics](metrics.md).

Accepted default DLL and pool candidate hashes remain unchanged. The local
Diffusers activation-lifetime patch remains a diagnostic candidate, not promoted.
Linear/NVMe paths, image environment and source admission policy are unchanged.

## Approved consolidated sequence

Hypothesis: observations identify the pressure cause without changing storage,
registration recycling, ownership or results. Never force global memory pressure.
First run two small contracts from DDO with its CUDA development environment:

```powershell
$previousPressureLibrary = $env:DDO_VMM_LIBRARY_PATH
try {
    $env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-pressure-observations-20261005\Release\ddo_vmm.dll'
    .\.venv-vmm\Scripts\python.exe -m pytest -p no:cacheprovider ddo-vmm-allocator/tests/test_source_cache.py::test_pressure_observations_snapshot_on_live_cache ddo-vmm-allocator/tests/test_source_cache.py::test_persistent_storage_survives_registration_recycling_without_reread -q
    if ($LASTEXITCODE -ne 0) { throw 'Stop: pressure/storage gate failed.' }
} finally { $env:DDO_VMM_LIBRARY_PATH = $previousPressureLibrary }
```

Require two passes, valid versioned observations, original thresholds, retained
bytes after unregister/re-register, correct payload and no reread/CUDA error.
Only if both pass, run one matched case from `E:\LTX-2`:

```powershell
$previousPressureAudio = $env:LTX_USE_AUDIO_CONDITION
$previousPressureBlocking = $env:CUDA_LAUNCH_BLOCKING
try {
    $env:LTX_USE_AUDIO_CONDITION = '0'
    $env:CUDA_LAUNCH_BLOCKING = $null
    .\run_ddo_vmm_video.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-pressure-observations-20261005\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 5 --steps 8 --ddo-vmm-resident-weight-budget-gb 2 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024
} finally {
    $env:LTX_USE_AUDIO_CONDITION = $previousPressureAudio
    $env:CUDA_LAUNCH_BLOCKING = $previousPressureBlocking
}
```

Keep BF16, seed 43, 121 frames, 24 fps, no input audio, generation monitoring off,
1.5 GiB staging and 24.625 GiB source/registration ceilings. The video environment
still uses the local lifetime-patched Diffusers fork; no source rollback is implied.
Require completed generation/cleanup/save, pressure fields in public JSON, zero
pool failures/quarantine/temporary fallback/plan mismatches and no memory anomaly.
Compare timing, pressure causes, retained bytes and bypass bytes with both prior
runs. No automatic budget increase, additional repetition or default promotion;
stop at the first failed gate or memory anomaly. Performance impact and historical
causality remain unproven until measured, even if the contracts pass.

## Executed evidence

The user approved the sequence. Initial pytest setup failed because the default
temporary directory was inaccessible; no contract assertions ran. The same two
contracts passed in 4.76 seconds using a fresh `.internal/pytest-pressure-<uuid>`
base directory. No workload, memory or CUDA parameters changed for this repair.

The conditional video exited zero and completed generation, cleanup, decode and
save. Scoped environment values were restored; the local Diffusers source and
candidate DLL hashes matched the recorded values. Artifacts in the sibling host:
`outputs/ddo_native_video_20261005_221330_003/`, MP4 and `metrics/*.json`.

| Measurement | Pool candidate, retaining forward | Lifetime candidate | Pressure-observations candidate, lifetime forward |
| --- | ---: | ---: | ---: |
| Progress final average, s/step | 26.09 | 29.44 | 28.60 |
| Generation console seconds | 209.4 | 236.1 | 229.4 |
| Full run JSON seconds | 282.7551 | 310.9901 | 303.0063 |
| Retained source bytes | 22604243016 | 16912135472 | 16979244336 |
| Peak committed source bytes | 23085449216 | 18387828736 | 17112760320 |
| Pressure evictions | 371 | 480 | 476 |
| Pressure-released bytes | 24897388544 | 32212254720 | 31943819264 |
| Pressure-blocked requests | 10702 | 14601 | 14706 |
| Bypass/source-read route bytes | 102139290488 | 139096044040 | 140737816120 |
| Native batch-submit wall seconds | 49.0033472 | 84.2293399 | 80.6325841 |

New observations establish the pressure cause for this run, not retrospectively
for the preceding runs: 31317 valid samples, zero failed samples, 241 entries and
241 recoveries. All 241 entries were commit-limited; RAM/both/unknown counts were
zero. Minimum sampled available physical RAM was 18449465344 bytes (17.1824 GiB),
whereas minimum available commit was 840253440 bytes (0.7825 GiB). At the last
entry, RAM was 18789392384 bytes and commit 1116717056 bytes, below the unchanged
1140850688-byte low threshold; recovery commit was 1219424256 bytes, above the
1207959552-byte high threshold. Last-episode reclaim was 67108864 bytes. Reclaim
was blocked by live ownership 16830 times; failed reclaim count was zero.

Thus pressure discarded cache storage despite substantial physical RAM headroom.
Increasing the configured cache ceiling cannot remove this commit guard. These
observations do not attribute commitment to the native cache, driver, other DDO
storage or other processes, nor demonstrate physical pagefile/disk traffic.

Read-only Windows CIM queries after completion found a fixed 32768 MiB pagefile
on E:, automatic management disabled, CommitLimit 103000539136 bytes and current
CommittedBytes 29389254656 bytes. Current physical available bytes were
45280845824. Pagefile reported current/peak usage 2217/7879 MiB. These are
post-run observations, not sampled denoise peaks or proof of which allocation
caused pressure. No pagefile, registry or OS policy was changed.

Safety observations: 192 materializations, 512 execution-buffer uploads, zero
temporary fallback tensors, plan mismatches, source-read failures, pool failures
and quarantine bytes. Tickets collected from two to zero. Pool wait was
0.0001472 seconds; mapped/pending bytes were 2017460224/1881044480. Driver usage
after generation was 4.95 GiB absolute. Tracker RAM observations reached
25.7253 GiB, including encoding; monitoring during generation stayed disabled,
so these are not continuous denoise peaks. The encoder executed Diffusers group
offload, not native encoder ownership. CPU media inspection decoded all 121
1280x704 frames at 24 fps, found one audio stream and nonconstant first-frame
pixels. Full-model numerical equivalence and perceptual quality were not tested.

The diagnostic passed its selected safety/reporting gate, not a speedup gate.
The 28.60 s/step result is still slower than 26.09. Do not promote the lifetime
patch or revise memory policy from a single run. Next work should attribute and
reduce avoidable commitment, or explicitly isolate environment commit capacity;
it must not raise cache budgets or disable guards automatically. Any additional
CUDA run, benchmark parameter change or system configuration change needs new
approval. Accepted default DLL and image environment remain unchanged.

## Follow-up static commitment audit

Read-only inspection after the approved run separated ownership categories:

- `native/src/host_platform.h::allocate` commits anonymous Windows storage with
  `VirtualAlloc(MEM_RESERVE | MEM_COMMIT)`. The progressive source cache commits
  only admitted slots; its lazy ring does not commit the entire configured
  24.625 GiB capacity at construction. `admit_slot` increments committed bytes
  after successful commitment. Registering existing storage does not create a
  second explicit DDO host allocation. Driver registration overhead is unmeasured.
- Native eager staging is separate committed/registered storage, 1.5 GiB in this
  workload. `dynamic_offload.py` sets the PyTorch staging pool to `None` when
  native transport is selected; no parallel PyTorch staging ring is constructed.
- Transformer bindings reported 3320 file-backed bindings / 37965068800 bytes
  and zero CPU-source bindings/bytes. Attachment replaces selected CPU parameters
  with native views and retains `_FileTensorDescriptor` metadata. The safetensor
  registry holds addresses/descriptors, not tensor owners. This does not certify
  every unrelated loader object or other component's lifetime.
- Native execution buffers and PyTorch backing use CUDA Driver allocation and
  mapping. Model virtual address size is not physical backing size: transient
  view setup maps the same bootstrap page and releases those temporary mappings.
  Arena, execution slots and span-pool backing are separate owners; their effect
  on Windows driver commitment cannot be reconstructed from byte counters alone.
- Encoder, connectors and auxiliary group-offload routes still use PyTorch's
  pinned host allocator. The video runner already has an explicit cache-release
  helper, but early cleanup calls leave `empty_pinned_host_cache=False`; its
  first executed cache-release boundary is after generation, before VAE decode.
  That late boundary recorded 3.248 GiB allocated before release and zero after.
  The artifact does not measure how much was idle at the earlier transformer
  boundary, so 3.248 GiB is not a proven recoverable pre-denoise saving.

The same artifact's `active_gb` counter was 37.4803 before and after that release,
despite allocated bytes falling to zero. Do not interpret it as retained pinned
storage or infer an additional 37 GiB leak. The installed Torch
`2.12.1+cu132` header defines allocated bytes as all allocator-owned blocks and
active bytes as a subset. Its no-stream free branch omits active-counter decrement,
and event processing decrements active bytes by the requested-size selector,
including the all-events sentinel -1. Those header anomalies are consistent with
the observation but are not verification of the compiled binary's internals.
Do not patch Torch or use the inconsistent active counter for admission policy.

Microsoft's [MEMORYSTATUSEX contract](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex)
defines `ullAvailPageFile` as the calling process's available commitment, at most
system-wide headroom, not free space in the pagefile. A subsequent CPU-only
query in the video's environment found process/system commit limits both
103000539136 bytes and available commit both 73309822976 bytes. The process was
in a job, but that snapshot did not exhibit a lower process-specific headroom.
It cannot exclude a transient process restriction during the earlier workload.

The bounded next correction candidate is an explicit, opt-in DDO-owned lifecycle
helper to release only unused PyTorch host-cache blocks after previous components
are detached and their transfers complete, before native transformer setup.
It must not clear the native persistent cache, drop live tensors, synchronize
each denoise step, change residency/admission budgets or alter stable image/
linear/NVMe defaults. Report allocation counters before/after and reject inconsistent
active-byte interpretations. A selected video gate must measure actual recovered
commit, retained source bytes, pressure counts and timing; no benefit is assumed.
Additional driver/process attribution may still be needed if reclaim is negligible.
No runtime code, binary, environment, OS policy or CUDA execution changed in this audit.
