# Native allocation startup diagnosis (2026-10-06)

## Scope and preserved checkpoint

Selected modular video gates succeeded at 5, 8 and 10 seconds. Eight seconds
used 179.51 seconds on the first step, approximately 86 on the second, then
approximately 41 per step. Ten seconds used 272.92, approximately 177, then
approximately 59-60. These observations do not establish a cache/driver cause.
Aggregate weight bytes, materializations, signatures and wait-event counts
were identical; cumulative native submission time changed approximately
28 to 32 seconds, without explaining the overall denoise increase.

No allocator, residency, cache, pressure, stream ownership or admission policy
has changed. Normal builds use `DDO_VMM_ALLOCATION_DIAGNOSTICS=OFF`; the new
driver-call wrappers and pool observations are compiled out. The accepted
pressure-observation DLL remains untouched, SHA256
`15D349814F4E6D19C3F055D91BB91F018125E2306ACE8FB60E484FCD00976323`.

Diagnostic candidate:
`ddo-vmm-allocator/build-allocation-diagnostics-20261006/Release/ddo_vmm.dll`,
SHA256 `AF3654AAB0A76D1CF17FC893C57D8FF6DDE752473778B5CFAA19C1FC68F1B242`.
Configured with span pool ON, allocation diagnostics ON, source dispatcher
OFF, test hooks OFF, Visual Studio 2022 x64 and CUDA Toolkit 12.8.93.
This candidate is not promoted or CUDA/performance validated.

## Instrumentation

The existing public execution timer supplies compute-stream event boundaries
and native execution/upload/dependency windows. New process-wide counters
measure native driver create/map/access/unmap/release/reserve/address-free
calls, host time and returned failure counts, plus successful pool reuse,
new pool backings and an observed mapped-byte high-water mark.

The modular video host enables these only with `--diagnose-denoise` and passes
a generic optional step observer to the custom block. No DDO state, internal
attribute or metric name is consumed by the host. No per-step CUDA wait is
added. The normal host branch remains unchanged without the flag.
The DDO scope emits the public versioned report; see [metrics](metrics.md).

CPU boundaries do not imply GPU completion. Host driver times cover this DLL,
not all allocations by other CUDA libraries. Compute-stream intervals include
dependencies and host launch gaps; native windows are not hardware-engine
activity. The pool high-water mark excludes model arenas and is not absolute
WDDM usage. Diagnostic overhead disqualifies the run as performance evidence.

## Completed non-CUDA validation

- Release DLL and standalone pure-CPU counter contract compiled successfully.
- Pure-CPU C++ contract passed: disabled measurements preserve calls, exclusive
  enable, fixed schema/untouched invalid output, failure counts and pool peak.
- Python contracts with the actual candidate exports: 41 passed. Native
  diagnostic/timeline ABI checks read or reject inputs without CUDA init.
  Coverage includes old/unsupported binaries, scope release, existing timing,
  protected-route refusal and preserving an original model error.
- Host modular CPU contracts: 33 passed with CUDA initialization forbidden;
  the step observer is declared and fires once for each of two CPU steps.
- The accepted DLL hash above was verified unchanged. No CUDA execution,
  image GPU regression test or benchmark followed these edits.

## Proposed single real-model diagnosis, pending confirmation

Eight seconds, 1280 x 704, seed 43, same prompt and no input audio. Use four
steps and packed-latent output: capture the two slow steps and two later steps
without spending another complete eight-step run or decoding video. This is
an explicitly changed diagnostic workload, not a matched performance benchmark.
All weight/staging/source/registration ceilings and chunk sizes remain fixed.

```powershell
Set-Location E:\LTX-2
.\run_ddo_vmm_video_modular.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-allocation-diagnostics-20261006\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 8 --steps 4 --seed 43 --no-use-audio-condition --output-type latent --diagnose-denoise --ddo-vmm-resident-weight-budget-gb 2 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources
```

Hypothesis: first-step driver-allocation cost or native execution/dependency
windows explain the startup concentration. A negative result must remain
explicit: unexplained residual time is not automatically driver paging.

Acceptance: four complete GPU and CPU diagnostic boundaries, native timeline
complete with no dropped samples, finite packed outputs, original native
failure/fallback/mismatch/quarantine counters zero and successful retirement.
Driver failed-call diagnostics include transient retries and must be inspected
separately rather than equated to inference failure. Stop on OOM/native failure
or memory anomaly; no automatic second case, parameter tuning or ceiling growth.

## Approved diagnostic execution (2026-10-06)

The single command above was approved and completed with exit 0. Artifact:
`E:/LTX-2/outputs/ddo_modular_video_20261006_131734_589/metrics/modular_run.json`.
Total elapsed: 532.6823 s; compute-stream phase: 494.1370 s. Four event
intervals: 204.8341, 89.8687, 106.2195 and 93.0882 s. This instrumented
four-step workload is not a matched benchmark against the earlier eight-step
run; its later intervals also remain slow.

Allocation scope complete, four CPU boundaries; native timeline complete,
1187 samples, zero dropped/failed/pending, event cleanup released. Original
encoder/transformer failure, denial, fallback, mismatch, quarantine and failed
observation/reclaim counters were zero. Tickets retired: encoder 7 to 0,
transformer 0 to 0. CPU-only inspection with CUDA initialization forbidden
confirmed finite packed video `[1, 22000, 128]` and audio `[1, 201, 128]`
float32 tensors. Selected diagnostic collection/safety criteria PASS, not a
performance improvement or image regression gate. The accepted DLL is not
replaced by this candidate.

Measured seven driver-call categories sum to 6.7117 host seconds, including
1.8117 create, 1.0149 unmap and 3.6678 release seconds. Pool reuse: 172595;
new backings: 1432; observed mapped high-water mark: 6457131008 bytes.
These call durations alone do not explain 494 seconds of denoise. They do
not exclude waits outside the wrappers or driver paging during computation.

Execution windows cover 493.2034 s; transformer-block window sums by step
are 203.9925, 89.5338, 105.8327 and 92.7434 s. Auxiliary-unit sums are
approximately 0.26-0.30 s per step. Slow windows vary across blocks; these
are dependency/host-gap-containing windows, not pure kernel time. Upload
window union: 21.4152 s, intersection with execution: 21.0858 s; dependency
window unions are approximately 0.003 s each. Do not add overlapping windows.

Two access calls returned non-success in the first step, with no final native
fault failure. The existing model-arena access retry calls `cuCtxSynchronize()`
on `CUDA_ERROR_NOT_READY`; this wait is outside the seven measured calls.
The report does not identify the return codes or establish whether these two
calls reached that retry. Therefore allocation-path waiting is not ruled out.
The next focused diagnostic should distinguish access return codes and time
the existing retry synchronization and enclosing fault path before proposing
any removal. Preserve synchronization/ownership and all ceilings until that
evidence exists. No new CUDA run or policy change was performed or authorized
by this completed diagnostic.

## Access-retry timing extension prepared (non-CUDA)

Separate candidate: `ddo-vmm-allocator/build-allocation-retry-diagnostics-20261006/Release/ddo_vmm.dll`.
SHA256: `335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.
Visual Studio 2022 x64, CUDA Toolkit 12.8.93; same diagnostic/span-pool ON
and dispatcher/test-hooks OFF settings. Both previous DLL hashes unchanged.

Optional native snapshot v2 preserves the v1 export/layout and adds existing
access-retry synchronization timing, enclosing fault-range timing and access
return categories (success/not-ready/out-of-memory/other). The Python adapter
prefers v2 but preserves v1 consumption with unavailable extra fields omitted.
DDO's public scope transports and prints these fields without host changes.
Fault timers include nested driver calls and waits; do not add overlapping
timers. The synchronization return and retry policy remain unchanged, including
the existing ignored return; diagnostics do not repair or remove this behavior.
Normal builds compile out observations. No memory or ownership policy changes.

Release DLL and standalone CPU contract built successfully. C++ contract
passed; 44 Python CPU contracts passed, including actual v1/v2 exports without
CUDA initialization, unavailable fields and public delta/format handling.
This is not a CUDA validation or promotion of the candidate.

### Proposed single follow-up, pending user confirmation

```powershell
Set-Location E:\LTX-2
.\run_ddo_vmm_video_modular.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-allocation-retry-diagnostics-20261006\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 8 --steps 4 --seed 43 --no-use-audio-condition --output-type latent --diagnose-denoise --ddo-vmm-resident-weight-budget-gb 2 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources
```

Hypothesis: access not-ready retry/context waiting or the enclosing fault
path accounts for a substantial slow-step interval. If it does not, report
the negative result rather than remove a safety wait or infer paging.
Criteria: four complete CPU/GPU boundaries with native schema 2, complete
timeline/no drops, finite packed outputs, original native failure/fallback/
mismatch/quarantine counters zero and tickets retired. Inspect access-return
categories and retry-sync failures separately. Stop on failure/memory anomaly;
no automatic repeat, parameter tuning, decode or ceiling increase.

## Approved access-retry diagnostic execution (2026-10-06)

Executed the single follow-up command above after user confirmation, exit 0.
Artifact: `E:/LTX-2/outputs/ddo_modular_video_20261006_144734_387/metrics/modular_run.json`.
Total elapsed 443.9171 s; compute-stream phase 406.3883 s; step intervals
244.9366/61.4196/49.0461/50.8503 s. Later-three mean 53.7720 s. This remains
diagnostic evidence, not a matched benchmark or a demonstrated speedup from
instrumentation. First-step time increased while later steps decreased relative
to the previous diagnostic; neither binary changes memory policy.

Native schema 2 collected all four boundaries. Timeline complete: 1187 samples,
zero drops/failures/pending, events released. Encoder/transformer original
failure, denial, fallback, mismatch, quarantine, failed observation/reclaim
and cache-pressure-entry counters were zero. Encoder tickets 7 to 0,
transformer 0 to 0. CPU-only artifact checks with CUDA initialization forbidden
confirmed finite float32 video `[1, 22000, 128]` and audio `[1, 201, 128]`.
Peak RSS 30.5757 GiB; GPU peak delta 6.9727 GiB over the recorded baseline.
Selected collection/safety criteria PASS; no image GPU regression claim.

The suspected model-arena access-retry synchronization was not invoked:
zero calls/seconds/failures. All 98 enclosing fault calls totaled 0.0981724 s.
One first-step access returned NOT_READY, with 2394 successful access returns
and zero out-of-memory/other categories. That non-success did not reach the
instrumented model-arena retry; the current report does not locate its call
site. Region creation has a separate access-failure cleanup/retry via the
span pool, but attributing this observation to that site remains an inference.

Seven measured driver categories totaled 7.3684 host seconds. Native execution
window union 405.3095 s; upload union 20.8419 s (20.4292 s overlapping execution).
Compute/upload and slot dependency unions approximately 0.0027/0.0019 s.
These windows are not hardware-engine activity or disjoint duration partitions.

Negative causal result: neither direct driver-call duration nor the measured
model-arena retry/fault path explains the long first step in this execution.
Keep its safety synchronization. The unresolved cost lies within broad execution
windows that include host launch gaps and GPU dependencies; separate forward
operators/host launch work from GPU kernel or paging time before implementing
a performance correction. No further CUDA run, tuning or policy change followed.
