# Implementation plan: persistent registered source in the VMM runtime

## Multiple native windows gate completed (2026-10-02)

Eight CUDA tests and the probe passed; identical image, zero failures, and an
actual peak of eight retained windows, zero at completion. Warm 1,423592 s/it,
Generate 24,8827 s: performance failed. The barrier between windows has already
been removed, but source waits still total 4,974880 s (not exclusive/additive
time). Do not revisit this difference as though implementation were missing;
it is insufficient for the target.
Milestone preserved, with no promotion/rerun/new ceilings. Before another patch,
identify the remaining cold critical path. Result/command below.

The previous gate passed correctness, but Generate 24,4543 s still failed.
The per-window barrier has now been removed in an isolated candidate: separate
submission and retirement, owners retained in at most eight flights, without
changing workers, memory, or the runner. The candidate text above records the
pre-execution checkpoint; compilation/CPU and CUDA have now passed, performance
has not. Command/hypothesis/criteria in
[native-source-flights-gate.md](native-source-flights-gate.md).
Milestone preserved, with no promotion or promised gains before measurement.

## Cold-start review after repeatability (2026-10-02)

Two approved repetitions passed: warm 1,412275 / 1,375524 s/it,
Generate 23,2125 / 23,2392 s, identical image and zero failures. Source still
empty before denoise; cumulative host readiness waits 5,719305 /
6,054061 s. Do not add this counter to registration/submission or attribute
everything to the first step without per-phase observations. The next correction
is designed in [native-cold-source-plan.md](native-cold-source-plan.md): separate
native asynchronous source preparation from DMA submission, with a bounded
queue, ownership, cancellation, and the demand path preserved. No runtime/CUDA/
ceiling patch in this review; reducing the first step by moving time into
prepare is not enough.

## New observed milestone: native pool 1,387751 s/it (2026-10-02)

Approved consolidated sequence completed: three small gates PASS (interior/
neighbor pointers, asynchronous cross-stream reuse, retirement without cache),
then one BF16/real-prompt/8-step inference with ceilings preserved. Identical
image, zero failures/quarantine/fallback/pressure/mismatch, tickets 736 -> 0,
complete timing, no monitor errors. Warm 1,387751 s/it versus protected
1,712871 (18,98% lower observed); Generate 24,9113 s, total 45,7932 s.
Prepare 0,8120 s, first step 7,481182 s. Still 4,34–5,13% above
1,33–1,32 s/it; one run does not establish repeatability or close the entire plan.

During denoise, pool evictions 53 -> 53 and host waits 0,1302295 s unchanged;
the exact-size candidate's churn did not recur. Unique backing 2.543.845.376 B
(4 mappings), final leases 457.065.984 B for 457.064.772 B requested (padding
1.212 B). Pending 2.086.775.808 B is unreaped completion metadata, not a measure
of GPU work still active. Partially occupied backing is not guaranteed
recoverable VRAM; do not compare leases alone with the former sum of per-request
mappings. Weight reads/DMA/backing match the controls; do not claim reduced DMA.

Protected DLL `4AC3D8B...` intact. New milestone saved at
`build-milestone-span-pool-8caad072/Release/ddo_vmm.dll`, with hash `8CAAD072...`
and a local source ZIP alongside, without changing the default or automatic
promotion. Approved gate exhausted, no reruns. Text encoder remains on group
offload: native source/arena support for its weights and profile not implemented
yet. Next work must preserve this milestone and distinguish prepare/cold step,
steady-state, and pressure/failure/repeatability evidence before new approval.

## Pool connected in isolated build (2026-10-02; CUDA pending)

`ddo_vmm_torch_pool.cpp` connects spans to real backing/events, validates context/
device/stream, publishes interior pointers, and retains ownership on failure.
Reuse enqueues dependencies; host waits occur when releasing an entire backing
without leases, outside the lock. Ceilings preserved, no runner flags/heuristics.
Build `build-span-pool-20261002`, experimental CMake selection/default OFF,
SHA256 `8CAAD072...`; protected `4AC3D8B...` intact.

Pool report v1 connected to the public API with unique physical accounting;
it does not reuse per-request footprint semantics. CPU ledger: 1.278 checks
PASS, CPU dependencies PASS, Python CPU 32 PASS on the candidate and 32 PASS
on the protected build. Interior-pointer/cross-stream-reuse probe prepared,
not executed. Failures do not automatically migrate the allocator after partial
submission; the entire previous route remains available in protected mode/build.
CUDA/fallback/performance gate still pending: compilation does not promote the
candidate. Details in the pool plan.

## Span ledger implemented (2026-10-02; CPU only)

`native/include/ddo_vmm_span_ledger.h`: best-fit/logical splitting, leases with
owner/generation, inherited fences with strong ownership, publication/rollback
before submission, retirement, safe coalescing, quarantine, and unique backing
accounting. Two-phase release blocks reservations while physical destruction is
pending; failure does not return partially destroyed backing to the pool.
C++20 CPU executable passed 1.270 checks, including repeated conservation;
existing CPU footprint and event contracts also passed.

Not connected to the allocator yet: it does not create/destroy CUDA mappings or
events, or prove interior pointers or speed. No DLL rebuilt/replaced,
no CUDA executed. Next integration requires real owners, safe submission,
versioned accounting, and pressure/cache guards. See detailed plan below.

## Structural correction contracts (2026-10-02)

Follow [native-allocation-pool-plan.md](native-allocation-pool-plan.md) for
splitting/suballocation: shared backing, span leases, dependencies inherited
during splitting, provably safe coalescing, quarantine on failure, and unique
physical accounting. The current `AllocationRecord` destroys an entire region;
returning an interior pointer without changing ownership would be unsafe.
Per-request v1 metrics cannot be reinterpreted as physical pool backing.
Design prepared, pool not implemented/connected yet; this document authorizes
no new CUDA execution.

## Exact-size cache gate: correct, but performance failed (2026-10-02)

Approved sequence completed once: cross-stream reuse probe PASS and full model
with ceilings preserved. Oversized excess zero before/after, identical image,
no failures/fallbacks/mismatch/pressure; tickets 32 -> 0.
Protected DLL intact, candidate `19AE9C2A...` not promoted.

Warm 1,950957 s/it (previous observer 1,756785; protected 1,712871),
Generate 29,7937 s, total 50,7639 s. Churn confirmed: evictions 384 -> 3.566,
eviction host synchronization time 0,147922 -> 12,857517 s. Eliminating larger
reuse without splitting backing causes allocation/eviction/waits; it is not the
final solution. Do not increase cache/budget to mask this. Preparation sections
below describe a historical candidate, not a recommendation for promotion.

Next direction: native splitting/suballocation that preserves reuse of larger
backing without handing the entire block to a small request. Ownership/fences
per span, coalescing only when safe, backing release only without consumers,
partial failure retaining ownership, and unique backing accounting without
double counting require contracts before another inference. Mapping granularity
and suballocation alignment must be distinct; no promise to recover all padding.
Not implemented/validated yet. CUDA approval exhausted; do not execute new
variants automatically.

## Oversized reuse correction prepared (2026-10-02; no new CUDA)

Native selection now requires equal mapped size and device. A larger block is
not transferred whole to a smaller request; it remains cached for its size class.
A miss uses the existing allocation/pressure/retry path; churn may increase.
No splitting/remapping regions, changing ceilings, or removing events.
Granularity padding was not corrected by this change.

Isolated build `build-exact-cache-20261002` (`19AE9C2A...`), three C++ CPU
contracts PASS and 28 Python CPU tests PASS. Protected DLL intact; candidate
includes the previous same-stream change and footprint observer, no promotion.
Next gate requires approval: reuse probe, then one inference with preserved
parameters; require zero reuse excess, correct image/ownership, no anomalies,
and evaluate timing/churn. No additional CUDA executed.

## Footprint gate completed (2026-10-02)

Approved sequence executed once: CUDA probe 16 MiB/cache 512 MiB with
cross-stream reuse PASS, followed by BF16/real-prompt/8-step model with
ceilings preserved. Identical image, no failures/fallbacks/pressure/mismatch;
tickets 1324 -> 0. Protected DLL intact; candidate not promoted.

Final snapshot: 457.064.772 B requested, 1.082.130.432 B minimum mapped,
1.598.029.824 B actually mapped, 308 records. Padding 625.065.660 B
(0,582138 GiB) and reuse excess 515.899.392 B (0,480469 GiB): total
1,062607 GiB, 71,4% of live mapping. Consistent decomposition before/after.
Not the peak during forward, or evidence of safely recoverable memory.

Warm 1,756785 s/it versus protected milestone 1,712871 (+2,56% observed);
Generate 27,4644 s, total 48,4887 s. Do not promote based on lower total;
the build also contains the previous same-stream change. Next correction must
evaluate oversized reuse and native suballocation of small requests, preserving
alignment, ownership/events, and protected paths. Do not increase budget
to mask excess. None of these corrections is validated by this gate.
Details/artifacts in the sidecar; approval exhausted, no automatic new runs.

## Footprint measurement prepared (2026-10-02; no CUDA)

Optional atomic export separates current request, minimum mapping, actual
mapping, granularity padding, and reuse excess. Public/JSON report integrated;
old DLL omits unavailable fields. Isolated build
`build-allocation-footprint-20261002`: DLL and two CPU contracts compiled,
both CPU executables passed, 13 Python CPU tests passed including the real
export. Protected DLL `4AC3D8B...` intact. The isolated build also includes the
previous unpromoted same-stream candidate: not a pure timing control.
With the protected DLL, 25 CPU telemetry/profile tests passed; absence of the
new export remains compatible, without inventing zeros.

Policy, cache selection, and ceilings unchanged; no CUDA execution performed.
Model measurement is still needed before deciding on a rounding/reuse correction.
Post-phase snapshots do not measure peak; accounting excess is not guaranteed
recoverable capacity. Protected 1,712871 s/it milestone preserved, no new gain
claimed.

## Accepted milestone and next investigation (2026-10-01)

### Full-model ledger gate completed

Only approved run, no profiler, explicit RSS, DLL/ceilings preserved:
warm 1,712871 s/it; Generate 27,3821 s; total 48,8927 s. Previous control:
1,718794 s/it / 27,7763 s / 49,4374 s. Identical image/hash; no failures,
fallback/pressure/mismatch, final tickets zero, no monitor errors. Four gauges
present in both snapshots. Validates integration and absence of observed
slowdown in this run, not causal gain or statistical non-regression.

Before/after: live 1.616.904.192 / 1.598.029.824 B (311 / 308 records), cache
39.845.888 / 222.298.112 B (13 / 29 records). Post-denoise: 1,488281 GiB live
and 0,207031 GiB cached, separate from weight backing 3.858.759.680 B.
Not peaks or total physical GPU memory; residual cache does not demonstrate
several recoverable GiB. Next code review: per-VMM-allocation rounding and
oversized reuse, distinguishing requested from mapped bytes before proposing
native compaction. Waste/recoverable peak not measured yet; do not increase
residency or remove fences based on these numbers. Sequence closed with no
rerun; details/artifacts in the sidecar. Any next CUDA requires a new approved
scope.

### PyTorch memory ledger connected to the public report

CPU export contract executed on the protected DLL: 10 tests passed
in 4,55 s, including four real gauges and release export capability,
without initializing CUDA. No rebuild or new inference.
The existing harness records before denoise and after readback/retirement;
these moments do not capture the intermediate peak. Next proposed gate is
one run without profiler, explicit RSS, and preserved parameters, to validate
integration and residual state, not determine a safe budget from the peak.
Requires command confirmation; no capacity increase included.

Python/documentation-only patch: the four existing C++ exports for live/cached
allocations/bytes are now consumed by the binding and reach
`DynamicOffloadMetrics`, without runner changes. Missing exports are omitted;
read errors are not converted into zero. Profile shows bytes in GiB with
explicit scope. Reads use independent locks, no CUDA queries/waits.

These are process PyTorch allocator gauges, not peak, available capacity,
or GPU total: they exclude weight arenas/slots, external workspaces, and records
that have left the ledger but still drain/retain backing. Do not add snapshots
from different times or subtract these values from a global peak. Catalog in
`docs/metrics.md`. Does not require rebuilding the protected DLL.

CPU: 38 passed, 1 optional real export skipped in 4,64 s. Initial attempts had
an inaccessible import path and tempdir; corrected only in the command
(PYTHONPATH src/submodule and a fresh basetemp in the workspace). No CUDA
executed or residency budget/algorithm changed. Linear and NVMe baselines were
not revalidated on GPU by this CPU gate.

Saved control: weight backing 3.858.759.680 B (arena 2.717.908.992 B +
slots 1.140.850.688 B), against the requested 4 GiB ceiling; cumulative H2D
187.588.136.576 B over eight steps. The average 23.448.517.072 B includes the
first step and does not replace per-step metrics. The next capacity decision
depends on distinguishing execution/cache/auxiliary memory; do not increase the
ceiling or assume equivalence between registered host memory and resident device
weights.

### Event completion capture: closed, invalid timing comparison

Only approved execution completed. Image/hash and bytes preserved; no failures,
fallback, pressure, or mismatch; final tickets zero and timeline released.
Protected DLL intact. All 55.066 events have completion timestamps;
the 130 copies following compute/upload dependencies in the tail start after
the fence (median 12,698435 ms). Does not prove the cause of normal behavior.

The proposed command omitted `--memory-monitor rss`; it used the full/USS
default, unlike the previous control. RAM queries accumulated 52,209703 s in
Generate (not exclusive/additive time). Capture also changes events and may
introduce artificial dependencies. Warm 3,391150 s/it / total 128,6426 s are
instrumented/confounded observations, not a runtime regression or comparable
gate. Do not attribute all slowdown to the profiler alone; do not repeat
automatically to correct the omission. Future proposals must explicitly specify
monitoring mode and obtain new approval before CUDA.

Offline analyzer extended, 18 CPU tests passed in 0,07 s. No scheduling patch
justified by this capture. Details and artifacts in the sidecar.

### Offline review after control/candidate

No new CUDA test, parameter, or DLL changed. In the saved control, host source
waits accumulate 6,843185 s, CUDA registration 0,478961 s, and batch submission
7,574608 s. These are cumulative, partially nested counters, not additive phases
or evidence that all waits belong to the first step.
Source reads 25.846.148.360 B once; per-step disk rereads do not explain current
steady-state. Prepare of 0,9104 s does not include all on-demand work.
Anticipating that work counts as optimization only if Generate improves too.

In the previously captured trace, the 10 s tail contains 5,089597 s of kernels
and 4,859931 s of H2D, but only 0,082586 s of actual intersection. Already
submitted host work does not prove both engines are simultaneously eligible.
Native wait windows do not substitute for that actual activity.

The current route has two native slots and one upload stream. Before considering
stream rotation, demonstrate a dependency blocking eligible uploads: relate each
transfer to its slot's last consumer, the reuse fence, and the event awaited by
the next consumer. Changing stream count alone is not a proven correction.
Preserve fences on partial failure and capacity ceilings; do not change the
execution unit or introduce CUDA Graphs without specific evidence. Any candidate
remains isolated and requires a new approved consolidated gate before
CUDA/benchmark.

### Slot dependencies: code and trace review completed

The native pool alternates slots. Acquire waits on the selected slot's fence;
release records last use in that slot and releases the lease only after success.
Publication failure retains ownership. In normal flow, block n upload depends
on compute n-2 that used the same slot, not compute n-1 in another slot.
Post-forward submission is host-side, with no global wait on this path.
This allows overlap, but does not prove the engines execute it.

The new `scripts/analyze_cuda_dependencies.py` opens SQLite read-only and
correlates events by process/device/context/eventSyncId, without massive joins.
In the approved 10 s tail: 18.638 compute/compute edges, 207 compute/upload,
and 687 upload/compute; no unknown identities in this interval. For the 207
compute/upload edges, the interval from the last recorded activity before fence
recording to the next copy has median 12,364094 ms, minimum 11,362514 ms, and
maximum 121,461979 ms; no negative intervals.
This predecessor is a lower bound on event completion, not its timestamp;
the interval is NOT fence wait time and does not prove absence of blocking.

Synthetic analyzer contracts and geometry regression: 14 CPU tests passed in
0,07 s. No CUDA test or runtime/DLL/capacity change.
No concrete slot dependency error found to justify removing fences or promoting
stream rotation as a solution. The distinction still open is actual eligibility
versus engine scheduling. A new capture, if needed, must explicitly identify
the missing event-completion evidence rather than repeat the same trace without
additional information.

**Full comparison completed, no observed gain:** control warm
1,718794 s/it / Generate 27,7763 s / total 49,4374 s; candidate 1,790891 s/it /
28,1763 s / 49,6903 s. Images/hash, bytes, and safety preserved. One sequential
pair does not separate variation from causal regression, but does not approve
promotion: validated DLL remains active and candidate isolated. Do not rebuild
candidate sources over `build/Release` while the performance gate remains open.
Do not execute another run automatically; prioritize actual overlap and cold
cost investigation, preserving ceilings. Full details in the sidecar.

**Corrected repetition approved and completed:** 47 CUDA contracts passed in
6,10 s with the isolated candidate. Active DLL preserved, candidate not promoted.
Closes the selected functional correctness gate; full-model performance and
non-regression remain open. Next proposed scope: control/candidate without
profiler, eight steps/real prompt and equal parameters, with safety inspection
between cases. New confirmation required before execution.

**CUDA gate interrupted:** 18 passed, 1 failure in 5,99 s. The candidate's three
direct contracts passed, but the fallback test applied a guard to the selected
DLL while the runtime prioritized the default DLL. Selection precedence fixed
and CPU contracts added: 51 passed in 4,99 s. Do not consider this mixed
execution complete candidate validation. Validated DLL intact; no benchmark
or new CUDA execution without confirmation.

**First candidate prepared, no CUDA execution:** cache reuse with real CUDA
identity (stream ID/device), conservative fallback on unknown identity, and
fence retained until new proof of completion.
Isolated Release `build-same-stream-20261001` compiled with CUDA 13.2; C++ CPU
contracts PASS and 41 Python CPU tests passed. Active DLL preserved.
Three small real-reuse CUDA contracts prepared and syntax-checked only;
no CUDA gain or safety validation performed yet.

The user considers the current level an acceptable advance over previous
runtimes near 3 s/it and wants continued optimization. Preserve this milestone's
code, DLL, and artifacts; 1,807403 s/it is an instrumented observation, not a
new official baseline. The previous unprofiled 1,763602 s/it result remains
the protected reference; do not compare these two classes as a proven regression.

Priorities: (1) safe simplification of redundant native cache waits, with real
stream identity and lifetime contracts; (2) explain and improve actual engine
overlap using existing data; (3) separate cold source/preparation cost from
steady-state. Moving work outside the sampler alone is not an end-to-end gain.
Reduced host API cost alone is not increased GPU throughput. Preserve budgets
and protected routes; do not pursue overcommit gains without a separate safety
gate.

Each candidate must be reversible and measure complete Generate time and GPU
steady-state, plus image, failures/fallbacks, memory, bytes, and closure.
CUDA and benchmarks require confirmation of consolidated scope; accepting this
milestone does not automatically authorize new runs.

## Dependency review: Driver capture completed on 2026-10-01

One approved Nsight capture executed, without changing DLL, presets, or ceilings.
Image/hash and observable safety contracts preserved; profiler details and
caveats in the sidecar. Instrumented warm 1.807403 s/it; not a new official
benchmark or proof of complete non-regression.

The export exposes Driver calls and confirms 18.638 compute → compute waits
in the 10 s tail, plus 207 compute → upload and 687 upload → compute.
Same-stream waits are a concrete target for native cache queue simplification,
but cost/causality cannot be inferred from count alone.
Do not compare only special/null handles to remove a dependency:
stream/context identity must be proven; retain the original fence for publication
failure, cross-stream waits, renewal on free, and safe cleanup.

Actual kernel/H2D overlap remains low (0,082586 s in the tail); host work was
already submitted in both flows. Warm native dependency windows total only
0,002009 s compute/upload and 0,000437 s upload/slot. Do not introduce per-step
synchronization, new budgets, or Python heuristics based on this.
Driver 13.4 was captured with Nsight 13.2 libraries and without event completion
timestamps; respect these limitations when interpreting causality.
The approved sequence ended. Any new CUDA execution requires confirmation.

## Next block: per-unit native chronology, without Kineto — design of 2026-10-01

**Functional gate completed on 2026-10-01:** corrected suite **43 passed in
5.96 s**, control and detailed case executed once each. Images identical to
the protected hash; budgets/backing/H2D preserved, no failures/fallback/pressure,
final tickets zero and no monitor errors. Timeline complete, 2444/4096 samples,
zero dropped/failures/pending, cleanup released, four windows and steps present.
Full table in `experiments/dynamic_offload_results.md`.

Control: GPU warm **1.819584 s/it**, phase **19.287336 s**, total **48.8310 s**.
Detailed: warm **1.813431 s/it**, phase **19.788992 s**, total **51.0807 s**.
Do not promote as optimization: overhead/variation still not separated and full
non-regression/target remain open. Recorder remains opt-in.

Steps 2–8: 12.572607 of 12.694026 s inside execution windows;
98.295% of the upload-window union intersects execution. Marked dependencies:
compute/upload 0.001710 s; upload/slot 0.000386 s. Release/eviction deltas during
sampler zero. This directs investigation inside forwards (host submission,
kernels, and contention), without confusing windows with actual engine activity.
It does not justify new speculative cache/slot adjustments. Next CUDA/benchmark
scope requires command/hypothesis/criteria and new confirmation; sequence below
is closed, do not repeat automatically.

### Causal cadence control completed — insufficient result

Approved gate: **44 passed in 6.17 s**, followed by one profiled run.
Image/hash, budgets/backing/read/DMA, and safety preserved; timeline 2444 samples
complete/released, zero losses/failures/pending; eight waits per contract.
Instrumented GPU warm 1.795861 s/it, phase 20.142836 s, total 50.3133 s;
details in the sidecar. These are not official benchmark results.

Host cudaLaunchKernel union fell from 6.763309 to 0.183407 s in the 10 s tail,
but actual kernel/HtoD overlap only rose from 0.100948 to 0.194569 s (historical
previous 2.733414 s). Waits shifted host waiting; they did not restore concurrent
execution or the target. Total increased 0.4744 s versus profiled control.
Do not promote waits or implement native throttling assuming a solution.
The hypothesis of per-step throttling as a sufficient correction is not supported
by this test.

Next work: review native upload/slots/compute graph and recorder effects, without
changing attention/cache/ceilings. Identify a concrete dependency before a patch
or new gate. No extra test/repetition authorized by this sequence.
Design and commands below are history of the closed gate.

DDO timer receives `diagnostic_step_wait=True` only through explicit opt-in;
runner passes `--ddo-diagnostic-step-wait`. Default false, no preset uses it.
Waits on the step compute event, never device-sync/scalar read/global upload wait.
Public report marks the diagnostic as benchmark-ineligible and measures
calls/host_seconds. DLL/residency/cache/ceilings unchanged. Focused CPU:
31 passed, 1 skipped in 5.75 s; host callback: 4 passed in 0.06 s.

Hypothesis: step-event-delimited cadence reduces long host launches and increases
actual engine overlap. A negative result also closes the hypothesis; do not turn
this control into an official solution. One profiled execution changes only this
option relative to `native_activity_trace_20261001`.

Proposed consolidated gate: selected CUDA contracts (including a new small
two-boundary/output test), followed by ONE profiled eight-step run, preserving
DLL, recorder, model, prompt, seed, monitor, and capacities. Stop at the first
failure/anomaly. Safety/output/bytes criteria match the previous gate; timeline
complete, no losses/failures/pending, cleanup released;
`diagnostic_step_wait.calls=8`, valid host_seconds, benchmark_eligible=false,
valid HtoD/kernel trace. Compare actual overlap, launch calls, and GPU
phase/warm, without adding host/GPU times or promising the target. Do not repeat
or change ceilings/parameters if the effect is negative. Only then discuss a
native solution.

Proposed commands, **not executed**:

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$ddoPython='E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src;E:\ProjetosIA\diffusers-dynamic-offloader\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& $ddoPython -m pytest tests/test_native_timeline_cuda.py tests/test_execution_timing_cuda.py tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-native-cadence-cuda-20261001
if ($LASTEXITCODE -ne 0) { throw 'Cadence CUDA contracts failed; stop' }
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:HF_HOME='F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& $ddoPython E:\ProjetosIA\diffusers-dynamic-offloader\scripts\trace_vmm_runner.py --trace-output outputs/ltx_image_modular_vmm/native_cadence_trace_20261001/cuda_trace.json E:\ProjetosIA\ltx2.3-image-diffusers\run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-gpu-timing --ddo-native-timeline --ddo-diagnostic-step-wait --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/native_cadence_trace_20261001/run
if ($LASTEXITCODE -ne 0) { throw 'Cadence activity diagnostic failed; stop' }
```

### Actual activity diagnostic completed — current DLL

Approved run executed once, exit 0, safety/output criteria preserved.
Timeline complete/released, no losses, failures, or pending entries;
instrumented GPU warm 1.853196 s/it, phase 20.492174 s, total 49.8389 s:
**diagnostic, not benchmark**. Detailed result in the experiment sidecar.

In the 10 s tail: kernels 5.230698 s, HtoD 4.727686 s, actual intersection
0.100948 s (previous trace 2.733414 s). Overlapping native windows did not
represent simultaneous engines. Host cudaLaunchKernel union 6.763309 s,
maximum 42.341 ms; previous 0.179496 s/maximum 0.1213 ms. Previous trace
shows eight synchronizations associated with `aten::_local_scalar_dense`;
current trace shows no explicit synchronizations in that tail. Current callback
does not read a CUDA scalar per step. This bounds a cadence/backpressure
hypothesis; it neither isolates DLL/recorder/profiler effects nor proves cause.
Do not restore scalar reads, global sync, or change attention/cache/ceilings as
an assumed correction.

Next causal control: vary one submission boundary, explicitly diagnostic,
preserving DLL/recorder/workload; only then decide any native policy bounded by
events/ownership, without an official Python heuristic. Requires new design/
command/criteria and approval before CUDA. Offline analyzer extended with a
clipped host API and without double counting nested calls:
**6 CPU tests passed in 0.04 s**. No second GPU test executed.

Design and command below are history of the completed run; do not repeat:

Offline review of the previous trace (not benchmark) with
`scripts/analyze_cuda_trace.py --upload-stream 17 --last-seconds 10`:
kernel union 6.416869 s, HtoD union 6.127331 s, intersection 2.733414 s;
HtoD without a recorded kernel 3.393918 s; gaps with neither activity 0.189213 s.
Kernels associated with addmm total 4.956772 s, attention 0.521386 s. Generic
CPU analyzer: **5 passed in 0.04 s**; no new CUDA test executed.

**Interpretation correction:** forward/upload window overlap is not equivalent
to engine overlap. Small explicit waits do not exclude HtoD from the critical
path or contention. An old trace does not automatically describe the current
DLL. Do not change attention, cache, slots, or budget based on this review.

Hypothesis: a single trace of the current DLL identifies how much of the warm
interval is GEMMs, copies, or gaps, and compares window geometry with actual
activity before correction. Kineto can change execution: timings will be
diagnostic only, never a benchmark/performance promise.

Proposal: **one execution**, same capacities/workload/monitor as the approved
pair, public GPU+timeline observation and existing profiler wrapper.
No shapes/stack/per-memory profiling, new heuristic, or ceiling adjustment.
Criteria: exit 0, protected PNG, eight steps, no failures/fallback/pressure,
final tickets zero, no monitor errors, timeline complete/released without losses,
bytes/backing/ceilings preserved, and trace with CUDA kernel/HtoD activity.
Stop at the first failure/anomaly; do not repeat or expand automatically.
Select upload stream by expected payload, do not assume ID 17 in the new trace.
Offline analysis compares intersections/unions and operators; absence of recorded
activity is not causal proof of CPU starvation.

Proposed command, **wait for confirmation before execution**:

```powershell
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src;E:\ProjetosIA\diffusers-dynamic-offloader\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
$env:HF_HOME='F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe E:\ProjetosIA\diffusers-dynamic-offloader\scripts\trace_vmm_runner.py --trace-output outputs/ltx_image_modular_vmm/native_activity_trace_20261001/cuda_trace.json E:\ProjetosIA\ltx2.3-image-diffusers\run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-gpu-timing --ddo-native-timeline --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/native_activity_trace_20261001/run
if ($LASTEXITCODE -ne 0) { throw 'Activity diagnostic failed; stop' }
```

### History of gate preparation and corrections

**Implementation prepared:** native recorder v1 and four-window markers,
native per-step markers, DDO API `execution_timer(native_units=True)`, and host
flag `--ddo-native-timeline`. All disabled by default. Bounded events,
generation tokens, reads without synchronize, pending/truncated/failed status,
window geometry reduction without double counting. Timing separate from lifetime;
no residency/prefetch/slot/ceiling change. One session/owner per context;
concurrent inference not validated. Details in the metrics catalog.

Release build PASS; C++ CPU PASS; DDO **79 passed in 4.57 s**, host
**5 passed in 4.68 s**. DLL SHA256
`4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.
Previous DLL preserved at `build/Release/ddo_vmm.pre-native-timeline-20261001.dll`.
Approved gate executed until first failure: probe PASS; first CUDA contract
failed by assuming stable pending state between queries (**1 failed in
4.70 s**). Sequence stopped; no benchmark executed. Python closure corrected
to preserve the report before releasing events, and test corrected to use a
scope not yet closed, without artificial sleep. CPU timeline:
**13 passed in 4.43 s**. DLL unchanged; CUDA corrections not validated yet.
No new performance measurement; target and non-regression remain open.
Design below is the patch specification, not a CUDA gate approved by compilation.

### Approved and completed sequence — command history

Last approved reexecution: **2 passed, 1 failed in 5.02 s**. Stream/lifetime
and capacity/token contracts passed; integration stopped because the fixture
used legacy cache slots, without model-windows/batches.
Fixture corrected to the geometry already used in the execution-buffer gate:
eight 512x512 linear units, GPU 6 MiB, staging 2 MiB, source 64 MiB;
explicit batch/upload assertions before windows. Syntax PASS.
At that moment CUDA had not been rerun. No DLL/runtime/benchmark change.
Reexecution and benchmarks subsequently completed according to the result
above; retain these failures as gate history.

One consolidated sequence, without Kineto: existing lifetime/reuse probe,
existing contracts plus three new ones, ONE control run and ONE detailed run.
Do not repeat or expand parameters. Stop at the first contract failure, image
mismatch, monitor error, memory anomaly, or incomplete/truncated/failed
diagnostic; never increase capacity automatically.

Hypothesis: native windows and waits locate starvation from uploads or slot
recycling in steps 2–8, without confusing a window with pure DMA.
Control measures the default route in this DLL and bounds recorder overhead.
Criteria: correct data/lifetime/cross-reference/tokens/capacity in contracts;
protected PNG and eight complete steps, zero failures/fallback/pressure,
final tickets zero, ceilings/H2D/backing preserved, no monitor errors.
Detailed case: timeline complete, dropped=0, failures=0, pending=0,
cleanup released, four window types and step markers present, valid offsets/order.
Compare GPU/Generate/total/VRAM/RSS with control and previous milestones; record
overhead or regression, without promotion. Observation does not promise ~1,3 s/it.

Exact proposed commands (execution will be supervised between gates).
The probe already passed with the same DLL: on reexecution, start with pytest
below, without repeating the probe. Both benchmarks remain conditional on gates:

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$ddoPython='E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$ddoLibrary='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src;E:\ProjetosIA\diffusers-dynamic-offloader\src'
$env:DDO_VMM_LIBRARY_PATH=$ddoLibrary
& $ddoPython ddo-vmm-allocator/scripts/vmm_torch_allocator_probe.py --library $ddoLibrary --megabytes 16 --cache-mib 128 --verify-stream-lifetime --verify-nonblocking-reuse
if ($LASTEXITCODE -ne 0) { throw 'Probe failed; stop' }
& $ddoPython -m pytest tests/test_native_timeline_cuda.py tests/test_execution_timing_cuda.py tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-native-timeline-cuda-model-source-20261001
if ($LASTEXITCODE -ne 0) { throw 'CUDA contract failed; stop' }
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:HF_HOME='F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& $ddoPython run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-gpu-timing --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/native_timeline_control_20261001
if ($LASTEXITCODE -ne 0) { throw 'Control failed; stop' }
# Only after inspecting control safety/output/metrics:
& $ddoPython run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-gpu-timing --ddo-native-timeline --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/native_timeline_candidate_20261001
if ($LASTEXITCODE -ne 0) { throw 'Detailed observation failed; stop' }
```

Scope: distinguish upload/compute dependencies on the experimental route;
do not change residency, ceilings, slot count, prefetch, or sampler. Do not
execute new CUDA tests/benchmarks using approval for the already closed sequence.

### Code review findings

- `native/src/ddo_vmm_source_cache.cpp`: tickets and batch completion use
  `CU_EVENT_DISABLE_TIMING`. The `dma_submit_seconds` and `dma_event_seconds`
  counters measure CPU/API time, not transfer execution time.
- `ddo_vmm_source_batch_submit` interleaves source requests and flushes of up
  to 64 ranges. A batch start/end window can contain source waits and
  submission gaps; do not call it pure DMA, especially in the first step.
- `ddo_vmm_source_batch_wait_on_stream` places the dependency on compute.
  `ddo_vmm.cpp` also has an upload dependency on the execution buffer's last
  consumer (`compute_fence`). Measuring only the first direction would omit
  potential blocking from slot recycling.
- `model_arena_adapter.py::_register_execution_unit` delimits generic
  execution units. Its pre-hook handles views/dependencies; the post-hook
  closes ownership. These are the execution-window marking points, without
  detecting LTX/UNet classes or moving scheduling into the runner.
- The previous diagnostic already excluded **temporary cache release/eviction**
  as an observed wait in that denoise. This does not exclude GPU slot
  dependencies or establish continuous engine utilization.

### Minimal instrumentation and boundaries

One optional native recorder per session/context, with bounded memory/events
and declared capacity, independent of safety events. Disabled by default;
the normal route gains no events or CUDA queries. Initialize observation before
the measured interval, never allocate events per tensor.
Do not convert existing lifetime events into temporary timing events:
they can be rerecorded and have specific ownership.

| Window | Marking point | Permitted interpretation |
| --- | --- | --- |
| `upload_batch_window` | Submission start/end on the upload stream | Batch service window; includes gaps between copies and host preparation that leaves the stream idle. Not pure DMA. |
| `compute_upload_dependency` | Immediately before/after the batch wait on compute | Interval in which this dependency blocks progress; may include host launch gaps. Not CPU wait cost. |
| `upload_slot_dependency` | Before/after the wait on the slot's last consumer, on upload | Buffer recycling blocked on the upload stream; do not confuse with eviction. |
| `execution_unit_window` | After dependencies/views are installed and before forward; end before post-hook cleanup/unpin | Unit execution window, including launch gaps; not a pure sum of kernels. |

Separate versioned native API: begin/mark/end/snapshot/close with explicit tokens
and context. Submodule bindings/adapter only mark units/dependencies; C++ owns
events, samples, and lifecycle.
DDO exposes the session through the public timing/report API, in explicitly
opt-in detailed mode. Runner only enters the scope and saves the report.
Do not create a parallel residency path or per-window host commands.

Recorder uses timing-enabled events and a common CUDA reference to relate
streams on the same device/context. Do not mix host and CUDA clocks or introduce
cross-stream waits/barriers to align clocks.
Preserve signed offsets if the reference executes after a sample;
do not clamp negative timestamps to zero. The small CUDA contract must validate
this cross-reference before full-model use.

Snapshot only after existing readback: first query event completion with
`cuEventQuery`, then calculate times. Do not use elapsed-time as proof of
completion. No per-unit synchronize, polling during
denoise or recycling of an event still referenced. NVIDIA documentation warns
that stream event intervals can include other operations:
https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html

Each sample carries session, sequence, unit, stream, window type, bytes when
applicable, CUDA start/end, and state. Bounded pool, no automatic growth:
saturation records `truncated`/dropped samples, never a fictitious complete
summary. Creation/record/query failure marks the diagnostic incomplete and
preserves required ownership; it cannot release weights/slots, remove a
dependency, or fabricate a zero timestamp. Diagnostic cleanup must not introduce
new global or per-sample synchronization.

### Analysis and result limitations

Separate the first step from steps 2–8 using existing public markers.
Relate upload/execution windows and dependencies by unit/sequence.
Report **window** union/intersection, explicitly not overlap of active kernels
and DMA. Adding nested waits without deduplication causes double counting.
Only if these windows do not separate the cause should individual DMA or GPU
activity be instrumented: do not add tens of thousands of events initially.
This observation measures dependency effects and guides the next correction,
but does not replace engine, clock, or driver comparison profiling.

### Closure before another full model

1. CPU: tokens/ownership, capacity/saturation, disabled without CUDA, error/
   pending, schema, signed reference/offsets, interval reduction without double
   counting.
2. Compile preserving the current DLL. Small CUDA, **with separate confirmation**:
   distinct streams, known upload, delayed consumer, both dependency directions,
   slot reuse, correct output, and events not rerecorded early.
3. One user-approved consolidated comparison: opt-in detailed measurement and
   control without it, same workload/ceilings; explicitly report observer overhead.
   Do not call an instrumented run a definitive benchmark. Commands, hypothesis,
   and criteria will be presented after implementation/build/CPU.

Recorder and public option implemented and compiled; **no CUDA/benchmark with
the recorder executed yet**. Summary and CPU validation at section start.

## Logging without CUDA reads and eviction observation — 2026-10-01

**Update: sequence below approved and executed once.** Probe
16 MiB/cache 128 MiB PASS, lifetime/nonblocking reuse/renewed fence verified;
CUDA contracts **40 passed in 5.94 s**. Real eight-step benchmark without
profiler exit 0, same image, zero failures/fallback/pressure, no monitor errors,
tickets **1312 → 0**, ceilings/H2D/backing preserved.
GPU complete: first 7,617333 s, steady **1,763602 s/it**, phase
**19,962551 s**. Host denoise 18,3214 s now excludes pending GPU work,
not the complete sampler. Generate **28,4103 s**, total **50,4205 s**:
regressions of 0,1141/0,5338 s versus last run (28,2962/49,8867 s).
GPU gain of 0,112021 s does not close full non-regression or ~1,3 s/it target;
no automatic promotion/repetition, no claim of statistical causality.

All pre-denoise/post-readback release deltas are **zero**: reset sync calls and
time, failures, eviction attempts/bytes/time. Process totals 0,223327 s reset /
0,175393 s eviction occurred before sampler. Not an observed bottleneck during
denoise in this run. Source events created/reused 1344/9824, versus 7558/3610,
did not yield proportional gains. Next diagnostic must distinguish GPU
H2D/compute service and overlap without profiler disturbance; do not persist
with the measured temporary eviction hypothesis. No additional run approved
in this sequence. Full results and limitations in the sidecar.
Preparation/proposal below is historical and does not authorize repetition.

Restricted patch completed: display timesteps resolved on CPU before denoise;
event markers, scheduler, sampler, and monitors preserved.
Public pre-denoise snapshot saved in harness with `--ddo-gpu-timing`, without
interpreting internal names. C++ measures only existing completion
synchronizations and eviction attempts. Policy, ceiling, and barriers unchanged.
Cumulative metrics documented in `docs/metrics.md`; use pre/post delta.

Release build passed; C++ CPU contract PASS; DDO **64 passed in 4.83 s**;
host **4 passed in 4.57 s** in final validation. New DLL SHA256
`79EF18CBE27DE535C950658D36AF369E86ECF204511223A7CABC7F6DB3888F16`.
Previous DLL preserved at `build/Release/ddo_vmm.pre-release-observation-20261001.dll`.
No CUDA test or benchmark with this patch executed.

### Approved and executed sequence — historical commands, do not repeat

One consolidated sequence: small asynchronous reuse probe, existing CUDA
contracts, and one run without profiler, with parameters preserved.
Do not repeat, run a trace automatically, or increase ceilings.

Hypothesis: removing display CUDA reads avoids stream draining by logging;
new counters quantify the contribution of native releases.
Do not assume this change explains the profiler's 2,732176 s difference.
Criteria: probe with correct data/dependencies and nonblocking reuse,
no contract failures; benchmark with protected PNG hash, eight complete GPU
intervals, final tickets zero, zero completion/transport/fault/fallback failures,
ceilings/bytes preserved and no monitor errors. Stop at the first failure or
memory anomaly. Compare steady 1,796179 s/it, denoise 20,0747 s,
Generate 28,1420/28,2962 s, and total 49,1456/49,8867 s; any regression is
recorded, without automatic promotion or repetition. Do not claim ~1,3 s/it or
causality without measurement. CPU display resolution and pre-denoise snapshot
remain included in Generate/total, although outside the GPU sampler interval.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$ddoPython='E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$ddoLibrary='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src;E:\ProjetosIA\diffusers-dynamic-offloader\src'
$env:DDO_VMM_LIBRARY_PATH=$ddoLibrary
& $ddoPython ddo-vmm-allocator/scripts/vmm_torch_allocator_probe.py --library $ddoLibrary --megabytes 16 --cache-mib 128 --verify-stream-lifetime --verify-nonblocking-reuse
if ($LASTEXITCODE -ne 0) { throw 'Allocator probe failed; stop' }
& $ddoPython -m pytest tests/test_execution_timing_cuda.py tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-release-observation-cuda-20261001
if ($LASTEXITCODE -ne 0) { throw 'CUDA contract failed; stop' }
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:HF_HOME='F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& $ddoPython run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-gpu-timing --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/release_observation_candidate_20261001
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed; stop' }
```

## Completion/retirement and GPU measurement — preparation of 2026-10-01

**Update: approved sequence executed.** CUDA **40 passed in 6.38s**;
eight-step trace exit 0, same image, ceilings/H2D/backing preserved,
zero failures/fallback/pressure. Collection after readback: **1216 → 0 tickets**,
without unregister/reread. Complete CUDA markers: eight steps, first
7,499001 s, steps 2–8 **1,406142 s/it**, phase 17,342396 s. Profiler active:
these times are not the final benchmark. Actual GPU trace confirms **2,733414 s
H2D/compute overlap** in the final 10 s window, versus zero in the previous trace.
Baseline-subtracted GPU peak 6,6859 GiB, RSS 29,7626 GiB; no monitor errors.
Backlog observed in this run recycled after completion, did not persist as batch
ownership; still not a full-model post-close ledger. Detailed results/limitations
in sidecar. Preparation status below is historical.

### Next confirmation: one run without profiler

**Executed after confirmation:** exit 0, same image, zero failures/fallback/
pressure, tickets **1312 → 0**. Complete CUDA timing: first 7,500963 s,
steps 2–8 **1,796179 s/it**, phase 20,074572 s; host denoise 20,0747 s.
Profiled 1,406142 s/it was not reproduced without profiler; do not promote
that number. Baseline-subtracted GPU peak 6,6675 GiB, RSS 29,5667 GiB,
same ceilings/H2D/backing and no monitor errors. Versus previous candidate,
host denoise 20,1530 → 20,0747 s, but Generate 28,1420 → 28,2962 s and
total 49,1456 → 49,8867 s; text phase explains 0,5388 s of total difference
0,7411 s. No statistical/causal proof from one run, but full non-regression
criterion not closed: do not promote defaults, repeat automatically, or increase
budget. ~1,3 s/it target not reached yet.
Investigate profiled/unprofiled difference and a demonstrable mechanism using
existing evidence before another change. Proposal below is historical, now
executed; details and artifacts in sidecar.

Do not change implementation/ceilings before closing unprofiled measurement.
Same parameters, with already validated `--ddo-gpu-timing` and post-readback
collection. Do not repeat contracts or runs automatically.
Environment/PYTHONPATH/DLL/HF_HOME match commands below, cwd in sibling host:

```powershell
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-gpu-timing --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/completion_timing_candidate_20261001
```

Hypothesis: the trace-validated asynchronous mechanism preserves performance/
correctness without profiler. Criteria: identical output, CUDA timing complete
with eight steps, tickets_after=0, ceilings/H2D/backing preserved, no monitor
errors and no failures/fallback/pressure. Record GPU and host times separately;
do not use host callbacks as s/it. Compare phase/total with unprofiled candidate
(host denoise 20,1530 s / Generate 28,1420 s / total 49,1456 s), recording new
observation overhead and peak (6,7392 GiB baseline-subtracted) without increasing
budgets. If slower/failing, stop and do not promote. One result does not
guarantee repeatability; 1,3 s/it can only be claimed if actually measured.
Still requires separate confirmation; no unprofiled run executed here.

C++ review: each submission collects ready tickets from ALL batches, but snapshot
is read-only and there is no collection after the last submission.
`reap_batch(false)` queries individual events and preserves pending DMA.
Close requires completion; pool keeps up to 256 idle events, not a limit on
in-flight events. `events_created` accumulates creations; 7616 does not mean
7616 events simultaneously live. No early release identified in this review;
a snapshot of 1312 tickets, without final collection, does not prove a leak.

Optional export `ddo_vmm_source_cache_collect_completed` implemented: uses the
same reaper and locking, without waiting on pending work or taking external
tickets. DDO offers `metrics_snapshot(collect_completed=True)`, with before/
after counts in the report. No per-window Python scheduler introduced;
collection is one explicit call at phase end. Ceilings, idle pool, registrations,
and residency policy unchanged. This measures/closes an observation gap;
it does not promise faster denoise or fewer events created in flight.

DDO `result.execution_timer(enabled=True)` marks start/steps/end on the consumer
stream. Snapshot resolves only completed events; default does not wait;
no per-step synchronization. Intervals include dependencies and launch gaps,
not a sum of kernels. Tail after last callback is separate.
Host only marks callbacks and consumes the report, without native stats names.
Sibling harness enables both with `--ddo-gpu-timing`, disabled by default,
only in public native consumption. Collection follows existing readback.

Release compiled; C++ CPU contracts PASS; CPU suite **94 passed in 5.10s**.
New DLL SHA256 `32531DBE284E8FC62F522130104AD2AD06EE7AD949D8601232E3DA3E03FFD917`.
Previous DLL preserved: `build/Release/ddo_vmm.pre-completion-collection-20261001.dll`.
Syntax checks passed. No CUDA/benchmark/trace with this patch executed.

### Next proposed sequence — requires confirmation

To avoid fragmentation: existing CUDA contracts + two new ones (timing and
collection), then ONE diagnostic eight-step run with trace and timing.
It checks completion/retirement and actual overlap together; profiled times do
not replace unprofiled benchmark or promote the patch. Do not automatically
execute another run. Before the model, confirm no contract failed.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src;E:\ProjetosIA\diffusers-dynamic-offloader\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& .venv-vmm\Scripts\python.exe -m pytest tests/test_execution_timing_cuda.py tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-completion-timing-cuda-20261001
if ($LASTEXITCODE -ne 0) { throw 'CUDA contract failed; stop' }
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:HF_HOME='F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe E:\ProjetosIA\diffusers-dynamic-offloader\scripts\trace_vmm_runner.py --trace-output outputs/ltx_image_modular_vmm/completion_timing_trace_20261001/cuda_trace.json run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-gpu-timing --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/completion_timing_trace_20261001/run
```

Hypothesis: increased tickets in snapshot result at least partly from CPU
advancing asynchronously; collection after completion reduces batch tickets to
zero, without discarding source. Markers measure per-step GPU times, while trace
checks whether the allocator correction produced actual overlap.
Criteria: all contracts pass, including protected pending work and preserved
external owner; complete timing phase with eight steps; tickets_after=0 after
readback, same H2D/source/backing bytes, zero rereads/fallback/new failures, same
PNG SHA256 `877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`,
no monitor errors and ceilings maintained. Record peak and compare with
6,7392 GiB baseline-subtracted; do not increase budgets if it rises or pressure
appears. Require actual GPU activity and quantify H2D/compute intersection.
Zero overlap refutes the performance hypothesis, not permission to remove
fences. Stop at first error/anomaly, no full model after a failed gate.
Still no promise of 1,3 s/it or full-model post-close ledger validation.

## Asynchronous temporary reuse — patch prepared (2026-10-01)

**Approved sequence executed:** three-stream CUDA probe PASS; CUDA contracts
**38 passed in 5.64s**; eight-step benchmark exit 0, same image.
Host denoise call 21,6363 → 20,1530 s; Generate phase with readback
29,1332 → 28,1420 s; total 50,0931 → 49,1456 s. No new failures/fallback,
same ceilings/H2D/backing. However GPU peak rose 6,4832 → 6,7392 GiB
(baseline-subtracted), snapshot tickets 32 → 1312 and events created
32 → 7616. Do not promote: investigate native retirement/completion and measure
GPU time correctly. perf_counter callbacks without fences are not per-step GPU
duration; 0,1887 s in step 2 exposes this. Do not claim target steady-state,
add per-step synchronize, or increase ceiling to hide the problem.
Pre-close snapshot does not prove a leak; full model still lacks a post-close
ledger snapshot. Small gates validate their own closure.
No trace/second run executed. Details in sidecar; proposal and preparation status
below are historical, now executed.

Implemented in C++ allocator: malloc uses the consumer stream for
`cuStreamWaitEvent`, without host wait on normal reuse. Previous event remains
in the record even after dependency enqueue: if publication fails, disposal
still needs to prove completion. Free records a new event covering the consumer
before retiring the previous one. Eviction and closure still drain events before
unmap. If CUDA cannot prove completion, the region is not cached/reused or
destroyed: backing is retained until process end and error reported on stderr.
This exceptional containment can retain memory; not a performance path.

Release compiled; C++ contracts with CPU callbacks passed; Python CPU tests
**50 passed in 5.61s**. No CUDA/benchmark with the new DLL.
SHA256: `FBABF22A4E3581E410E653B437F477FD728AA441188749624C8CC23E326AB3ED`.
Previous DLL preserved as `build/Release/ddo_vmm.pre-stream-reuse-20261001.dll`.
No ceiling, default, linear/NVMe runtime, or Python scheduler changed.

### Proposed sequence — awaiting confirmation

Hypothesis: stream dependency lets host enqueue compute and look-ahead without
waiting for temporaries, maintaining correct order and content.
First prove asynchronous reuse across three streams, event renewal, and payload;
then existing CUDA contracts; only then an unprofiled benchmark with approved
candidate parameters. Stop at first failure, completion error, new fallback,
or memory anomaly; no automatic repetition.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src;E:\ProjetosIA\diffusers-dynamic-offloader\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& .venv-vmm\Scripts\python.exe ddo-vmm-allocator\scripts\vmm_torch_allocator_probe.py --library $env:DDO_VMM_LIBRARY_PATH --megabytes 16 --cache-mib 128 --verify-stream-lifetime --verify-nonblocking-reuse
if ($LASTEXITCODE -ne 0) { throw 'Allocator gate failed' }
& .venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-stream-reuse-cuda-20261001
if ($LASTEXITCODE -ne 0) { throw 'CUDA contracts failed' }
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:HF_HOME='F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/stream_reuse_candidate_20261001
```

The 128 MiB cache is exclusive to the small probe; benchmark bootstrap retains
the same 512 MiB. Require probe PASS, all selected contracts passing, clean
test ledgers, and no error containment.
Benchmark: same PNG SHA256 `877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`,
ceilings maintained, zero new failures/fallback and no monitor errors.
Reference: denoise 21,6363 s / steady-state 1,922283 s/it; if slower, do not
promote patch or adjust budget to mask the result. One run does not establish
repeatability. Benchmark can show gains, but only a GPU trace proves overlap;
no additional trace included in this sequence.

## Overlap — trace review and preparation (2026-10-01)

**Approved trace executed:** exit 0, same image, H2D, and backing, zero
failures/fallback/pressure. Captured actual GPU activity: transformer copies
on stream 17, kernels on stream 7. In the final 10 s window: H2D 4,387028 s,
kernels 4,314711 s and **zero overlap**. Global allocator code reveals a normal
barrier outside the weight pool: malloc ignores stream and synchronizes each
reused cached allocation's event on the host. The previous observation below
was limited to pool/transport; it does not exclude this temporary barrier.
Next patch must enqueue dependency on the consumer stream, preserving
event/backing and safe failure. Do not remove fences or adjust ceilings.
Not implemented yet; post-patch CUDA/benchmark require new confirmation.
Details, attribution limitations, and artifacts in sidecar. Profiled times do
not replace the unprofiled candidate; no defaults promoted.

After registration A/B, inspection did not demonstrate improper host
serialization on the normal route: pool acquire uses cuStreamWaitEvent on
the slot's last consumer; batch records completion on upload; consumer waits
on that event on compute stream; next-slot look-ahead follows current compute
enqueue. Registered source uses cuMemcpyHtoDAsync. Observed host waits cover
registration recycling, cold I/O, pressure/error, and closure. Do not remove
these safeguards assuming every synchronize is a bottleneck.
Approved look-ahead counters do not prove actual device overlap.

`scripts/trace_vmm_runner.py` prepared, a generic diagnostic wrapper without
model dependencies or runner/runtime/DLL changes. It relaunches through DDO
bootstrap before importing/starting profiler; uses CPU/CUDA Kineto without
shapes, stack, or allocation profiling; preserves arguments and refuses trace
overwrite. Not for linear/NVMe routes. CPU validates bootstrap, dispatch,
state restoration, and refusal without support. First CPU attempt encountered
TEMP permissions; repetition with local basetemp passed.
Final CPU gate: **47 passed in 4.57s**. No CUDA execution in this preparation.

Proposed execution, still **not authorized/executed**, in sibling host,
preserving PYTHONPATH/DLL and approved candidate configuration:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe E:\ProjetosIA\diffusers-dynamic-offloader\scripts\trace_vmm_runner.py --trace-output outputs/ltx_image_modular_vmm/registration_trace_20261001/cuda_trace.json run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/registration_trace_20261001/run
```

Hypothesis: trace reveals H2D/compute intersection, bubbles, and streams causing
residual cost, without changing dependencies. Require actual GPU Memcpy HtoD
and kernel activity (CPU-only events do not validate the diagnostic), equivalent
output, respected ceilings, and no new failure/pressure/fallback.
Profiler may cause overhead and collection synchronization: times from this
run are not benchmark, do not replace 1,922283 s/it, or promote defaults.
If CUPTI/Kineto supplies no GPU activity, stop/report limitation; do not install
tooling or repeat automatically. Correct only the trace-indicated mechanism,
preserving last-consumer and persistent source.

## Full persistent registration — preparation of 2026-10-01

**Update after confirmation:** sequence executed. CUDA **38 passed in
5.80s**; capacity **1 passed in 14.36s**, no skips, errors, or anomalies.
Full source: 394 registrations, zero unregisters/waits before trim,
second pass without reread and ledgers zeroed on closure.
Eight-step A/B: control 28,9837 s denoise / 2,997579 s steps 2–8;
candidate 21,6363 s / 1,922283 s. Same PNG, H2D, and backing; candidate
388 registrations / zero unregisters, control 2805 / 2797. Setup 0,8961 →
0,9582 s and first step 7,9970 → 8,1660 s: per-scope regression caveats.
Observed reduction 25,35% denoise / 35,87% steady-state in one A/B run;
not statistical proof, automatic promotion, or linear/NVMe validation.
Do not expand GPU or execute another case without approval. Next block:
check dependencies/overlap of existing pipeline, without a new Python buffer.
Full results and artifacts in sidecar. Preparation text below is historical;
CUDA/capacity/A/B are no longer pending in this sequence.

Priority: validate registration coverage using existing config v5, without
another cache, DLL change, GPU budget, or Python policy.
Retention and registration remain independent. Proposal does not increase
defaults: the explicitly approved candidate may register 24,625 GiB of source
instead of control's 512 MiB. Staging remains 1,5 GiB aggregate and GPU ceiling
4 GiB. Host reservation/registration is not GPU backing.

Two contracts prepared: small full payload over three passes, without rereads
or churn after warmup; real source over two passes, 394 segments of 64 MiB,
without loading model. Real gate compares only 17 bytes per segment: capacity/
lifetime evidence, not full throughput. Trim preserves content, clear/close
return ledgers. Previous 512 MiB recycling contract remains separate and
unchanged in capacity. Full registration requires two opt-in flags and explicit
exact ceiling; shard geometry changes fail before committing/registering source.

CPU validation: **75 passed in 7.64s**, configuration, backend, profiles, presets.
Gate syntax checked. CUDA/capacity/benchmark **not executed**.
DLL preserved: `F189D035D10089AE77DD9A7F21EED2FF82B103D2714118DE573D227B346D8258`.
Protected performance milestone: setup 0,8870 s, denoise 29,7999 s,
steps 2–8 3,002195 s. No new performance result.

### Proposed sequence, still dependent on confirmation

In DDO, with ComfyUI and other GPU consumers closed by the user:

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src;E:\ProjetosIA\diffusers-dynamic-offloader\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-full-registration-contract-20261001
```

Hypothesis: complete coverage eliminates normal recycling without changing
DMA, events, parity, or fallback. All CUDA cases must execute and pass;
skip does not close a contract. Stop at first error/anomaly. Only then:

```powershell
$env:DDO_VMM_CAPACITY_GATE='1'
$env:DDO_VMM_FULL_REGISTRATION_GATE='1'
$env:DDO_VMM_CAPACITY_REGISTERED_LIMIT_BYTES='26440892416'
$env:DDO_VMM_CAPACITY_SOURCE_DIR='F:\hf_cache\hub\models--elismasilva--ltx2.3-image-distilled-1.1\snapshots\5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9\transformer'
& .\.venv-vmm\Scripts\python.exe -m pytest ddo-vmm-allocator/tests/test_source_capacity.py::test_persistent_source_directory_full_registration -q -x -s -p no:cacheprovider --basetemp .pytest-full-registration-capacity-20261001
```

Proposed capacity: 24,625 GiB source commit/registration + 1,5 GiB staging,
8 GiB RAM/commit headroom in native guard; GPU destination 2 MiB.
Criteria: 394 total registrations, zero unregisters/waits before trim,
second pass without reread, correct samples, zero additional host copy or
fallback, ledgers zeroed on closure. Failure blocks harness.
Remove the three authorization/ceiling variables after the gate; do not leave
them enabled for future generic suites.

After both gates and A/B scope confirmation, in sibling host, using DDO Python
and same PYTHONPATH/DLL, execute in fresh processes:

```powershell
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 0.5 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/registration_control_20261001
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/registration_candidate_20261001
```

Do not overwrite existing artifacts. BF16/real prompt and other defaults remain
equal; manifest must confirm this. Do not compare external sampler with total
minus prompt as though scopes were identical.
Check control before candidate; anomaly or loss of comparability stops sequence.
Candidate requires equivalent image, registration/retention within ceilings,
no churn after warmup without pressure, no new failures/fallback, and times
no worse than valid control. Record first step, steady-state, setup, and total
separately. One A/B run identifies a candidate; does not prove statistical gain,
automatically replace milestone, or close linear/NVMe regression.
Overlap and greater GPU residency will be addressed only after this result,
without changing three variables simultaneously. Protected-route regression
benchmark remains mandatory before promotion/delivery.

## Native two-slot and batch pipeline — implementation of 2026-09-30

### Setup accounting — contracts passed, performance not approved

Remaining cost inspection found an untimed copy path: auxiliary linears outside
arena move entirely to CUDA during setup. The roughly 0,364 s still unattributed
in the 0,8870 s milestone cannot be called planner overhead without separating
these copies. Fast route now records metadata, native resources, and auxiliary
linears as distinct scopes. Auxiliaries preserve exactly the transfer method,
order, and timing; no stream/buffer created or work delayed into first step.

**119 CPU tests passed in 5,29 s**, including simulated copy-call equivalence,
bytes actually eligible for transfer, already positioned modules, and fast-only
metrics versus linear/NVMe. First CPU/meta fixture required correcting simulation
because PyTorch disallows changing Parameter.data from CPU to meta; transfer
code was not changed to bypass this test restriction. DLL and budgets unchanged.
Approved CUDA gate: **37 tests passed in 6,66 s**, exit 0. Then one approved
eight-step run, real prompt and unchanged parameters, finished with exit 0 and
same image hash/backing/transfers/fallbacks.
Setup 1,2644 s: attach 0,6487 s, auxiliaries 0,3882 s (0,4122 GiB), small
parameters 0,1768 s, metadata 0,0231 s, resources 0,0071 s, root 0,0003 s.
Approximate residual 0,0202 s. Auxiliary copies are a proven cost in this run;
do not use this measurement to attribute the historical residual exactly.

Denoise **36,3482 s**, versus protected milestone **29,7999 s**: run
**not promoted as a performance result**. Unchanged encoder/connectors also
slower, without demonstrated cause. Global baseline-subtracted GPU peak remains
6,9727 GiB; encoder peak increased. Error-free monitor does not prove absence
of interference. Instrumentation remains applied, no rollback or new ceiling
change. Controlled comparison requires new approval; do not repeat automatically.
Details and artifacts in sidecar.

### Grouped setup views — CPU/CUDA and benchmark passed

Attach now prepares/releases the temporary view range once per contiguous
structural unit instead of repeating map/unmap of the same pages per tensor.
Uses existing native API and its single physical bootstrap page, aliased across
the unit's virtual pages; not weight materialization or increased residency.
Only current unit is mapped: unit changes and exit/error release the range.
No compute allowed in this scope; hooks only used after attach completes.
Release error prevents successful attach.

Span refusal retains previous individual/fallback path. Offset, shape, dtype,
source, batch, DMA, and event protection unchanged. Setup ranges retain only
integers, no CPU tensors. No DLL or linear/NVMe route changes.
Prepare/create/release counts/times delivered by public report, including failed
attempts.

Final consolidated CPU gate: **113 passed in 5,03 s**, with error scopes,
fallback, alignment, ownership, presets/profiles/backend and previous contracts.
Expanded CUDA gate prepared for original/alternative names and one/two-tensor
units, preserving original test. Approved gate executed:
**37 passed in 5,43 s**, exit 0. Then approved real eight-step case:
setup 0,8870 s, attach 0,3931 s, denoise 29,7999 s, steps 2–8 average
3,002195 s. Same hash, backing/H2D/read/reuse and zero failures/fallbacks.
No monitor errors; baseline-subtracted GPU peak 6,9727 GiB matches previous.
1152 views use 48 prepare/release pairs, scopes 0,017449/0,053906 s;
DLPack creation 0,099560 s. Single-run comparison, not statistical proof.
Initial steady range reached in this measurement; setup 0,16 s and 1,3 s/it
target not yet. Linear/NVMe regression gates and text encoder/UNet expansion
not closed by this sequence. Detailed results in sidecar.

### Structural preparation — CPU/CUDA gates and benchmark passed

Experimental adapter discovers modules once and shares layout between sizing
and attach. Selection preserves regex, descendants, parameter order/alignment.
Units come from `nn.ModuleList` items, without architecture names; nested lists
use outer item as fence. Outside these lists, linears remain individual units.
No model-family detection or new convolution support; explicit public filter
still respected, without changing defaults/presets.

Immutable arena geometry (address/page/virtual size) queried once; residency,
signatures, and pressure remain live native queries.
No DLL, budget, event, DMA, or linear/NVMe runtime changes.
**58 CPU tests passed in 5,40 s**, including selection/offset equivalence,
alternative names, 2/10 order, UNet-like nested structure, individual fallback,
and layout use without rediscovery. **45 CPU preset/profile/backend tests passed
in 4,81 s**. These are not performance/CUDA gates.

Existing small CUDA test parameterized for original name and `layers`,
preserving parity, cancellation, ceilings, and event protection. Approved sequence
passed **35 CUDA tests in 5,89 s**, exit 0, then one real eight-step run:
same image hash, equivalent backing/H2D/read/events, zero failures/fallbacks.
Setup 1,1137 s (before 1,3067), attach 0,5874 s (before 0,7862),
denoise 30,8719 s (before 31,4093). Baseline-subtracted GPU peak 6,9727 GiB,
same as previous; does not indicate room to increase budget.
No monitor errors. Results in sidecar.

Single comparison: small denoise difference does not prove statistical gain;
0,16 s setup not reached yet. Linear/NVMe code unchanged, but regression
benchmarks for these routes were not part of this sequence.

Persistent route now creates a C++ pool with two reusable slots. Geometry comes
from largest execution unit, rounded to native page; total charged to existing
GPU ceiling, without increasing it. C++ alternates slots, keeps backing protected,
and chains reuse to last compute event.
Look-ahead can prepare next slot while current one executes. Cancelling a unit
uses upload event, without global synchronization on normal path.

Each unit prepares immutable descriptors once (retained file owners, offsets,
sizes). Source-cache ABI v11 / stats v8 submits batch in C++: window requests,
reads, registration, DMA, fallback, and ticket recycling.
Ticket events reused only after completion; one persistent event covers the
entire batch, including bypass and partial failure. Failure of
recording that event preserves ownership and blocks unsafe closure.
Python views are cached by unit/slot; hooks still bind parameters and orchestrate
units, without Python residency or cache-admission policy.

Host metrics now include complete request/copy/close scopes, event management,
registration scheduling, and batch submit/reap. These are nested scopes, not
device DMA times or evidence of actual overlap. Previous snapshot prefixes
remain compatible; linear/NVMe routes unchanged.

Release compiled; **42 CPU tests passed in 4,50 s**. DLL SHA256:
`F189D035D10089AE77DD9A7F21EED2FF82B103D2714118DE573D227B346D8258`.
Approved consolidated CUDA gate: **34 passed in 5,32 s**, exit 0, same DLL
and DDO environment. Closed three-pass parity, cancellation/slot rotation,
batch ownership, event/source reuse, and previous snapshot prefixes.
Subsequently approved four-step LTX diagnostic finished with identical output,
zero evictions/temporary fallback, and 3,59375 GiB backing within 4 GiB ceiling.
Denoise 87,9825 s, steps 2–4 average 20,6509 s; 8,03% improvement versus
single slot (95,6604 s), still far from control 36,3550 s.
Not final-target validation or proof of full-model overlap.

Broad upload scope 66,7107 s versus native submit 13,5557 s leaves 53,1550 s
unattributed. Batch/metadata creation and Python/stream boundaries sit outside
native timer; close these scopes before another benchmark/tuning.
Pooling worked (32 events created, 5632 reused), with small measured cost.
Full results in `experiments/dynamic_offload_results.md`.
Do not expand budget or start another case without approval.

### Upload scope closure — approved validation executed

Additional Python adapter instrumentation separates transport dispatch/prepare/
create/submit, creation and submission at ctypes boundary, lock waits, argument
preparation, result, and existing stream-context entry/body/exit.
Does not change DMA, registration, lifetimes, selected stream, residency, or
DLL/ABI. Metrics follow public DDO report; host unchanged.

**47 CPU tests passed in 4,74 s**, including deterministic clocks to close sums
and cover partial failures. Syntax checked. Approved sequence:
**34 CUDA tests passed in 6,07 s**, followed by same four-step LTX.
Identical output, no evictions/fallback, backing 3,59375 GiB within 4 GiB ceiling;
denoise 80,4517 s, 2–4 average 18,2180 s, total 94,1583 s.
No gain attributable to instrumentation: H2D and execution not optimized in
this run.

Scopes located residual: Python transport/submit 67,25 s, FFI call 67,2392 s,
C++ function 13,4883 s (232 calls in both). Creation, metadata, locks, and stream
entry/exit are small. Dominant residual lies within DLL call interval;
its final cause not demonstrated yet.
Host monitor queries USS every 100 ms and also queries CUDA: telemetry/GIL
interference is now a concrete hypothesis to validate, not a conclusion.
Do not silently replace USS with RSS or attribute gains to removing metrics.
Next observation comparison requires approval; do not expand budget.

### Host observer correction — explicit RSS, LTX diagnostic approved

Host `RunTracker`/`PeakMemoryMonitor` have opt-in `rss` mode, passed by runner
via `--memory-monitor rss`. Uses cheap working set instead of USS scan every
100 ms. `full` default preserved; do not silently change old controls/routes.
GPU still queried by same method/cadence. JSON and console identify RAM definition;
RSS and USS are not interchangeable. Each phase records RAM/GPU query costs,
sample counts, and errors. Failures invalidate monitor, without hiding them.

**62 CPU tests passed in 4,66 s**, including previous adapter/configuration
contracts and new monitor/lifecycle/labeling tests. Syntax checked.
No DLL, allocator, residency, source, transfer, or budget change.
After approval, same four-step run with explicit RSS ended exit 0:
denoise 18,8714 s (before 80,4517 s), steps 2–4 average 3,2788 s,
total 29,0629 s. FFI/native difference fell from 53,750849 s to 0,003168 s.
Image hash, 3,59375 GiB backing, 95.137.822.848 H2D bytes, and source reads
unchanged; no temporary fallback or failure. Sampled VRAM peak 6,5172 GiB;
RAM 27,2251 GiB RSS not comparable with previous USS. GPU queries remained
active; no monitor errors. Results/command in experiment sidecar.

Contrast supports USS observer interference as dominant cause of previous
residual, but does not demonstrate precise GIL mechanism without a trace.
Not a gain from a new native algorithm. `full` default remains intact.
Approved eight-step/real-prompt benchmark also ended exit 0:
denoise 31,4093 s, steps 2–8 average 3,150964 s, transformer setup
1,3067 s, prompt phase 19,7553 s, total 61,7671 s (42,0118 s without that phase).
FFI/native difference 0,005513 s; zero failures/temporary fallback,
backing 3,59375 GiB. Image visually checked; no real-prompt numerical parity
control in this sequence. GPU peak 6,9727 GiB subtracts baseline around
1,03 GiB and does not indicate room for budget expansion.

Average still exceeds 3,13 s by roughly 0,67%; prepare/setup has not reached
0,16 s target. Do not equate total without prompt with reference sampler scope.
Text encoder remains group offload; native expansion and regression/delivery
gates still pending. New execution or parameter change requires approval.

## Persistent storage correction / separate registration (2026-09-30)

Implemented in C++ as config v5 / ABI v8 and integrated into public route.
`vmm_source_cache_gb` ceiling limits persistent host storage; new
`vmm_source_registered_gb` limits CUDA registration within that storage.
Zero preserves same ceiling, no automatic increase; positive values are whole
64 MiB segments and cannot exceed storage ceiling.
Host passes `--ddo-vmm-source-registered-gb`, without implementing policy.

C++ allocates/commits segments progressively and fills each immutable segment
once. Before DMA, registers source under its ceiling; if needed, reuses another
segment's registration after its transfer event. Unregister preserves bytes
and identity. Leases protect content, DMA events protect registration, arena
pins protect destination/compute: independent lifetimes.
No loader CPU tensor or parallel PyTorch cache.

Registration refusal allows transfer of same retained bytes through existing
native ring, with host→host copy accounted, without rereading file.
Fallback larger than one slot progresses in chunks, waiting only on slot event
before reuse. Actual RAM/commit pressure can remove ready unleased content;
registration trim in config v5 does not discard content.
Process ledger accounts commit separately from registration, including staging
in admission limits, without obtaining a full budget per component.

Configs v1–4 preserve behavior and ABI prefixes; linear/NVMe and their defaults
unchanged. Retention capacity remains explicit: 512 MiB does not cover a
24 GiB model. No full-model buffer authorized/allocated.
Implemented correction does not mean demonstrated gain.

Release DLL compiled. After implementation without tests, user approved
consolidated CPU/CUDA gate: **63 passed in 5.67s**, no failures, using DDO
`.venv-vmm` and `-x`. Sequence included capacity configuration, prefetch ownership,
native cache, public transport, and fast runtime.
Passed unregister/reregister without rereads, recycling with live DMA ticket,
retained-byte fallback below/above one slot, and independent limits.
DLL SHA256: `F345CA76B8AB5BDD052AA38DADE264DCF1D835C569A73B7E62C5AF80C48BCCC8`.
No LTX benchmark or full-model capacity gate executed.
Linear/NVMe performance regression and inference gains remain pending;
previous numbers do not validate this DLL's performance.

Subsequent capacity gate (staging 1536 MiB, retention 512 MiB,
persistent registration 128 MiB): **1 passed, 1 failed in 5.96s**.
Legacy control passed; persistent test verified data and no rereads, but
confused 16 DMAs with 16 registrations (15 observed, with valid reuse).
Expectation corrected to 14–16 and exact registration balance, without changing
runtime/DLL or capacity. Trim and final checks not reached in this case;
gate remains pending until approved repetition.

Approved repetition, same command/DLL/ceilings, only expectation corrected:
**2 passed in 6.14s**, exit code 0. Closed legacy and persistent controls:
512 MiB retained + 1536 MiB staging, source registration limited to 128 MiB,
16 correct transfers without rereads, no host→host fallback, trim without content
loss, clear/close with zeroed ledgers. Aggregate guard rejected additional staging
at commit ceiling. This 2 GiB host gate does not prove full-model coverage or
LTX performance gain.

LTX A/B approved after gate, four steps and same previous ceilings:
denoise **36.3550 s without cache / 40.1332 s with 512 MiB**, steps 2–4
**8.8628 / 9.9731 s**. Both finished with identical PNG output, no fault/
temporary-fallback failures. No demonstrated gain.
H2D remained 93,927,995,136 bytes; only 497,850,384 from retention (23 hits).
512 MiB filled/registered once, no unregisters or waits; registration ceiling
equal to retention did not exercise recycling in this harness.
Partial coverage still leaves almost all traffic in ring. Do not repeat this
scenario to eliminate full-model traffic; next advance requires explicitly
approved source coverage and RAM guards, not new PyTorch heuristics.
Details in `experiments/dynamic_offload_results.md`.

Full-shard coverage gate, separately approved: **1 passed in 15.08s**.
Real source 26,248,801,544 bytes, 394 segments, commit 24.625 GiB,
staging 1.5 GiB, aggregate ceiling 26.125 GiB, source registration limited to
512 MiB, RAM/commit margin 8 GiB. Second pass: 394 hits, zero rereads;
no bypass/pressure/fallback. Trim preserved bytes and clear/close zeroed ledgers.
GPU destination 2 MiB; only 17-byte samples per segment compared;
approved capacity/lifetime do not prove DMA throughput or transformer performance.
LTX benchmark at this capacity still pending.

Subsequent full benchmark approved (24.625 GiB retention, 512 MiB registration,
1.5 GiB staging, 4 GiB arena, four steps): exit 0 and identical PNG, but
**122.5626 s denoise / 28.0650 s steps 2–4**. Zero ring/bypass reads:
all H2D copies (93,927,995,136 bytes) came from retained bytes.
388 actually demanded windows, 5192 hits, source read 25,846,148,360 bytes.
1413 registrations / 1405 unregisters, current/peak registration 512 MiB,
no capacity waits/pressure/fallback. Coverage solved rereads; narrow registration
ceiling caused intense recycling and execution regressed substantially.
Registration time not isolated: churn is concrete suspect, not proven cause.
Do not increase ceiling or repeat benchmark without approval and RAM/commit guard.

Next instrumentation prepared: ABI v9 / stats v6 adds host seconds for
registration, unregister, and explicit recycling waits, preserving config v5
and old stats prefixes. Does not change ceilings, admission, events, or policy;
adds no synchronization. Release DLL compiled, hash
`16E872447364F7BFE155CA4739145715A6337047EF5FA6B1FDA2A64C25BA009B`.
Contracts prepared for timings and v4/v5 snapshots without suffix writes.
No CPU/CUDA test or benchmark executed after instrumentation;
ABI v8 results above do not validate new DLL.

ABI v9 sequence subsequently approved: **65 passed in 7.37s** in selected
contracts, including times and old snapshot prefixes.
Next full benchmark passed RAM/commit preflight (45.694/71.230 GiB), but
exited code 1 before prompt/transformer/denoise events:
`OPENSSL_Uplink(...,08): no OPENSSL_Applink`. No report/image or new timing
measurement. Sequence stopped without repeating or changing dependencies/
parameters. Cause not yet determined; ABI v9 performance remains pending,
although selected functional gate passed.

Startup diagnostic without CUDA: failure reproduced in
`ssl.create_default_context()` without Torch/DDO. Inherited `SSLKEYLOGFILE`
enables TLS logging; removing it only in diagnostic child made default context
pass, retaining mandatory certificate and hostname verification.
No dependency/DLL replacement or global change. Next proposed repetition omits
this variable only at launch and preserves all limits; not executed.

ABI v9 repetition approved, omitting variable only at launch:
exit 0, identical PNG, denoise **132.1528 s**, steps 2–4 **29.9801 s**,
RAM/VRAM peak **27.1082/6.5367 GiB**. Full source without reread/bypass,
same 1413 registrations/1405 unregisters. Measured times: registration
**0.5851 s**, unregister **6.3593 s**, capacity waits **0 s**.
Direct cost sum 6.9444 s does not explain entire regression; hypothesis of
registration as main bottleneck not supported by measurement.
Broad host upload scope **80.2949 s** mixes source readiness, scheduling/
retirement, and enqueue, not pure DMA.
Do not increase ceiling blindly: next diagnostic must separate native transfer/
waits and compare ring/retained-source transport. No new tests.

Additional separation prepared without new policy: ABI v10 / stats v7 adds
host time for source readiness in direct upload, native H2D submission call,
event recording, and waits when closing submitted ticket. Does not measure DMA
via GPU event, include entire Python layer, or add synchronization.
Stats v1–v6 prefixes preserved; config v5, ceilings, failure handling unchanged.
Release DLL compiled, hash
`BF2FBFE702A19166DFB13BBD3566D40D8E129E702DC3E2E532E8FFFD9917D5FF`.
Contracts expanded for new fields and v4/v5/v6 sentinels; static checks passed.
In preparation, no CPU/CUDA test or benchmark executed with ABI v10.

Subsequent consolidated sequence approved: **66 passed in 6.13s**, then
full four-step diagnostic with same ceilings, exit 0 and identical PNG.
Denoise **136.9374 s**, steps 2–4 average **31.6803 s**, total **150.4132 s**.
Source readiness 5.2250 s, native submission 0.0377 s, event recording 0.0049 s,
close waits 0.0030 s; registration 0.5468 s / unregister 6.4005 s.
Broad upload scope 66.6854 s: these counters do not explain entire scope or
measure GPU DMA duration.
H2D remains 93,927,995,136 bytes despite full retained source. Model virtual
arena and full host storage are not equivalent to full CUDA backing;
GPU residency still limited to 4 GiB. Gain not approved.
Do not repeat or expand capacity before closing architectural difference and
obtaining new confirmation. Full results in experiment sidecar.

## Architectural gap confirmed after ABI v10

Full execution recorded 2663 arena page evictions, 360 signature hits,
zero pressure evictions and zero temporaries. Active ceiling makes
`fault_range` evict pages to admit new addresses; it does not set an admission
limit routing remainder into a reusable native execution buffer.
Normal route therefore still maps/copies/unmaps weights throughout model,
even when all host bytes are retained.

Adapter `DeviceTransferBuffer` is only a `torch.empty` wrapper;
its `ensure()` is not consumed by current route. Exceptional fallback
`temporary_cuda_source()` still uses transient host source and `.to(cuda)`.
Do not treat this object as implementation of target native CUDA buffer.

Next closure: native admission-limit contract + reusable per-stream CUDA execution
buffer, views without PyTorch allocation ownership, persistent-source→direct DMA,
and compute event protecting reuse.
Preserve complete virtual arena, admitted-page signatures, safe exceptional
fallback. Do not raise budget or alter linear/NVMe.
Current numbers identify functional difference, not how much total time it
explains. Requires implementation and explicitly approved CUDA gate before a
new harness; do not repeat unchanged full cache.

## Execution buffer connection — implemented, CUDA pending

On 2026-09-30 the persistent-window-source route began configuring a native
admission limit before first materialization. One execution unit's capacity
(largest, page-aligned) reserved within existing GPU ceiling; not added to budget.
Units crossing limit use a reusable C++ mapped buffer.
No `.to(cuda)` or PyTorch backing allocation on this normal path: tensors are
only views and persistent-source/ring transport uses existing native API.

Buffer keeps its pages pinned. C++ records compute event and enqueues dependency
on upload stream before next overwrite.
Cleanup drains that local event; recording/drain failure retains ownership and
stops execution. Creation/admission failures do not increase capacity.
Limit can decrease under actual pressure; adapter observes native decision.
Look-ahead does not overwrite buffer used by current compute. This connection
has one reusable GPU slot, not a double-buffering pipeline or final-performance
promise. Routes without persistent source, linear, and NVMe preserve existing
paths.

Release compiled; DLL SHA256
`86DFD402FCAA0E9D4853697D920674CB3457EE7E8C2BA3E5A58C0EC1191AE505`.
Python syntax validated. Ownership/dispatch/look-ahead CPU contracts:
**11 passed in 4.67s**. Small public CUDA gate prepared in
`tests/test_vmm_execution_buffer.py`: three passes, eight-unit model,
4 MiB GPU ceiling, 1 MiB page, 64 MiB source, 2 MiB staging.
Requires numerically equivalent output, signature reuse, no rereads on later
passes, zero tail evictions/temporaries, aggregate residency within ceiling.
During preparation, not executed yet.
Previous ABI v10 benchmarks/gates do not validate this DLL or performance.

First approved execution: **1 failed in 5.03s**, exit 1, stopped by `-x`.
First pass passed numerical comparison and exercised buffer, but test required
1 MiB backing and observed 2 MiB. Requested page size is aligned to minimum
driver granularity in C++; expectation corrected to exact capacity of a 1 MiB
unit rounded to reported native page. No ceiling, runtime, or DLL changed.
Subsequent aggregate ceiling, eviction, three-pass, signature, and reread checks,
and remaining sequence tests, not reached.
Not an approved CUDA gate; repetition requires new confirmation.

Explicitly approved repetition, same command, DLL, ceilings, and basetemp,
only alignment expectation corrected: **32 passed in 5.33s**, exit 0.
New public contract completed three passes with numerical parity, exact capacity
aligned to native page, aggregate residency <=4 MiB, signature reuse, no new
source reads after first pass, zero tail evictions and zero temporaries.
Existing public transport, fast runtime, and source cache contracts also passed.
Closes selected functional gate, not proof of maximum overlap, no linear/NVMe
performance regression, or LTX model gain.
No additional benchmark executed; separate approval still required to measure
this DLL with comparable harness.

Subsequent LTX benchmark explicitly approved, four steps and same ceilings:
**95.6604 s denoise / 23.0855 s steps 2–4 average / 109.2922 s total**.
Exit 0, identical PNG; zero arena evictions/temporaries and 168 uploads through
native buffer. Arena 3.03125 GiB + buffer 0.53125 GiB = 3.5625 GiB weight backing,
below 4 GiB ceiling. Global VRAM/RAM peak 6.3416/27.1073 GiB.
Mechanism eliminated observed remappings, but performance criterion failed:
95.6604 s still exceeds cache-off control 36.3550 s despite reduction from
previous 136.9374 s. H2D remained 93,525,317,376 bytes and broad upload scope
65.5584 s. Do not attribute all cost to DMA, registration, or single slot:
narrow timers do not cover entire operation.
No new case, capacity increase, or eight-step benchmark.
Preserve functional closure; next diagnostic must close unaccounted boundary/
event work and overlap difference, without repeating unchanged scenario or
promising equivalence from full cache.

## Validation history before ABI v8 correction

Previous status: updated on 2026-09-30; immutable source, registered cache,
small public integration, and 768 MiB host registration gate validated in CPU/CUDA.
Harness configuration parity validated in CPU and four LTX cases executed;
2 GiB host gate passed. Model coverage and performance still pending;
partial cache regressed in 4-step case.

Validated progress:

- Window transport restoration gate: 3/3 CUDA contracts passed before change,
  in DDO environment.
- Immutable source: ABI v1, file identity, range/metadata validation, retained
  handles, positioned reads shared with ring, and ownership wrapper implemented.
  Release compiled; 14 source contracts + 1 platform test passed without CUDA
  initialization.
- CUDA regression of rebuilt DLL: 3/3 contracts passed in 5,33 s, no failures
  or warnings, after user confirmation. No weight cache activated and no new
  benchmark or performance-gain claim.
- Next: complete source scheduling/tickets/cancellation with registered cache.
  Current foundation is a read primitive; it does not replace existing
  asynchronous pipeline with synchronous runtime reads.

Current block 3 increment: `ddo_vmm_source_cache.cpp` retains content of registered
ring slots, reuses its asynchronous read pool, and maintains independent events
per H2D ticket. Exact/contained hits share bytes; misses without slots and
partial overlaps return bypass. `clear`/`close` refuse live tickets; cancellation
keeps I/O under cache ownership until drain. Native snapshots measure capacity,
retained content, tickets, hits/misses/bypass, reads, and DMA.

First validated increment limit: advance slot allocation, hard 128 MiB ceiling,
microcontract-only use. A lifetime validation step, not complete memory policy
implementation of section 2. That gate still lacked progressive growth,
aggregate cache+staging limits, pressure/hysteresis, polling, injected failure
coverage, temporary destination, large-range partitioning, and public integration.
Runner unchanged. Capacity will not expand before corresponding gates;
experimental limit not product default.

Release compiled; 15 source/platform tests without GPU passed in 4,90 s.
CUDA gate executed on 2026-09-22 after confirmation: six cache contracts +
three transport regressions, **9 passed, 11 deselected in 7.79s**, no failures
or warnings. Each new test registers at most 4 MiB host and uses small arena
(up to 4 MiB), without loading model. Previous contracts retain configurations.
Increment DLL SHA-256:
`C00F12D369E4C29DD22F854AD4A9D701807B1BF924C195A76186059F5DD63372`.

Command executed from submodule with DDO Python:

```powershell
& ..\.venv-vmm\Scripts\python.exe -m pytest tests/test_source_cache.py tests/test_native_ranges.py -m vmm -k 'source_cache or ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks' -p no:cacheprovider --basetemp ../.pytest-vmm-source-cache
```

Approved functional criterion: GPU eviction/rematerialization test preserved
host bytes, validated copied data, and counted only one source read for two
uploads. Partial retention/bypass, aliases, ticket ownership on closure, two
streams, and rejection from headroom/changed source also passed.
Not inference performance measurement or full-model capacity validation.

Next increment (2026-09-22), implemented, compiled, **validated in CUDA gate**:
cache ring starts with metadata/workers only; each admission registers one slot
after checking RAM/commit and process aggregate cache limit. Hits do not increase
registration. `clear()` drains reads and frees slots, returning aggregate capacity;
live tickets still block release. If cleanup partially fails, new requests remain
blocked until completion.

Smallest `process_limit_bytes` among live caches governs new admissions.
Creating a cache with a limit below already registered usage returns bypass
without changing others. Experimental ceiling remains 128 MiB. This ledger
still covers only caches: staging, other libraries, and automatic pressure
reduction not included. RAM/commit margin queried at each growth;
no OOM-based capacity search or GPU budget change.

Cache ABI v2 preserves C v1 prefixes and adds capacity, peak, aggregate usage/
limit, admission refusals. New wrapper requires DLL v2.
Release compiled; no-GPU regression: **15 passed in 4.59s**.
CUDA gate executed after confirmation: **12 passed, 11 deselected in 4.64s**,
no failures/warnings. Nine previous contracts plus three growth/two-cache/
capacity-return/stricter-limit contracts passed, including sentinel checking
v1 snapshot does not write beyond size.
DLL SHA-256: `69DF3D95EB722959C71CC0F8E3E4D6ABECC63F1642EF0D82F20A65D24A049588`.

Command executed in submodule with DDO environment:

```powershell
& ..\.venv-vmm\Scripts\python.exe -m pytest tests/test_source_cache.py tests/test_native_ranges.py -m vmm -k 'source_cache or ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks' -p no:cacheprovider --basetemp ../.pytest-vmm-source-progressive
```

Gate validates small contracts, not inference performance, full-model capacity,
or a new NVMe baseline measurement.

Joint accounting increment (2026-09-29), **compiled and validated in CUDA gate**:
native staging rings enter process ledger at creation and return registration
on closure. Cache admission considers this usage with its slots; existing ring
retains normal creation. ABI v3 adds `process_staging_registered_bytes` to
snapshot, preserving v1/v2 prefixes. Experimental 128 MiB ceiling remains.
Two small CUDA contracts prepared for bypass/return and rejection of a limit
below already registered staging. Automatic pressure reduction and runtime
integration still pending. Other-library memory excluded from ledger and still
observed through RAM/commit margin.
Release compiled; no-GPU regression: **15 passed in 16.18s**.
Approved gate executed in DDO environment: **14 passed, 11 deselected in 6.50s**,
no failures. Includes two new contracts and previous regressions.
DLL SHA-256:
`FE2FD87BFA12033DDC82DF8B58619688968A7E1FF1F26CCB4EC7FF186CC447D0`.

```powershell
& ..\.venv-vmm\Scripts\python.exe -m pytest tests/test_source_cache.py tests/test_native_ranges.py -m vmm -k 'source_cache or ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks' -p no:cacheprovider --basetemp ../.pytest-vmm-staging-accounting
```

Gate confirms accounting and lifecycle with small buffers. Does not measure
inference, performance gain, or NVMe baseline.

Pressure increment (2026-09-29), **compiled and validated in CUDA gate**:
cache requests query available RAM/commit. Below configured margin, new admissions
stop and ready slots without tickets return from tail to preserve slot indices.
Admission resumes after additional one-slot margin. Existing hits remain
available. `trim(target)` tests same release without exhausting machine RAM;
returns BUSY if active entry blocks target. ABI v4 adds release/byte/blocked-request
counters and suspended state, preserving v1-v3 snapshots.
Release compiled; no-GPU contracts: **15 passed in 5.04s**.
15 CUDA contracts collected and executed after approval:
**15 passed, 11 deselected in 4.86s**, no failures. DLL SHA-256:
`74D09DCDF8E23FFDE844C7FF7D812F3AC6D23B5CE15A067BAF524A017A025A17`.

```powershell
& ..\.venv-vmm\Scripts\python.exe -m pytest tests/test_source_cache.py tests/test_native_ranges.py -m vmm -k 'source_cache or ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks' -p no:cacheprovider --basetemp ../.pytest-vmm-source-pressure
```

Automatic release only occurs on next cache request. Not a background pressure
response, no actual-pressure test yet; new test exercises explicit-target release
and ticket protection.

Integration prerequisite (2026-09-29), **compiled and validated in CUDA gate**:
H2D ticket now exposes native event completion query, without stream/device wait.
Wrapper can only release ticket after query returns completed; CUDA query failure
is error, not pending. Cache ABI v5 adds symbol without changing v4 snapshot
layout. Two-stream contract tests query before enqueue, after enqueue, and after
completion.
Release compiled; no-GPU source contracts: **15 passed in 4.67s**.
15 CUDA contracts executed after approval:
**15 passed, 11 deselected in 4.82s**, no failures. DLL SHA-256:
`45257BDDBFDF360308B53C5126807531C3DF0A111463E0C9051048EE1F26369D`.

```powershell
& ..\.venv-vmm\Scripts\python.exe -m pytest tests/test_source_cache.py tests/test_native_ranges.py -m vmm -k 'source_cache or ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks' -p no:cacheprovider --basetemp ../.pytest-vmm-ticket-poll
```

DDO boundary review showed two integration points still open:
`_prepare_vmm_fast_runtime` instantiates `PinnedStagingPool`, while native cache
was validated with ring; micro 128 MiB capacity includes registered staging,
below sizes used by model harness.
Model integration must configure aggregate host limit explicitly and preserve
safe temporary path when materialization fails.

Experimental submodule adapter integration (2026-09-29), **small-model CUDA
contract passed**: native transport receives cache as opt-in, routes admitted
ranges through tickets and remainder through ring. Completed DMA tickets collected
by nonblocking query before next batch; closure drains remainder.
Temporary fallback uses transient positioned reads with safe synchronous copy
when no PyTorch staging available.
Ring prefetch suspended while cache active to avoid pre-reading a hit.
Cache metrics enter store snapshot. Small test covers numerical result, weight
hit, bias bypass, ring/cache 2 MiB each and aggregate 4 MiB limit.
No-GPU adapter contracts: **4 passed in 4.52s**.
Approved adapter CUDA gate passed **2 passed, 9 deselected in 4.79s**:
`python -m pytest tests/test_module_adapter.py -m vmm -k 'native_source_cache_adapter_reuses_weight_and_bypasses_bias or model_arena_file_backed_binding_uses_native_shard_transport' -p no:cacheprovider --basetemp ../.pytest-vmm-adapter-source-cache`.
Second upload forces only signature invalidation in test;
not physical GPU eviction or inference performance evidence.

Public `_prepare_vmm_fast_runtime` now accepts native cache as opt-in
(`DDO_VMM_NATIVE_SOURCE_CACHE_SLOT_MB` and `DDO_VMM_NATIVE_SOURCE_CACHE_SLOTS`).
Without it, previous PyTorch staging remains. With it, DDO uses only submodule
registered ring and validates native aggregate ceiling before attach.
At first integration ceiling was 128 MiB and 640 MiB staging rejected.
Aggregate 4 MiB limit (2 MiB ring + 2 MiB cache) and absence of second staging
have public CUDA contract.
In first gate new test passed, legacy test failed by requiring an event in all
slots after forward using only one. Expectation corrected to at least one event;
approved CUDA rerun passed **2 passed, 2 deselected in 4.67s**.
No-GPU configuration/regressions: **46 passed in 4.73s**.
No performance benchmark at this stage.

Next-block review: simply raising 128 MiB number is unsafe.
Before change `_NativeShardTransport` created eager ring first, then cache;
aggregate cap applied at cache admission, when ring was already registered.
Order reversed; native ring creation now checks aggregate cap and RAM/commit
before registering bytes while cache active.
In gate ceiling remained 128 MiB; Release compilation and no-GPU contracts
passed (**19 passed, 7 deselected in 4.59s**).
Approved CUDA ring rejection/rollback and adapter regression gate passed
(**2 passed, 22 deselected in 4.76s**).
Expansion must separate local cache ceiling (128 MiB) from experimental aggregate
ceiling (minimum needed: 640 MiB ring + 128 MiB cache).
Partial failure must close both and return accounting. Separate gates:
limit/rollback microcontracts; then explicit 640+128 MiB test; only then model
harness, without changing 4 GiB GPU budget.

Expansion prepared in ABI v6: native configs v1/v2 preserve aggregate cap up to
128 MiB; config v3 accepts up to 1 GiB cache + ring.
Local cache capacity remains 128 MiB and each slot up to 64 MiB.
Binding selects v3 only for explicitly larger cap and diagnoses old DLL.
DDO calculates cap from actual ring + local capacity; 640 MiB staging +
128 MiB cache yields 768 MiB. Release build compiled; CPU contracts:
**48 passed in 4.83s** in DDO, **19 passed, 7 deselected in 4.58s** in submodule.
Approved compatibility CUDA gate passed **17 passed, 11 deselected in 4.85s**;
approved 640+128 MiB capacity gate passed **1 passed in 4.62s**.
Configured cap does not register cache bytes in advance.
DLL SHA-256: `5F676304490E92AEB41E9CEED59EF0FF054B6E8B0587FFE86A4E0838185C936E`.

`tests/test_source_capacity.py` gate prepared to register 640 MiB ring and
progressively fill two 64 MiB cache slots from synthetic file.
Checks 768 MiB cap, refuses additional ring, observes zero accounting after
clear/close. Requires explicit opt-in `DDO_VMM_CAPACITY_GATE=1`.
Approved gate passed in 4.62s: ring=640 MiB and cache=128 MiB when filled;
both zero on close. Does not load LTX model.

## Objective and evidence limitations

Preserve immutable weight bytes in native CUDA-registered host memory,
to reuse through asynchronous H2D after GPU arena page eviction.
Implementation restricted to `vmm_fast_runtime`, in `ddo-vmm-allocator` submodule.
`linear_runtime` and `vmm_runtime` remain protected.

Current transport already has registered host slots. New capability is source
retention between uses; changing pinning API alone does not provide it.
File remains recoverable origin. Cache will own native buffers, filled by file
reader, without retaining transformer CPU PyTorch tensors.
Allows retaining weight bytes in RAM: explicit expansion of previous source
requirement without weight retention, preserving prohibition on second copy
in CPU tensors. No new Python safetensors cache or full-shard mapping as default.

Three important test interpretation limits:

- `93,11 GiB` in four-step diagnostic are bytes requested from file reader,
  not measured bytes physically read from SSD. OS cache may serve some reads.
- Cumulative ticket/wait/enqueue times can overlap. Not additive portions of
  total time or standalone measures of DMA duration.
- Registered memory, process commit, working set, arena virtual reservation,
  local/nonlocal DXGI usage are different observations. Not demonstrated
  that registering a source of a given size produces equal nonlocal usage,
  or that this alone explains performance target.

Functional success means source reuse without file rereads or host→host copy
on hit. Performance success requires reducing measured latency, including cache
population cost. Increasing memory monitor number is not a gate.

## Starting state and protected regressions

Parity audit after approved public gate: recent host two-step JSON reports
`vmm_staging_buffer_gb=0.5`, 32 MiB pages, 48 units/1.152 tensors, active
4 GiB budget. Current runner passes value as per-slot size and
`native_transport_slots=3`; current configuration therefore corresponds to
1.536 MiB ring, not 512 MiB total.
JSON does not record effective ring capacity, so alone does not prove usage
in historical execution. Do not retrospectively correct measurements by assumption.

New public API interprets staging as total and experimental cache+ring cap is
up to 1 GiB. Migrating runner with `0.5` without conversion reduces capacity;
using 1.536 MiB plus cache exceeds validated ceiling.
Before any benchmark, decide with user between preserving current configuration
through new expansion block/gates or explicitly approving new manifest with
smaller ring. Do not silently change arguments, execute LTX, or expand host
registration.

Public parity preparation (2026-09-29): `vmm_native_transport` allows same ring
with cache off/on; cache remains opt-in. Argument helper now passes transport
selection and slot size/count
into DDO configuration, without requiring runner environment variables.
Staging means **total** capacity, rounded once into windows up to 64 MiB.
PyTorch and protected NVMe defaults unchanged. CPU configuration/argument/preset
regression: **54 passed in 4.75s**. First attempt had a `NameError` failure in
a new test from an assertion placed in the wrong test; assertion moved before
this result. No runtime failure.
Expanded CPU regression including profiles/bootstrap: **70 passed in 4.81s**;
lint of changed argument/test files also passed.

Public gate with three generic layers, source alias, three passes,
2 MiB ring and 2 MiB cache off/on.
Checks numerical output, signature hits, retention/bypass, JSON, and zeroed ledger
after closure. Harness still calls adapter directly and multiplies stated
capacity by three slots. Confirm historical command and preserve total capacity,
pages, unit selection before migration; 640+128 MiB registration gate does not
authorize benchmark change.
Enabled cache suspends speculative ring prefetch: declare this difference in
future A/B, without attributing all result solely to retention.

CUDA gate executed after confirmation: **2 failed, 4 passed in 4.87s**.
Both new variants failed numerical output, with/without cache; four previous
public contracts passed. Harness progression blocked until correction and
gate repetition. Inspection found reuse within same ticket: each tensor occupies
at least one window even when byte sum fits one slot.
Two-window batch in one-slot ring overwrote first range before enqueuing DMA.

Adapter correction prepared: demanded uploads split into tickets with at most
physical ring slot count, including a tensor larger than ring;
reuse waits only on slot event. Speculative tickets larger than ring refused
before I/O. No memory increase, residency change, or DLL alteration.
CPU transport/adapter contracts: **7 passed, 7 deselected in 4.47s**.
Evidence does not authorize LTX execution.
Expanded CPU coverage for cache bypass: **8 passed, 7 deselected in 4.46s**,
new test lint passed. Includes small ranges totaling less than a slot and
single range larger than entire ring.

Approved corrected CUDA repetition: **1 failed, 5 passed in 4.80s**.
Both variants passed numerical output across three passes, signatures, JSON;
cache variant also passed zeroed-ledger probe. Without cache, failure occurred
only when creating probe after close: cache API requires current primary context
and arena closure may remove it from thread.
Reactivation via one-element PyTorch allocation attempted before probe,
without changing zero-registered-byte check. Approved repetition still had
**1 failed, 5 passed in 4.81s**, at same probe creation:
an allocation served by PyTorch cache does not guarantee current context.

Preparation now uses explicit CUDA Driver API control only in tests:
retain primary context, push on thread, probe, pop/restore, release,
including probe exceptions. No runner workaround or runtime/DLL/registered-memory
change. Five CPU tests with simulated driver validated ownership/failures:
**5 passed in 0.03s**; lint passed.
CUDA repetition with explicit preparation approved/executed in DDO environment:
**6 passed in 4.77s**. Both variants passed numerical results across three passes,
signatures, descriptors without runtime-retained CPU source, JSON, and zeroed
cache/ring ledger after closure; four previous staging/cache/fallback/profile
public contracts also passed. No DLL, GPU budget change, or LTX execution.
Closes this small gate, not full-model cache performance/capacity validation.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:DDO_VMM_LIBRARY_PATH="$PWD\ddo-vmm-allocator\build\Release\ddo_vmm.dll"
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py -m vmm -q -p no:cacheprovider --basetemp .pytest-native-transport-public-cuda-context
```

| Existing evidence | Use in this implementation |
| --- | --- |
| Registered ring, 64 MiB windows, 2 MiB reads, up to eight workers, bounded queue | Reuse mechanisms; keep these parameters in first gates. |
| Arena with 32 MiB logical pages and 4 GiB budget | Preserve GPU policy/budget during source work. |
| Four steps: `22,5448 s`, `5,6362 s/it`, no fallback | Recent transport reference; not final promotion yet. |
| Earlier two steps: `11,5994 s`; after progressive change: `11,9267 s` | Short gate has not proven gain; compare A/B without declaring block closed. |
| DMA per 2 MiB chunk: `12,2607 s` in two steps | Rejected experiment; do not reintroduce thousands of small enqueues. |
| Full shard mapping: `93,1694 s`, RAM `27,004 GiB` | Rejected experiment; preserve historical record. |
| Arena reservation `24,03 GiB`, residency `4 GiB`, nonlocal DXGI ~`1,56 GiB` | Size cache by actual unique payload, not virtual reservation/padding. |

NVMe baseline around `4,91–4,95 s/it` belongs to its own historical configuration.
Preserve route/manifest; do not compare directly with new gates under different
conditions.

Before first functional block, record revisions of both repositories, local
diffs, DLL hash, environment versions, and gate manifest.
Tree modified: SHA alone does not identify executed code.
No history cleanup, compatibility removal, or default changes in this block.

## Architecture and responsibilities

```text
DDO: arguments, immutable descriptors, component, report
  -> submodule adapter: bindings, native calls, compute dependencies
    -> native source service (handles, cache, tickets, pressure, metrics)
       hit: persistent registered bytes ------------> DMA -> GPU arena
       admitted miss: file -> registered cache ------> DMA -> GPU arena
       miss without capacity: file -> existing ring -> DMA -> GPU arena
```

Hit bypasses ring. Admitted miss reads directly into final cache buffer,
without copying into ring first. Native service chooses hit, admission, or
bypass. Runner does not know tickets, slots, registrations, internal metrics,
or retention policies. GPU arena eviction does not invalidate host source.

GPU residency, transient staging, and persistent cache limits independent.
Aggregate host memory includes ring + cache + other DDO components, avoiding
granting entire budget to each component.

### 1. Source identity and integrity

- Persistent opening by handle; stable file identity and session generation,
  validated size, offset/length with overflow protection.
  Path and tensor address not sufficient identities.
- On Windows reject incompatible writers/deletion while source open.
  Detect source replacement/change at validity boundaries; modified source
  fails explicitly, without delivering stale weights.
- Register descriptors before discarding loader CPU storages. Aliases and
  overlapping ranges share physical bytes/accounting, with explicit valid-span
  map. Arena padding does not become cache payload.
- Normalize windows from declared ranges; respect EOF, unaligned offsets,
  last partial span. Do not load entire shard for small tensor;
  measure read amplification when present.
- V1 supports immutable dense bytes in execution dtype/layout.
  Weight changes, adapters, casts, layouts require new generation/compatible
  source or already-supported path; never silently reuse incompatible bytes.

### 2. Memory, admission, and pressure

- Segmented buffers up to 64 MiB, with progressive allocation/commit/registration;
  virtual reservation does not mean filling/registering entire model at attach.
  Reuse existing Windows `VirtualAlloc` + CUDA registration mechanism.
- Explicit committed/registered-byte ceiling, including rounding/fragmentation;
  validate before each growth. Query available RAM/commit and configured margins.
  DXGI additional observability, not measure of CUDA registration capacity.
- One native process controller accounts all DDO caches/rings, including
  concurrent components. Other-library memory remains external pressure,
  observed through system limits.
- Small cache must not perform cyclic LRU rereading everything each pass.
  Initial policy: admit under ceiling, retain admitted entries while useful/
  without pressure, bypass remaining misses. No choice by module name,
  transformer prefix/suffix, or Python hook order.
- Under pressure suspend growth, remove only eligible entries without active
  I/O, lease, DMA. Use hysteresis against continuous reregistration;
  reduce effective capacity without raising approved ceiling. Account cause/bytes.
- Allocation/registration failure rolls back partial attempt and uses ring,
  without aggressive per-tensor retries. Ring keeps protected capacity.
- Do not discover capacity by allocating to OOM. Future `auto` bounded by payload/
  observed margins with finite ceiling. Unavailable telemetry requires explicit
  fallback, not unlimited capacity.

### 3. States, concurrency, and events

Entry state: `empty -> filling -> ready -> retiring -> empty`, with `failed`
for invalid fill. Leases/tickets and pending DMAs counted separately:
ready entry admits multiple readers.

- Concurrent same-range requests share fill. Publish `ready` only after all
  required reads succeed and bytes visible to consumers. Short read never a hit.
- Ticket retains source from acquire until DMA enqueue or cancellation.
  Cancellation waits/completes only its I/O before reuse allowed.
- Successful enqueue transfers protection to H2D completion events.
  Every reader stream represented; overwriting one shared event must not hide
  earlier transfer.
- Compute stream waits on upload event. GPU pages stay pinned until compute
  use completes. Host source may release lease after H2D but stay cached for
  next pass.
- Separate GPU residency signature from source identity. Signature hit can
  avoid H2D; source hit only avoids read. Partial/failed copy never publishes
  valid-content signature.
- Check arena validity before source request on demand path.
  Speculative prefetch bounded/accounted: already GPU-valid ranges should not
  cause reads or useless host cache fills.
- Queues/workers/tickets/events bounded. Backpressure waits only on work that
  can release required capacity; batch larger than cache/ring progresses by
  windows, without deadlock.
- Document lock order. Do not wait on I/O/event holding global cache/arena lock
  needed by producer; worker context/device policy explicit and validated on
  stream/destination acceptance.
- No global sync on normal path. Event recording error after enqueue requires
  draining affected stream or retaining protected resources;
  fatal asynchronous CUDA failure ends operation, not silent fallback.

### 4. Fallback and closure

Host capacity failure: bypass via ring. GPU materialization failure:
already planned temporary CUDA tensor, consuming same source/lease when possible
without unnecessary reread. Unrecoverable integrity/I/O error:
propagate cause, do not execute partial bytes.
Temporary fallback also respects pressure and may explicitly fail without memory.

Closure: block new requests; cancel unstarted speculation; complete/join active
I/O; drain session-owned transfer events/GPU uses; restore bindings per existing
contract; release leases; unregister/free buffers; close handles/workers;
release retained context. Explicit `close()` verifiable/idempotent in wrapper,
including incomplete attach or forward exception. Finalizer secondary protection.
Never free resource whose use has not provably ended.

## API contract and integration

Add C API with opaque source/session/cache/ticket handles, configuration/snapshot
with `struct_size`/version. Operations: create, open/declare ranges, request,
query readiness, enqueue copy, release/cancel ticket, trim cache, snapshot, close.
Upload calls must accept arena and validated temporary destination path.
Do not expose raw host pointers to runner.
Status distinguishes recoverable bypass from fatal error/invalid source.

Bridge checks DLL capabilities/symbols. Explicit cache request with incompatible
DLL produces clear diagnostic; disabled cache preserves legacy symbols.
No C++ exception crosses ABI, including `bad_alloc` and worker failures.
Closure must return observable error when incomplete.

| Location | Change responsibility |
| --- | --- |
| `ddo-vmm-allocator/native/include/ddo_vmm.h` | Opaque API, capabilities, configuration, status, versioned snapshot. |
| `ddo-vmm-allocator/native/src/` | Isolated source/cache implementation; reader/events shared with ring without duplicate pools. |
| `ddo-vmm-allocator/CMakeLists.txt` | Compile new files/native contracts. |
| `ddo-vmm-allocator/src/ddo_vmm_allocator/native.py` | ctypes bindings, capability detection, explicit ownership. |
| `ddo-vmm-allocator/src/ddo_vmm_allocator/model_arena_adapter.py` | Use native service; retain binding/compute; eliminate Python cache decisions. |
| `src/diffusers_dynamic_offloader/dynamic_offload.py`, `backend.py` | Same native route in public API/harness; DDO configuration/lifecycle. |
| `src/diffusers_dynamic_offloader/vmm_weight_store.py` | Deliver descriptors; preserve legacy routes, avoid accidental storage retention. |
| `src/diffusers_dynamic_offloader/metrics.py`, `profiles.py` | Per-component capacity report/profile. |
| DDO/submodule tests | Contracts, public integration, failures, cache-disabled regression. |

Public API currently prepares `PinnedStagingPool` in `_prepare_vmm_fast_runtime`
while harness already uses native transport. Unifying consumption at boundary
mandatory for delivery; harness-only success does not close block.

Proposed public configuration: `vmm_source_cache_gb`, corresponding to
`--ddo-vmm-source-cache-gb`, distinct from `vmm_host_cache_gb` registering loader
CPU storages. `0` disables new cache; positive values explicit ceiling;
`auto` promoted only after validation. During development, default disabled.
DDO resolves configuration/margins/capabilities; runner only passes arguments.

## Metrics and profile

Add `runtime.source_cache` to `DynamicOffloadMetrics` with own version, without
renaming existing fields. Storage units bytes; display GiB; time seconds.
Session counters, per-step deltas, separate peaks.
Snapshot does not synchronize GPU; incomplete events pending, never fabricated
zero duration.

| Group | Required fields/measurements |
| --- | --- |
| Capacity | requested/effective limit, source unique payload, reserved/committed/registered/valid/in-flight/peak bytes, ring bytes, admission reason. |
| Reuse | request/hit/miss/bypass bytes/counts, fill bytes, coalesced fills, evictions/reasons, byte-weighted hit ratio. |
| Files | read calls/bytes/failures, amplification, I/O service/wait; physical disk bytes only when measured with appropriate tool. |
| Copies | H2D cache/ring/temporary bytes; host→host bytes; CPU enqueue separate from CUDA event duration. |
| Safety | registration/commit refusals, cancelled tickets, invalid source, fallback causes, live resources after closure. |
| Phases | attach, first load, first step, later steps, prepare+denoise, cleanup, wall-clock total. |
| System | available RAM, commit, working set, local/nonlocal DXGI/budgets; collection context/time. |

Detailed CUDA duration may remain diagnostic-only to limit overhead.
Document each field's unit, scope, collection time, backend, interpretation,
diagnostic action in `docs/metrics.md`. VMM profile records capabilities/admission
reasons without GPU residency calibration. Text encoder uses same source mechanism/
aggregate control after transformer approval.

## Implementation order and gates

Do not begin next block before compiling, validating contracts, recording previous
result. Every CUDA/benchmark execution requires exact command, hypothesis, limits,
approval criterion, and user confirmation.
Use DDO `.venv-vmm` exclusively. LTX remains in sibling host.

1. **Close starting state/specification.** Validate window transport restoration,
   freeze manifest, resolve pending A/B before default policy promotion.
   Update obsolete submodule guidance (LTX environment, old baseline/cache scope)
   when implementation starts.
2. **Source/ownership.** Implement identity/descriptors, ABI, lifecycle;
   reuse existing reader. No-GPU tests for limits, aliases, file modification,
   short reads, full queue, cancellation, close.
3. **Registered cache/DMA.** Segmented memory, admission/pressure, tickets,
   events, bypass, failures. First small CUDA test with at most 128 MiB cache
   and synthetic files; no full model/exploratory memory increase.
   Byte comparison, hit without new read/host→host, two streams, in-flight closure,
   partial failure rollback mandatory.
4. **Public integration/reports.** Connect DDO/adapter, single configuration,
   profile/JSON, temporary fallback. Small dense-model CUDA test with aliases,
   multiple units, repeated steps, insufficient capacity;
   compare numerical result/counters. Test disabled cache and confirm stable/
   protected-route regression before heavy harness.
5. **Transformer diagnostic.** 2/4-step gates, fake prompt, `1280x704`,
   seed `43`, BF16, one generation, 4 GiB arena, 512 MiB total staging.
   Host cache new explicitly recorded variable. First propose 512 MiB extra
   to validate integration/limits; does not test full-coverage hypothesis.
   Only after success propose actual-payload/measured-headroom ceiling,
   with specific confirmation before high RAM usage.
6. **Performance/delivery.** Cache-off/on A/B with same code, DLL, environment,
   OS-cache policy, load. Compare cold/warm costs, including first load and
   prepare+denoise. Final real-prompt/8-step benchmark; then documentation,
   defaults/preset, submodule pointer review. Expand text encoder later
   with own gates.

Minimum pressure contract includes: capacity below one unit, all entries occupied,
pass larger than cache, registration/read/enqueue/event failure, close during
I/O/DMA, two concurrent components. Failure tests controlled injection;
never exhaust global RAM/VRAM to validate fallback.
Handle/registration/lease/worker counters must return to initial state after closure.

For synthetic full-coverage gate, after one complete pass and forced eviction
of GPU arena only, second pass must have zero file reads and zero host→host
copies for admitted ranges, identical data, H2D protected by correct events.
Partial cache must demonstrate retained hits and bounded bypass without
continually reloading every entry.

For performance, require correctness, no unexpected failure, respected limits
before evaluating speed. Denoise gain merely shifting cost to prepare not enough.
If reads disappear but time remains high, observe waits, actual DMA, compute
to locate limit; do not increase memory by assumption.
Repeat measurements when small differences/variance make results inconclusive,
without selecting only best run.

Historical targets (`prepare` ~`0,16 s`, steady-state `2,80–3,13 s/it`, sampler
~`49,66 s`) and later target around `1,3 s/it` must retain manifests/units.
`1,3 it/s` corresponds to ~`0,769 s/it`: do not treat as equivalent.
Confirm reference log unit before final comparison; record per-phase times,
including separate text encoder.

Deliverable results go to `experiments/dynamic_offload_results.md`, with command,
manifest, effective configuration, decision reason. Mark old texts historical
without erasing negative experiments. Update README, both repositories' AGENTS,
catalog, minimal snippets according to actually implemented capabilities.
Windows first; Linux later; no WSL/macOS VMM promise.
Preserve required license notices; history audit/publication separate work
requiring own confirmation.

## Initial gate executed and next regression

Hypothesis: restored transport preserves positioned reads and window DMA with
correct slot protection. Approval: three selected contracts pass without crash/
CUDA failure. No capacity change or LTX execution.
Result: **3 passed**, before source changes; nonfatal pytest-cache permission
warning.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator
& ..\.venv-vmm\Scripts\python.exe -m pytest tests\test_native_ranges.py -m vmm -k 'ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks' --basetemp ..\.pytest-vmm-window-restore
```

Post-build CUDA regression executed after confirmation: same Release DLL with
source API added, three contracts above, using `-p no:cacheprovider` and
`--basetemp ../.pytest-vmm-source-foundation`. Result: **3 passed, 11 deselected
in 5.33s**, no CUDA failure/warnings. Criterion passed; validates selected
transport contracts, not performance or future registered cache.

Commands for new cache contracts will be presented after implementation and
compilation. They are not advance permission for later gates.

## Integrated file-window preparation (2026-09-29)

Source-cache ABI 7 adds opt-in configuration v4. Native admission now shares
aligned file windows between neighboring tensor ranges; the last window is
clamped to EOF. A range crossing a window boundary is explicitly bypassed;
the adapter splits tensor descriptors first. Legacy v1-v3 requests keep their
exact-range behavior. No loader CPU tensors become cache owners.

`vmm_source_cache_gb` / `--ddo-vmm-source-cache-gb` selects this configuration.
The 32 GiB cache and 40 GiB aggregate ceilings are configuration bounds, not
approved allocations or defaults. Registration remains lazy and guarded by
native RAM/commit checks, the aggregate staging/cache ledger, and ticket
lifetimes. Large-capacity execution still requires its own explicit approval.

The adapter submits at most 64 source descriptors before issuing their DMAs.
This permits native reads to overlap rather than request/wait serialization;
it does not establish successful inter-unit look-ahead or measured speedup.
`source_read_bytes` counts whole windows, including bytes not requested by a
tensor; `dma_bytes` counts requested tensor ranges. Account for this read
amplification when comparing useful coverage and cold preparation.

Release build succeeded; grouped CPU contracts passed (100 passed, 7 CUDA
contracts deselected). The CPU suite now mocks device capacity so planning
tests cannot silently initialize CUDA. The approved consolidated CUDA repeat
passed (21 passed in 4.98 s) with the same Release DLL. The first attempt
stopped at a test-accounting expectation (a bypass also counts as a miss);
only the assertion changed. Full harness parity, larger capacity gates and
performance remain pending.

## Public harness gate (2026-09-30)

### Full-unit GPU look-ahead correction (implemented; CUDA approval required)

Native transport now faults and enqueues the next whole execution unit after
the current forward has enqueued compute. Native admission determines capacity;
speculation never waits for compute to free capacity and never creates a
temporary fallback. Successfully prefetched pins transfer to the consuming hook,
which waits on upload dependencies without repeating fault or H2D. Source-only
look-ahead remains a diagnostic contract, not the native transport execution path.

Skipped/reordered units, pass boundaries and close cancel abandoned look-ahead
only after draining its owned upload stream. A failed later upload batch drains
that stream before invalidating signatures and releasing destination pins. If
draining itself fails, pins remain protected and the error propagates. Forward
hooks run cleanup on exceptions, without unpinning units whose pre-hook failed.

No C++/ABI, GPU/host budgets, source-cache admission, runner arguments or stable
runtime routes changed. CPU regression: **90 passed, 15 deselected in 6.51 s**;
new test-file lint passed. The approved CUDA sequence then passed **32 tests,
5 deselected in 5.98 s**. Both 4-step LTX cases exited successfully with identical
images, 188 GPU-prefetched/consumed units and no capacity skips, cancellations,
fault/read failures, temporary fallbacks or execution-order mismatches.
Performance did not improve: denoise **34.4810 s / 39.2374 s** (cache off/on),
versus the preceding **31.8246 s / 35.9451 s**. H2D remained 93,927,995,136 bytes
for either case; the 512 MiB cache still supplied only 497,850,384 DMA bytes.
Functional validation passed; the performance gate remains unresolved.
This closes an implementation gap, not the performance target or source coverage.

One proposed validation sequence: small CUDA contracts, then the existing
4-step cache-off/on pair, stopping at the first failure or memory anomaly.
Keep 4 GiB active arena, 1.5 GiB total staging and 0/512 MiB source cache.
Confirm the next-unit H2D precedes its consumer, no duplicate fault/upload,
correct numerical output and safe skipped-unit cancellation. Only then evaluate
prepare+denoise and per-step times; no speedup claim based on contracts alone.

Commands approved and executed (each LTX case launched separately after checking
the previous result; no additional run or parameter change):

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH="$PWD\ddo-vmm-allocator\src;$PWD\src"
$env:DDO_VMM_LIBRARY_PATH="$PWD\ddo-vmm-allocator\build\Release\ddo_vmm.dll"
& .\.venv-vmm\Scripts\python.exe -m pytest ddo-vmm-allocator/tests/test_source_cache.py ddo-vmm-allocator/tests/test_module_adapter.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py -m vmm -q -x -p no:cacheprovider --basetemp .pytest-gpu-lookahead-cuda
if ($LASTEXITCODE -ne 0) { throw 'CUDA gate failed; do not run LTX' }

Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
foreach ($cacheGiB in @(0, 0.5)) {
    & 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe' run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb $cacheGiB --width 1280 --height 704 --seed 43 --steps 4 --fake-prompt --no-pag --local-files-only --metrics-level 2 --show-ddo-profile
    if ($LASTEXITCODE -ne 0) { throw 'LTX case failed; stop sequence' }
    # Inspect the resulting metrics and memory before starting the next case.
}
```

Previous case JSON/images were preserved under the sibling output directory
`before_gpu_lookahead_20260930` before executing. Do not run the next LTX case
automatically if the first reports faults, temporary fallback or memory anomaly.
No increased cache capacity, real prompt, 8-step run or new budget is included.

### Cache-aware look-ahead follow-up (validated; performance unresolved)

Subsequent preparation: C++ read-only arena signature observation added under
the arena mutation lock. It does not call CUDA, materialize, pin, evict or
update counters. The adapter can avoid speculative reads whose previously
uploaded signature is still native-resident; demand fault/recheck remains
mandatory, preserving recovery after an intervening eviction. Old DLLs return
an unknown observation and preserve existing speculation. New metric:
`fast_prefetch_resident_skips` (tensor candidates, not units).
Release rebuild succeeded; **104 CPU tests passed in 5.15 s**. New DLL SHA256:
`8029EF09D9D0F83F8B3D5ADF4E766FC3131BAAE105BD6B6BDF457A69DA520682`.
The approved CUDA gate passed: **23 tests in 4.92 s**, including the
no-side-effect/eviction/generation contract, using the DDO `.venv-vmm`.
This does not change source-cache admission or solve useful retained coverage.

Approved CUDA battery: **22 passed in 5.56 s**. Four-step A/B with unchanged
budgets completed correctly and produced matching images. Cache-on prefetch:
346 hits/0 skips, equal to cache-off. Denoise: 35.9451 s cache-on versus
31.8246 s cache-off; performance target not achieved. Partial cache hits and
DMA bytes remain unchanged, and full H2D payload is still 93,927,995,136 bytes.
Scheduling is now functional, but useful retained coverage and unnecessary
speculation remain unresolved. No larger capacity or 8-step gate was run.
Detailed numbers, variability caveats and saved prior artifacts are recorded
in `experiments/dynamic_offload_results.md`. The preparation wording below
is historical and superseded by these execution results.

The native adapter now allows inter-unit look-ahead with cache enabled. A
bounded speculative ticket first asks the C++ cache for each file window;
only native bypasses enqueue ring reads. Unsubmitted cache tickets remain
separate from DMA polling. Consumption transfers ownership to per-DMA
tickets and returns a final upload-stream event covering cache and ring.
Cancellation closes source leases while the native cache owns in-flight I/O;
stale ring tickets cause cancellation rather than unsafe reuse. Cleanup also
drains unused speculation. Partial source/enqueue failures retain ownership
until safe closure; no global synchronization is added to normal operation.

Grouped CPU suite: **103 passed, 7 deselected in 6.03 s**. The mixed cache/ring
CUDA consumption/cancellation contract is prepared but unexecuted. No native
C++ or DLL change, capacity change or residency heuristic was introduced.
This addresses scheduling; useful native admission and full coverage remain
separate unresolved work. Hardware correctness and a new 4-step cache-off/on
comparison require confirmation before execution.

Approved execution: capacity contract passed in 7.72 s, followed by all four
2/4-step cache-off/on cases. Output PNG hashes match within each step count;
no fault/read failures or temporary fallbacks occurred. Four-step denoise:
27.6858 s cache-off versus 33.2483 s cache-on. Cache-on skips all 188 inter-unit
prefetch attempts. Its 23 cache hits and 497,850,384 cache DMA bytes are
unchanged between 2 and 4 steps; useful additional reuse has not been shown.
Close cache-aware speculative ticket lifetimes and investigate native
admission usefulness before increasing memory. Results and manifests are in
`experiments/dynamic_offload_results.md`. The preparation text below records
the pre-execution proposal; its pending-run wording is historical.

The sibling LTX harness now offers `--ddo-public-native-runtime` and
`--ddo-vmm-source-cache-gb`. Its normal public `enable_offload`, report,
profile and cleanup path owns execution; the diagnostic adapter remains
available and is mutually exclusive. Bootstrap and local remote-module
cache remain unchanged. The host staging argument retains its diagnostic
per-slot meaning: 0.5 GiB is mapped to DDO's TOTAL 1.5 GiB (24 x 64 MiB).
The host passes its module pattern, 32 MiB pages and DXGI reserves to DDO.
Output names distinguish the public cache-off and cache-on configurations.

CPU configuration gates passed: 31 tests in 4.54 s, including the actual
harness settings block extracted without importing its CUDA bootstrap.
This validates configuration parity, not equivalent numerical/performance
behavior on the full transformer. LTX loaders and local module cache remain
in the sibling repository; the harness is not part of the generic DDO API.

Prepared capacity contract: `test_model_windows_1536_mib_ring_plus_512_mib_cache`
registers at most 2 GiB host bytes, with a separate 512 MiB test file and
4 GiB native RAM/commit headroom. It checks progressive 512 MiB retention,
second-pass hits without reread, aggregate ceiling rejection and zero ledger
after closing. It is opt-in via `DDO_VMM_CAPACITY_GATE=1`; it has not run.
The public runtime still uses the cache's existing native headroom default;
DXGI system reserves govern arena mappings, not host-cache admission.

Following approval and capacity success, compare public cache-off/on with
the same 4 GiB active budget, 1.5 GiB total staging, 1280x704, seed 43, fake
prompt, no PAG and 2/4 steps. Stop on failure or memory/metric anomalies.
512 MiB is partial retention, not full-model coverage and not a promise of
the final throughput target. Large coverage and real-prompt 8-step gates
require separate approval based on the observed headroom and results.
