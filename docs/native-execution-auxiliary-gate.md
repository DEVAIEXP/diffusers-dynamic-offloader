# Native execution-unit auxiliary ownership candidate

## Scope and status — 2026-10-06

The bounded operator diagnostic identified CPU auxiliary parameters outside the
dense-Linear attachment contract. This opt-in candidate adds their immutable
sources and native views to existing execution units. It does not add a Python
weight cache or change native pressure, transfer, eviction or budget policy.

Status: the narrowed corrective candidate passed **CPU and small Windows CUDA
contracts; real-model validation remains pending**.
The earlier broad candidate passed small CUDA contracts but its real-video
diagnostic was interrupted after a major timing regression. Neither candidate
is accepted for promotion; real-model performance and image regression remain
unproven. Preserve the default-off switch and accepted default route.

- Public switch: `--ddo-vmm-execution-auxiliary-ownership`, default off.
- Configuration: `vmm_execution_auxiliary_ownership=True`, fast VMM only.
- Component-wide ownership and execution auxiliary ownership are mutually exclusive.
- Selection is architecture-independent: direct non-Linear parameters within
  existing selected ModuleList execution units, excluding parameters within the
  existing small-tensor placement threshold. Buffers and outside parameters
  retain their previous route; this is not arbitrary execution-graph support.
- Auxiliary bytes increase the arena payload and aligned execution-slot geometry,
  **not the GPU ceiling**. Native immutable batches and the existing two-slot
  pool handle storage, DMA and final-consumer retirement.
- Exact Parameter aliases inside one unit publish a single shared Parameter.
  Cross-unit/external/Linear aliases, distinct overlapping storage objects,
  unmaterialized or noncontiguous inputs and missing immutable sources refuse
  attachment before native allocation or module mutation.
- Dense prefetch order remains intact. The existing component forward boundary
  supplies reentry/output-view guards and cooperative BaseException cleanup.
- Published auxiliary loader references are dropped. Shutdown waits for proven
  completion, then invalidates auxiliary aliases with shared meta placeholders.
  It does not reconstruct CPU weights or promise reuse after unloading.
- The modular video host explicitly resets the new flag for native text-encoder
  configuration, preserving the accepted component-wide encoder route.

No C++ source or ABI was changed for this candidate. The existing native buffer
and source-batch API is reused; no DLL rebuild is required. The selected diagnostic
DLL remains:

`ddo-vmm-allocator/build-allocation-retry-diagnostics-20261006/Release/ddo_vmm.dll`

SHA256: `335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

## CPU evidence

DDO `.venv-vmm`: **110 passed, 2 subtests passed**, selecting:

```text
tests/test_execution_auxiliary_cpu.py
tests/test_vmm_component_adapter_cpu.py
tests/test_vmm_tensor_owners_cpu.py
tests/test_vmm_execution_buffer.py
tests/test_auxiliary_sources.py
tests/test_backend.py
tests/test_dynamic_offload_presets.py
```

The new tests cover aligned unique-alias geometry, early refusals, publication,
dense queue preservation, discarded loader references, completion failure/retry,
alias retirement and unchanged/default-off protected route contracts. Native
views and events are CPU doubles, not evidence of CUDA transfers.

Host `.venv-vmm` in `E:/LTX-2`: **53 passed, 31 subtests passed** across
`test_ddo_video_host_cpu.py`, `test_ltx2_video_blocks_cpu.py` and
`test_ltx2_video_modular_runner_cpu.py`. An initial attempt using DDO's environment
failed collection because that environment lacks `torchaudio`; it was not an
executed model failure. No dependencies were installed to bypass that distinction.

## Small CUDA gate — approved and passed

Hypothesis: auxiliary BF16 parameters use immutable native sources and shared
device views, retain numerical results, reuse source bytes across repeated
forwards and retire safely after ordinary errors and cooperative interruption.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-allocation-retry-diagnostics-20261006\Release\ddo_vmm.dll'
$taskBase = Join-Path (Get-Location) ('.internal\auxiliary-cuda-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskBase | Out-Null
& .\.venv-vmm\Scripts\python.exe -B -m pytest tests/test_execution_auxiliary_cuda.py -q -x -p no:cacheprovider --basetemp $taskBase --tb=short
```

Approval: both named parameterized cases pass; BF16 parity tolerance 0.016,
six immutable file bindings and zero retained CPU source bindings; repeated
passes do not increase source read bytes; zero fault failures, temporary fallback,
source admission denial/bypass, pending completion tickets or transfer-owned
events; alias identity preserved during execution and after meta retirement;
source registry and forward boundaries cleanly released.

Actual result, 2026-10-06: **2 passed in 5.26 s**, process exit 0. Both
`RuntimeError` and `KeyboardInterrupt` cases completed all assertions above.
The DLL hash was rechecked and matches the selected hash recorded above.
No real-model benchmark was run under this approval; no C++ rebuild, budget
increase, dependency replacement or default-route promotion occurred.

The tiny gate uses a 0.07 GiB native ceiling, 2 MiB transport/page geometry and
64 MiB source storage/registration, isolated from real-model benchmark settings.
It is a lifecycle gate, not real-model capacity, performance or full allocator
bootstrap evidence. Stop on the first failure or memory anomaly. Real-model
execution, including an image regression case, requires subsequent explicit
approval and must preserve the selected benchmark ceilings and parameters.

## Real-video diagnostic — approved, interrupted, not accepted

Repeat the last bounded 8-second, 4-step latent diagnostic with the same model,
BF16, dimensions, seed, source/registration settings, encoder route, chunking
and diagnostic capture. The only workload control added is the explicit
transformer auxiliary-ownership switch. Do not decode or encode an MP4.

```powershell
Set-Location E:\LTX-2
.\run_ddo_vmm_video_modular.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-allocation-retry-diagnostics-20261006\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 8 --steps 4 --seed 43 --no-use-audio-condition --output-type latent --diagnose-denoise --diagnose-operators --diagnose-operator-module transformer_blocks.0 --ddo-vmm-resident-weight-budget-gb 2 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources --ddo-vmm-execution-auxiliary-ownership
```

Hypothesis: the six identified pageable auxiliary H2D copies disappear from
the sampled unit forward because the tables are supplied by its native batch,
without raising the existing 2 GiB transformer ceiling or changing attention.

Acceptance: successful generation and finite CPU-readable packed latents;
both selected CUDA operator captures complete; ownership report accounts for
auxiliary groups/bytes; the six identified pageable table transfers disappear;
no final fault failure, temporary fallback, source admission denial, ownership
mismatch/quarantine or pending-event/cleanup anomaly. Stop on the first failure
or memory anomaly; do not increase ceilings or extend steps/duration.

Record all per-step/phase times and absolute versus baseline-relative memory
observations. Profiler overhead and run variability prevent treating this single
diagnostic as a matched speedup benchmark. If it passes, image regression and
unprofiled performance verification still need a separately approved scope.

### Actual evidence — 2026-10-06

Output directory: `E:/LTX-2/outputs/ddo_modular_video_20261006_155602_695`.
Console: `E:/LTX-2/outputs/auxiliary_ownership_console_20261006_155602_636.log`.
The only completed operator capture is `operator_traces/operators_invocation_00.json`.
The second capture started but was not exported. No final metrics JSON or packed
latent artifact exists; final counters and numerical output are **unavailable**,
not zero or passing.

The user explicitly chose to interrupt and investigate. The owned Python
process 39028 was stopped after verifying its start time (15:56:02). The wrapper
reported child exit -1; the enclosing command ended with exit 1. This is a
user-requested process termination, not a logged CUDA exception or proof that
normal cancellation/cleanup passed. Post-termination driver sample: 3% GPU
utilization, 385 MiB used of 8192 MiB. Earlier active sample: 99%, 7839 MiB used.
These are sampled absolute observations, not allocator peaks or residency proof.

Partial timing:

- Attachment setup: 14.1365 s (previous diagnostic 6.7469 s).
- First progress step: 473.24 s (previous CUDA-event step 160.689953125 s).
- Two-step progress elapsed: 11:11; second step approximately 198 s, inferred
  from rounded console elapsed, not a recorded CUDA-event duration.
- The progress-bar 311.71 s/it is its estimate, not a four-step average.
- Working set sampled around 29.6–29.7 GiB; no final RAM/GPU peak report exists.
- Post-attachment source observation reported CPU storage bytes 0 and the old
  CPU detachment helper reported no source-backed CPU tensors to detach.

First-capture comparison with `ddo_modular_video_20261006_150751_571`:

| Observation | Previous | Auxiliary candidate |
| --- | ---: | ---: |
| Pageable H2D copies | 6 | 0 |
| `cudaStreamSynchronize` events | 6 | 0 |
| Kernel events | 897 | 897 |
| Sum of kernel durations | 2.08166727 s | 1.905582029 s |
| All H2D events | 85 | 195 |
| Captured H2D bytes | 773300608 | 1546781440 |

The six identified pageable transfers disappeared, but that does not establish
net speedup. Kernel-duration sums may overlap and describe one sampled unit only.
The new BaseException-safe forward wrapper changes profiler scope: its native
preamble is now inside the profiler pre/post hooks. In the candidate capture,
98 H2D events occur before the first direct-child forward range. The increased
captured byte total is therefore not a matched measurement of whole-step DMA
or evidence that the model reread twice as many bytes.

### Read-only investigation and next boundary

The implementation also expands ownership to *all* direct non-Linear parameters
inside selected units, including small parameters previously handled by the
existing auxiliary placement route. This changes immutable batch descriptors,
aligned source/view spans and slot geometry beyond the six large table transfers.
The small gate proves aliases/lifetimes, not equivalence of those real-model
source windows or pressure behavior. These are concrete changed surfaces to
inspect, not a proven cause of the slowdown.

The captured preamble has 98 pinned transfers, including the complete table
payloads 73728, 36864, 40960 and 20480 bytes. It also contains six 64-byte,
twenty-four 4096-byte and fifteen 8192-byte transfers. These observations show
that the batch handles more than the four complete residual tables; they do
not by themselves identify source rereads or explain the timing regression.
The existing persistent path still calls the native immutable batch submit and
native execution-pool acquire/compute fences. No alternate PyTorch weight cache
was introduced. Larger raw capture counts include both the newly captured
preamble and following-unit prefetch.

Because final native counters were not exported, source rereads, eviction,
pressure response and final admission cannot be inferred from this run. Do not
attribute the regression to attention, driver paging or source cache alone.
Preserve existing budgets and the accepted route. Before another model execution,
review batch/slot geometry against the previous dense contract and ensure any
next approved diagnostic preserves partial native counters at step boundaries.
No corrective patch or additional benchmark was executed under the interruption
request. A new CUDA execution requires separate confirmation.

## Read-only geometry audit after interruption

The user agreed to continue investigation. A CPU/meta-model inspection read
only the local config and safetensors headers, not payloads. CUDA lazy
initialization was forbidden; the script reported `cuda_initialized=false`.
The local-only script is `.internal/scripts/inspect_auxiliary_geometry.py`.
Header parameter count is 4186, matching the previously recorded source mapping.

| Geometry | Original dense route | Broad auxiliary candidate |
| --- | ---: | ---: |
| Largest aligned unit payload | 773131264 bytes | 773393408 bytes |
| Two slots rounded to 2 MiB pages | 1547698176 bytes | 1547698176 bytes |
| Immutable source spans, first block | 68 | 86 |
| Source windows, first block (64 MiB) | 13 | 13 |
| Units touching additional source windows | — | 0 |

The 864 added groups contain 12582912 bytes (12 MiB). Of these, 672 groups,
4325376 bytes (4.125 MiB), fall within the unchanged 16 KiB small-parameter
contract. The remaining 192 groups, 8257536 bytes (7.875 MiB), are the four
complete residual tables per block. The six 64-byte *transfers* in the capture
are not evidence of six 64-byte Parameters: no added Parameter has that size
in the headers. Transfer fragmentation and Parameter sizes are distinct.

This rules out enlarged rounded slot capacity and newly touched source windows
as the simple explanations proposed for the regression. It does not measure
runtime cache misses, queue pressure, DMA ownership or driver paging. The source
batch preserves individual spans; more descriptors can mean more requests/copy
events within the same windows. The native execution pool waits on compute via
the upload stream, and the allocation ledger supports pending-span reuse with
stream dependencies; those mechanisms remain present. Their existence alone
does not prove absence of pressure under the new asynchronous cadence.

Proposed corrective scope, not yet implemented or GPU-approved:

1. Include only residual auxiliary parameters outside the **existing** small
   placement contract. Leave the 672 already-working small parameters alone;
   do not raise the threshold, introduce a Python cache or change native budgets.
2. Preserve native event ownership and cooperative-interruption guards. Do not
   add global synchronization to reproduce the old accidental pageable waits.
3. Persist detached public native counter snapshots at step boundaries so an
   interrupted diagnostic retains pressure/source/transfer evidence. Avoid extra
   completion collection or CUDA waits in that observation.
4. Validate selection and geometry in CPU first; propose the smallest necessary
   CUDA gate separately. No further long real-model trial until that review.

These changes narrow an unnecessary behavioral expansion and improve missing
evidence; they are not a proven performance fix or permission to publish/promote
the candidate. The large regression's complete cause remains unestablished.

## Narrowed corrective candidate — CPU and revised small CUDA gate passed

The user approved implementation and CPU validation only. Discovery and native
attachment now receive the same `config.small_tensor_threshold_bytes` used by
the existing DDO auxiliary route. Parameter bytes **less than or equal to** that
threshold are excluded from the new ownership path, without changing their
placement contract or increasing the threshold. The submodule standalone
default is also 16 KiB; it does not introduce another runner budget control.
Native aliases, immutable source ownership, interruption guards and scheduling
remain unchanged. The broad candidate's earlier CUDA pass is historical evidence,
not validation of this revised selector.

CPU contracts: **120 passed, 2 subtests passed**, including exact threshold
equality, preservation of small Parameter identities, invalid threshold refusal,
partial-report detachment, no completion collection, collision refusal and
release of the result/model reference after the journal scope. Host contracts:
**53 passed, 31 subtests passed**, using `E:/LTX-2/.venv-vmm`.

Repeated CPU/meta header inspection reports `cuda_initialized=false`. The
corrected first block has 72 source spans (68 dense + four residual), not 86.
Its aligned payload is 773303296 bytes. Rounded two-slot capacity is still
1547698176 bytes. The expected workload includes 192 residual groups (7.875 MiB)
while preserving 672 small groups (4.125 MiB). This is geometry evidence, not
runtime admission or performance evidence.

DDO now provides `result.metrics_journal(directory, enabled=False)` as a public
opt-in factory. The modular runner uses it only under `--diagnose-denoise`:
one `before_denoise.json`, then `step_0000.json` etc in
`metrics/denoise_snapshots/`. Each snapshot uses the public metrics report with
`collect_completed=False`; no CUDA timing resolution or synchronization is added.
Files are individually published without replacing existing evidence. CPU
flush/fsync/publication has diagnostic I/O overhead; filesystem errors propagate.
On abrupt termination the current temporary file may remain, but earlier
published snapshots remain readable. Steps are host callbacks after scheduler
submission, **not proof of GPU completion**. Snapshots do not replace normal
final metrics or numerical/safety acceptance.

No C++ or DLL changes were made. DLL hash remains
`335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.
At corrective preparation, no CUDA test or model run had been executed. The
subsequent approved revised small CUDA result is recorded below.

### Revised small CUDA gate — approved and passed

Use the command in the small CUDA gate above, selecting
`tests/test_execution_auxiliary_cuda.py` only. The fixture now uses the production
16 KiB threshold, retaining its 32 KiB shared BF16 table in native ownership and
adding one 256-byte non-Linear parameter per block through the existing small
placement route. Budgets, four repeats and the two interruption cases remain
the same. Approval additionally requires preserved small Parameter identity and
CUDA placement, alongside numerical parity, six file-backed native bindings,
source reuse and successful native alias retirement. Stop at the first failure.
This is a small lifecycle gate, not permission for another long video benchmark.

Actual result, 2026-10-06: **2 passed in 5.20 s**, process exit 0, using
DDO's `.venv-vmm`. Both `RuntimeError` and `KeyboardInterrupt` parameterized
cases passed with the production 16 KiB threshold. Small Parameter identity and
CUDA placement were preserved, the six native immutable bindings retained no
CPU source tensors, repeated passes did not increase source read bytes, and
all asserted fault/fallback/admission/bypass/pending/transfer counters and
alias/source/hook cleanup checks passed. The unchanged DLL hash was rechecked.

This accepts the revised **small contract only**. The previous broad real-video
regression is not erased, image regression is untested and the metrics journal
has CPU evidence only; the fixture did not exercise journal publication under
CUDA. No real-video benchmark or model execution was run under this approval.

### Revised residual-only video diagnostic — interrupted, not accepted

The user subsequently approved this bounded real-model case:

```powershell
Set-Location E:\LTX-2
.\run_ddo_vmm_video_modular.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-allocation-retry-diagnostics-20261006\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 8 --steps 2 --seed 43 --no-use-audio-condition --output-type latent --diagnose-denoise --ddo-vmm-resident-weight-budget-gb 2 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources --ddo-vmm-execution-auxiliary-ownership
```

Hypothesis: residual-only ownership avoids the broad candidate's regression
without changing budgets. Acceptance required finite packed latents, durable
partial snapshots and successful final fault/fallback/ownership/cleanup checks.
Operator profiling was omitted; this is not a matched performance comparison
with the earlier four-step profiled diagnostic. The DLL hash was unchanged.

Actual evidence, 2026-10-06:

- Output: `E:/LTX-2/outputs/ddo_modular_video_20261006_165752_982`.
- Console: `E:/LTX-2/outputs/residual_auxiliary_console_20261006_165752_896.log`.
- Transformer attachment: **8.3725 s**. Partial report confirms 192 groups,
  8257536 bytes and the preserved 16384-byte threshold.
- Before-denoise snapshot published at 16:58:35; first callback snapshot at
  17:01:33. Its host elapsed value is **177.9379351 s**; buffered progress later
  reported **177.87 s/it** for the first step. These are not resolved GPU timings.
- First partial snapshot reports zero fault/admission failures, pool quarantine,
  source-read failures and cache host-copy bytes. Pending pool bytes are nonzero;
  no completion collection was requested. These observations do not establish
  final safety or cleanup acceptance.
- No second callback arrived. At the user's explicit stop request, the owned
  Python process was terminated at **17:14:21**, approximately **768 s** after
  the first snapshot. This interval is a host lower bound, not a completed
  second-step CUDA timing. The user initially chose to continue after the delay
  warning, then requested interruption.
- Child exit -1 / wrapper exit 1 reflect forced termination, not a captured CUDA
  exception. Only the two published partial JSON snapshots remain; no final
  metrics, packed latents or normal cleanup acceptance is available. Post-stop
  sampled GPU usage returned to 3%, 379 MiB.

The candidate is **not accepted for promotion**. Restricting ownership to the
192 residual groups did not eliminate the real-model slowdown. This also means
the broad migration of small parameters alone is not a sufficient explanation.
Do not attribute the stall to disk, cache or driver behavior from these partial
snapshots alone. Preserve the accepted default route and previous binaries;
investigate the remaining ownership/scheduling change before proposing another
CUDA run. No automatic budget increase or follow-up benchmark was performed.

### Post-interruption read-only investigation

The user approved investigation, not another CUDA execution. CPU/meta header
inspection again refused CUDA initialization (`cuda_initialized=false`). At
the snapshot's effective arena fault ceiling of 599785472 bytes, comparing
original dense offsets with appended residual offsets reports **zero changed
unit route decisions**: the same 64 of 76 units already required execution
slots. Appended auxiliary placement is therefore not an explanation for a
whole-unit route change in this selected workload. Maximum rounded slot size
and touched file windows also remain unchanged, as previously recorded.

A concrete remaining native limitation is pending-span fragmentation:

- `SpanLedger::coalesce()` merges pending neighbors only when their dependency
  lists are identical. `ddo_vmm_pool_free()` creates a distinct completion fence
  for each free, so adjacent pending spans commonly retain different fences.
- The allocator can reuse one sufficiently large pending span with stream
  dependencies. It cannot assemble a larger request from adjacent idle spans
  carrying different fences. After a nonblocking reap fails to yield a fit,
  allocation can grow another physical backing.
- The existing CPU contract explicitly requires a 3072-byte request to fail in
  the illustrated fragmented 4096-byte backing until completion observations
  permit coalescing. This is a documented conservative restriction, not evidence
  of unsafe early reuse or absence of asynchronous allocation support.
- Removing blocking CPU table copies removes their implicit stream waits. That
  can leave more distinct pending fences during subsequent allocations. It is
  a plausible connection to this candidate, **not a proven runtime cause**.

The first callback's detached snapshot has 1046478848 mapped pool bytes,
637726720 leased bytes, 307721728 available bytes, 101030400 pending bytes and
zero quarantined bytes. It does not observe the later stall, does not report
the largest reusable contiguous span, and cannot establish backing growth
during the second step. No such missing observation is treated as zero.

Before changing this restriction, require a CPU contract for transactional
multi-span reservation with every predecessor fence retained, no live/reserved/
quarantined spans included, failure rollback and byte conservation. Any later
native change must retain stream dependency and destruction safety, use a
separate candidate binary and receive an approved small CUDA gate before a
real model. Do not remove existing safety tests, add global synchronization,
increase budgets or promote this hypothesis as the confirmed video diagnosis.

No production code or DLL was changed in this read-only investigation, and no
additional CUDA test or benchmark was executed.

### Pending-span CPU reproduction — confirmed limitation

The user approved a CPU-only reproduction, without changing production code.
Local ignored source: `.internal/scripts/pending_span_fragmentation.cpp`.
Build artifacts: `.internal/pending-span-cpu-a8142c313b7e457282dfd464d5a057e6`.
Compile with MSVC 19.44 x64, `/std:c++20 /EHsc /W4`, including only
`ddo-vmm-allocator/native/include`. The program links no CUDA libraries and
performs no CUDA calls, allocations, waits or physical-backing growth.

Actual result: **PASS, 61 checks**, exit 0. The unchanged production ledger
refuses a 2048-byte reservation in a 4096-byte backing with no live leases and
four adjacent 1024-byte spans retired behind distinct pending fences. Individual
512-byte pending reuse preserves its dependency. Observing two neighbors
complete produces a 2048-byte fit; observing all four recovers a 4096-byte fit.
The shared-fence control can reserve all 4096 pending bytes with that dependency
retained. Live/quarantined barriers, rollback, destruction refusal and byte
conservation checks pass.

This confirms the ledger's conservative fragmentation restriction, **not**
backing growth in the interrupted video or its causal contribution. No safe
multi-span implementation was exercised. The original CPU ledger contract was
also compiled and executed separately: **PASS, 1292 checks**, exit 0; it remains
unchanged. The candidate DLL
SHA256 remains `335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

The proposed next implementation boundary is a separately selected multi-span
reservation operation, preserving the legacy reserve/fit contract. It must
retain the union of all predecessor fences, transactionally reserve only
contiguous idle spans within one backing, exclude live/reserved/quarantined
spans, conserve bytes on splitting and rollback, and preserve quarantine after
partial dependency submission. Selection and reservation must use the same
fit geometry. This is not authorization for a CUDA test or model run.

### Opt-in joined reservation implementation — CPU accepted, CUDA pending

Implemented `SpanLedger::reserve_joined` / `fit_size_joined` as separate
operations. Legacy single-span reserve/fit is preferred and unchanged. Only
contiguous available/pending spans within one ledger can join. The lease keeps
the unique union of every consumed predecessor fence; a partial last span
retains its original dependencies. Replacement metadata is fully constructed
before nonthrowing splice/erase commits. Unsubmitted rollback conservatively
retains the union, while failed dependency publication quarantines the joined
range. Existing stream waits, context checks, release fences and backing
destruction rules remain in force.

Native pool selection and publication both select the new operation only when
compiled with `DDO_VMM_NATIVE_JOINED_SPANS=ON`. This CMake option defaults OFF
and requires `DDO_VMM_NATIVE_SPAN_POOL=ON`. No Python scheduling heuristic,
runner option, global synchronization or budget increase was introduced.

Actual CPU validation: new contract **176 checks passed**, including seven
injected metadata-allocation failures with the original partition preserved;
legacy ledger **1292 checks passed**; stream dependency contract **PASS**.
All process exits were 0. The new test is also registered in the isolated
toolkit-independent CPU CMake recipe. CUDA test source syntax parsed successfully
without importing Torch or executing CUDA. These results are not CUDA validation.

Candidate build: MSVC 19.44 x64 / CUDA toolkit **12.8.93**, Release, span pool
and joined spans ON, allocation diagnostics ON, source dispatcher and fault
injection OFF. DLL:
`ddo-vmm-allocator/build-joined-spans-cu128-20261006/Release/ddo_vmm.dll`.
SHA256: `EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Initial unpinned toolkit discovery selected 13.2; that separate build is not the
selected candidate and was never executed on CUDA. The explicit 12.8 build above
preserves the prior toolkit. The previous DLL remains unchanged with SHA256
`335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

Proposed bounded CUDA gate, **not yet approved or executed**:

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-joined-spans-cu128-20261006\Release\ddo_vmm.dll'
$taskBase=Join-Path (Get-Location) ('.internal\joined-cuda-' + [guid]::NewGuid().ToString('N'))
.\.venv-vmm\Scripts\python.exe -B -m pytest tests/test_torch_cache_stream_cuda.py ddo-vmm-allocator/tests/test_joined_span_cuda.py -q -x -p no:cacheprovider --basetemp $taskBase --tb=short
```

Hypothesis: two adjacent spans can be reused as one allocation without growing
backing or discarding either last-writer event. Acceptance: the two new 2 MiB
cases preserve both predecessor writes, reuse the same address with unchanged
mapped bytes/backing count, produce the expected final values and retain zero
failure/quarantine counters. Six existing same/cross/legacy-stream cases must
also pass. Stop at the first failure. No real-model execution follows
automatically; even a passing gate does not prove the interrupted video's cause.

### Joined-span small CUDA gate — approved and passed

The user approved precisely the eight cases in the command above. Actual
result, 2026-10-06: **8 passed in 6.09 s**, process exit 0, DDO `.venv-vmm`.
Temporary gate path:
`.internal/joined-cuda-bdb917b6b54b4813b887d058f17f713a`.
Selected candidate SHA256 was checked immediately before execution:
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.

Both joined-span cases verified that two halves behind distinct unfinished
last-writer events reuse the original 2 MiB address as one allocation. Physical
mapped bytes and backing count did not grow; consumer values remained correct
after both producer streams completed, with no failure or quarantine counters.
The six existing same/cross/legacy-stream cases with cache limits 4/0 MiB also
passed. No global PyTorch allocator replacement or model execution was used.

This accepts the selected small CUDA ownership/ordering contract only. It does
not establish a real-video improvement, the cause of the previous stall, image
regression acceptance or general multi-stream/model support. The new build
option remains default OFF, existing binaries/defaults are preserved and no
video benchmark followed this gate.

### Joined-span bounded real-video diagnostic — completed

The user separately approved repeating the residual-only 8 s / 1280x704 /
seed 43 / two-step latent-output command above, changing **only** `-LibraryPath`
to `build-joined-spans-cu128-20261006/Release/ddo_vmm.dll`. No operator profiler,
audio conditioning, budget or chunk-size change. Approved stop limits: 240 s
per host callback, 120 s after the second callback for final completion. These
limits did not trigger. Candidate hash was checked before execution.

Actual result, 2026-10-06, process exit **0**:

- Output: `E:/LTX-2/outputs/ddo_modular_video_20261006_175732_123`.
- Console: `E:/LTX-2/outputs/joined_spans_console_20261006_175732_034.log`.
- Attach **7.8578 s**; total **339.5551 s**; denoise compute-stream phase
  **300.58353125 s**; step event windows **104.2100390625 / 196.28828125 s**;
  tail **0.08518723 s**. Host callback intervals are approximately
  **104.21015 / 194.24813 s**. Two steps do not establish settled steady state,
  despite the report's second-step `steady_state_seconds_per_step` field.
- Both callbacks and `before_denoise.json` are preserved, together with final
  `metrics/modular_run.json` and packed `.pt` output.
- Independent CPU-only artifact inspection, CUDA initialization prohibited:
  float32 video `[1,22000,128]` and audio `[1,201,128]`, both finite.
  Artifact SHA256 `6F4281E7818858DDF6E8C787848ED336C208D4D8A589C3F8B89C9968E41A26EE`.
- Final selected fault, guard/CUDA/capacity admission, source-read, allocator
  pool and pressure-observation failure counters are zero; execution-plan
  mismatches, quarantine, temporary fallbacks, source-cache live tickets and
  transfer-owned events are zero. Timeline completed with 601 samples, zero
  dropped/failed/pending samples and released diagnostic events. Native result
  retirement and unload completed normally; process returned 0. The final
  pre-retirement pool snapshot still contains pending bytes and is **not** a
  claim of an entirely empty allocator pool.
- Sampled baseline-relative GPU peak **6.97265625 GiB**, RSS peak
  **29.54099274 GiB**. GPU baseline about 1.03 GiB: do not describe the delta
  as absolute GPU occupancy. Post-exit GPU observation: 1%, 385 MiB.

The reported native activation pool peak is **6606028800 bytes (6.15234375 GiB)**
in both diagnostic snapshots. This is a cumulative maximum of observed pool
backing growth while diagnostics are enabled, not a separately reset per-step
peak. The native counters have process lifetime and are not reset on snapshot.
Combined with independently resident weights this
indicates possible physical VRAM pressure; it is not proof of WDDM paging or
the second step's cause. First-callback mapped pool bytes are **1195376640**,
higher than **1046478848** in the interrupted control. Therefore these snapshots
do not establish that joined reservation reduced physical pool growth in the
real workload, despite the successful small no-growth contract.

This accepts **bounded selected-workload completion and finite output**, not
parity, image regression, full eight-step performance or default promotion.
Compared with the interrupted control, first callback fell from about 178 to
104 s and the second completed rather than exceeding 768 s. The interrupted
control has no final GPU timing, and one completed candidate run is not a
repeatable speedup proof or confirmation of fragmentation as the sole cause.
The raw final JSON retains a static historical `validation` string saying CUDA
execution is pending; preserve that artifact and use this actual execution
record rather than treating the string as a current validation conclusion.
Defaults/old DLLs remain untouched. No follow-up benchmark was started.

### CPU-only video modulation geometry inspection

Read-only inspection of the current local Diffusers transformer, followed by
meta-device shape evaluation using the selected checkpoint configuration,
identified an activation allocation outside the host's feed-forward and RMSNorm
chunking. CUDA initialization was explicitly prohibited; no weight payloads
were loaded and no production forward or allocator was changed.

For batch 1, 22,000 video tokens, 4,096 channels and nine BF16 modulation
parameters:

- Root `temb` has shape `[1,22000,36864]`: **1,622,016,000 bytes**.
- Root embedded timestep has 22,000 x 4,096 elements:
  **180,224,000 bytes**.
- Each block's table addition creates another **1,622,016,000-byte** storage.
  Its nine unbound views and named parameters retain that storage through the
  block; deleting an individual view cannot release it while others remain.
- Computing only the three parameters needed by the current self-attention,
  text-attention or feed-forward phase would require **540,672,000 bytes** for
  that phase's modulation storage. Releasing every reference before the next
  phase is necessary for the expected **1,081,344,000-byte** reduction in this
  specific allocation, not a guarantee of an equal reduction in GPU peak.

A tiny CPU arithmetic comparison passed **36 exact comparisons**, covering
FP32/BF16, batch 2, scalar-token and varying per-token embeddings, and all three
parameter groups in execution order. This checks table-addition algebra only;
it does not validate a rewritten block, model parity or CUDA performance.

The actual host passes `use_cross_timestep=True`; audio/video cross-modulation
therefore uses the opposite modality's scalar sigma. Do not multiply those
cross-modulation tensors by 22,000 tokens in this workload. Main video timesteps
include image-conditioning masks: the demonstrated constant-timestep broadcast
identity is not permission to collapse them to one scalar.

Next proposed correction is an opt-in phase-local modulation path in the local
Diffusers forward, with exact CPU block parity and original-path preservation
before any separately approved CUDA execution. Keep DDO generic, preserve image
defaults, do not increase budgets, and do not promote the joined-span candidate
based on this shape study. Large root timestep storage remains a separate
limitation; this study does not establish why the second diagnostic step took
196 s or prove WDDM paging.

### Opt-in phase modulation implementation — CPU contracts passed

Implemented in the local Diffusers `transformer_ltx2.py`, not DDO or its native
allocator. `enable_phase_modulation()` / `disable_phase_modulation()` toggle
all transformer blocks; each new block defaults to the original path. The
modulation helper accepts a phase's parameter range before table addition.
Main video/audio parameters are materialized in groups `(0,3)`, `(6,9)` when
applicable, and `(3,6)`, and their references are deleted after the consuming
phase. Main per-token timesteps, root embeddings, cross-modality modulation,
weight budgets and attention algorithms are unchanged. The option is not part
of the serialized weight/config schema.

The modular video host exposes `--phase-modulation`, default false, calls the
model API before DDO attachment, and records the requested setting in its plan.
The image runner and native binaries were not edited in this block.

Actual local Diffusers CPU result: **84 passed in 1.77 s**, including 48 exact
FP32/BF16 block-output and disable/restore comparisons, four asymmetric batch-2
comparisons, view-lifetime/storage-size observation under inference mode,
gradient comparison at `rtol=atol=1e-6`, model-wide toggle/schema checks, and all
29 earlier activation-lifetime contracts. CUDA initialization was prohibited.
The first test attempt exposed test-harness issues (inference tensors have no
`_base`, and removing `del` left empty AST conditionals); these were corrected
in the tests, not bypassed in production. Final complete suite passed.

Local transformer SHA256 after this change:
`3EB14AA6A9E011E568528A342142E8C06E88E421D0540C70FE477B9920C5C376`.
This is CPU acceptance only: no CUDA/model-performance execution or image
regression acceptance has occurred. The root 1.51 GiB modulation embedding is
still present, and autograd can retain saved tensors despite Python reference
release; no training-memory reduction is claimed.

Final host CPU regression suite: **54 passed, 31 subtests passed in 11.25 s**,
including the new opt-in/default-off plan-reporting contract. Existing modular
blocks and reference-host CPU contracts passed in the same invocation.

### Phase-local modulation bounded real-video gate — approved and passed

The user approved the exact previous joined-span 8 s / 1280x704 / seed 43 /
two-step latent-output command, adding only `--phase-modulation`. Resident
ceilings remain 2 GiB for the transformer and 4 GiB for the native text encoder;
source cache/registration remains 24.625 GiB and feed-forward/normalization
chunks remain 1024. No audio conditioning, operator profiler or decode.

DLL SHA256 was verified before execution:
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Local Diffusers transformer SHA256 was verified as
`3EB14AA6A9E011E568528A342142E8C06E88E421D0540C70FE477B9920C5C376`.
Execution used `E:/LTX-2/.venv-vmm`, not the image host environment. Stop
limits were 240 s per host callback and 120 s for finalization after the second
callback. Neither limit triggered. Process exit **0**, status **completed**.

Evidence, 2026-10-06:

- Console: `E:/LTX-2/outputs/phase_modulation_console_20261006_192405_914.log`.
- Output/report directory: `E:/LTX-2/outputs/ddo_modular_video_20261006_192405_981`.
- Denoise CUDA-event phase **98.68746875 s**, versus **300.58353125 s** in the
  previous matched diagnostic: approximately 3.05x faster in this selected pair.
- CUDA-event step windows **49.8096640625 / 48.85390234375 s**, versus
  **104.2100390625 / 196.28828125 s**. Host submission intervals
  **49.8095988 / 48.0434901 s**. Tail **0.02389798 s**.
- Total **138.3953 s**, versus **339.5551 s**. Both totals exclude video/audio
  decoding because the selected output type is latent.
- Cumulative observed native activation-pool peak **5,523,898,368 bytes
  (5.14453125 GiB)**, versus **6,606,028,800 bytes (6.15234375 GiB)**:
  reduction **1,082,130,432 bytes (1.0078125 GiB)**, consistent with the
  phase-local storage calculation. These are cumulative pool maxima, not
  separately reset per-step or whole-GPU peaks.
- Sampled GPU baseline-relative peak remains **6.97265625 GiB**, with baseline
  **1.02685546875 GiB**. Do not claim the 1.01 GiB pool reduction also reduced
  sampled absolute GPU occupancy. Sampled RSS peak **28.87080383 GiB**.
- All selected final fault/admission/source-read/pool/pressure failure counters,
  execution-plan mismatches, quarantine, temporary tensors/bytes, live source
  tickets and transfer-owned events are zero. Native timeline completed with
  601 samples, zero drops/failures/pending samples, diagnostic events released.
  Result retirement and unload completed normally. Final pre-retirement pool
  still has 188,859,904 pending bytes; it is not an empty-pool proof.
- CPU-only saved artifact comparison, CUDA initialization prohibited, found
  video `[1,22000,128]` and audio `[1,201,128]` float32 finite and **bit-exact**
  against the previous joined-span artifact; maximum absolute differences
  are zero. Metadata also matches. Artifact SHA256 is identical:
  `6F4281E7818858DDF6E8C787848ED336C208D4D8A589C3F8B89C9968E41A26EE`.

The approved bounded gate passes correctness, cleanup, pool-peak reduction and
non-regressing denoise criteria. This is strong selected-workload evidence for
the phase-local correction, not settled steady-state, eight-step/full-decode,
image-regression or general model/platform acceptance. No defaults, original
DLLs, image runner, linear runtime or protected NVMe route were changed. No
additional CUDA execution followed automatically. The next performance gate
would be the full eight-step video workload, subject to separate approval.

### Phase-local modulation eight-step gate — completed with timing limitation

The user separately approved the same command with `--steps 8` instead of 2.
Resolution, duration, seed, latent output, no audio conditioning, native encoder,
resident/source/registration/staging ceilings, chunks, auxiliary ownership and
phase modulation were preserved. DLL and local transformer hashes were checked
and matched the two-step gate above. Executed from the host `.venv-vmm`.
Approved 240 s callback / 120 s finalization stop limits did not trigger.

Actual result, 2026-10-06, process exit **0**, status **completed**:

- Console: `E:/LTX-2/outputs/phase_modulation_8steps_console_20261006_195209_465.log`.
- Output: `E:/LTX-2/outputs/ddo_modular_video_20261006_195209_548`.
- CUDA-event denoise phase **454.507125 s**, total latent-output run
  **494.8370 s**, tail **0.05054547 s**.
- CUDA step windows, in order: **47.2805664, 47.4211758, 50.8688789,
  43.6166172, 44.1700820, 46.5497617, 129.3569375, 45.1925703 s**.
  Mean of steps 2–8 **58.16800335 s**, median **46.54976172 s**. The seventh
  step is a material outlier; do not exclude it from the reported mean or
  claim stable approximately 49 s/step acceptance.
- Host callback intervals approximately **47.2820, 46.6563, 50.8039, 43.7005,
  44.0890, 45.3033, 130.4907, 45.3180 s**. All eight durable snapshots and
  `before_denoise.json` were preserved. Finalization completed promptly.
- Cumulative pool peak **5,523,898,368 bytes (5.14453125 GiB)**, unchanged
  from the phase-local two-step gate. Sampled GPU baseline-relative peak
  **6.97265625 GiB**, baseline **1.02685546875 GiB**, RSS peak
  **29.34511948 GiB**. No claim of reduced whole-GPU occupancy.
- Final selected fault/admission/source-read/pool/pressure failures,
  execution-plan mismatches, quarantine, temporary tensors/bytes, live source
  tickets and transfer-owned events are zero. Native timeline complete with
  **2359 samples**, zero drops/failures/pending samples and diagnostic events
  released. Retirement/unload completed normally. No owned test processes
  remained after exit; post-exit sampled GPU usage was 676 MiB, 1% utilization.
- Independent CPU-only artifact inspection, CUDA initialization prohibited:
  float32 video `[1,22000,128]` and audio `[1,201,128]`, both finite.
  Artifact SHA256
  `985D5415B672FBEEF9408A6EAA920E4FB8CF4283EBA152F3060AA87FE74A267D`.
  This is not an eight-step bit-exact comparison or decoded-video quality gate.

Read-only timeline inspection finds 48 block windows per step. Their summed
durations are approximately **45.2385, 45.9359, 49.5343, 42.2564, 42.8404,
45.2007, 127.9759, 43.8353 s**. The seventh-step delay is distributed across
block execution windows, not isolated to one execution unit. Those windows
include dependencies and are not engine/kernel utilization measurements;
the available evidence cannot identify WDDM paging, clocks or another root
cause. Native driver host calls and final error counters do not establish
the cause either.

Accept selected eight-step completion, finite output and cleanup only. The
approximately 49 s/step stability objective remains open because the seventh
step raises the steps-2–8 mean to 58.17 s. Do not compare the latent-output
total directly to historical MP4/full-decode totals or extrapolate the earlier
two-step speedup. Defaults remain off; image/linear/NVMe paths unchanged.
No repeated benchmark, decode, audio-conditioned run or parameter change
followed automatically.

### Read-only seventh-step investigation

Compared existing callback snapshots, native timeline windows and the current
host/model forward without executing CUDA or changing production code.
The user reported no known competing GPU workload during the interval. Do not
attribute this outlier to another application without evidence.

| Observation | Step 6 | Step 7 | Step 8 |
| --- | ---: | ---: | ---: |
| CUDA step window | 46.5498 s | 129.3569 s | 45.1926 s |
| Upload batch window union | 3.0428 s | 3.0741 s | 2.9908 s |
| Compute/upload dependency window union | 0.000109 s | 0.000156 s | 0.000250 s |
| Source-ready host counter delta | 0.013600 s | 0.014617 s | 0.013796 s |
| Batch-submit host counter delta | 12.5358 s | 20.3591 s | 12.3513 s |

Each step records one execution-plan pass, 64 execution-buffer uploads, 88
compute-wait events, 37,378,058,752 upload bytes, 11,614,405,880 native source-read
bytes in 6,328 chunks, 2,792 cache hits and 1,251 misses/bypasses. No additional
main-model pass or expanded transfer volume appears in step 7. Registrations,
unregistrations, cache pressure evictions, registration-wait time and registered/
committed-cache deltas are zero in steps 6–8. Callback mapped-pool deltas are
also zero; endpoint stability does not establish identical intermediate
residency or WDDM usage.

Step 7 block 0 remains approximately 0.93 s. Blocks 1–35 take roughly
3.02–3.43 s, versus approximately 0.85–1.1 s normally; block 36 returns to
approximately 1.07 s and subsequent blocks are near their usual durations.
Step 6's last block also takes approximately 2.15 s. This abrupt, bounded
slow interval is not a uniform seven-step workload change. Host guidance is
configured for a single conditional forward and the code has no seventh-step
special attention or modulation path in this configuration.

The upload-window union is event-window geometry, not isolated DMA engine
time. Batch-submit counters are host API elapsed time and can contain waits;
they overlap execution and must not be subtracted from CUDA phase time or
interpreted as extra DMA duration. The increased submit counter alone does
not account for or identify the roughly 83 s step-time increase.

No GPU clock/power/P-state history, external workload census or sampled WDDM
residency was recorded during this run. Available measurements cannot
distinguish driver paging/scheduling, transient GPU state or an operator-level
slowdown. Retain the successful phase-local correction as explicit opt-in,
preserve the complete mean including the outlier, and make no speculative
allocator/attention/budget patch. A future separately approved necessary run
should include read-only GPU-state sampling alongside existing step snapshots
before considering another performance correction; no standalone repeat or
new CUDA test was started during this investigation.

### Read-only GPU-state observer preparation

Prepared a local-only observer in ignored
`.internal/scripts/observe_windows_gpu.ps1`, independent of the model/runner.
It never imports Torch, installs an allocator, initializes CUDA, changes clocks
or alters residency/budgets. NVIDIA queries record index/UUID, timestamp,
utilization, reported memory, graphics/memory clocks, power, temperature and
P-state, with an owned-process query timeout. Windows performance counters
record adapter and per-process dedicated/shared memory, plus active GPU-engine
instances and process names. Memory units and sample validity are explicit.
Unavailable sources are marked unavailable, never silently replaced by zero.

Output is a new, non-overwritten JSONL file with start/end and counter timestamps;
configured sleep is between queries, not a guaranteed sampling period. A stop
file and finite sample ceiling bound its lifecycle. It has observation overhead
and is not a CUDA engine profiler, WDDM paging trace or proof of physical
residency. NVIDIA UUID and Windows adapter LUID are distinct identifiers; do not
silently join them by index or sum distinct adapters as one device.

Actual read-only smoke on 2026-10-06: two samples completed, NVIDIA/Windows
sources both readable, 87/88 retained counter records and zero invalid records.
No inference or CUDA test was run. Local evidence:
`.internal/gpu_observer_smoke_759c138244ec4ebea7b06ea947f8c027.jsonl`.
The proposed next separately approved workload is the same eight-step video
configuration with final video output (rather than another latent-only repeat),
observed by this helper; no approval/execution is implied by its preparation.

### Interrupted eight-step video-output observation — 2026-10-06

The separately approved video-output attempt used the same explicit candidate
DLL and phase-local modulation controls, with the independent observer enabled.
It reached eight host callbacks but was interrupted at the agreed limit of
120 seconds after the last callback without a published MP4. The wrapper
reported failure following intervention. No final CUDA timing, final metrics,
finite-output check or normal retirement acceptance is available for this run.
Host callbacks are submission boundaries, not proof of GPU completion.

The command omitted `--stream-video-decode --decode-device cuda:0`, selecting
the runner's CPU/non-streaming decode defaults. The earlier accepted full-video
run used streaming CUDA decode (42.5483 s for that phase). Consequently, this
attempt's finalization timeout is not a matched decode regression. Buffered
console output also prevents identifying the exact interrupted phase from the
last printed callback alone.

Local evidence: host output directory
`E:\LTX-2\outputs\ddo_modular_video_20261006_204135_932`, console
`E:\LTX-2\outputs\phase_modulation_video_console_20261006_204135_862.log`,
and observer `.internal/video_gpu_state_20261006_204124_182.jsonl` (195 samples,
SHA256 `0CDE43E7335F5E0A98AC7D6C3DD69330744C44EC033D92EA9CEC56BFFAFC4C51`).
Owned runner/observer processes were stopped; evidence was retained.

Host callback intervals were approximately 62.02 / 48.94 / 61.49 / 45.82 /
46.33 / 166.69 / 124.98 / 53.19 s. These are not final CUDA step measurements.
The worker's Windows shared-memory counter was already approximately
26.50 GiB during step 2, reaching approximately 26.82 GiB during the slow
intervals. Thus, missing large shared allocation does not explain this run.
Graphics/memory clocks remained near 1920–1935/6801 MHz; median reported
power fell from approximately 167–177 W in preceding intervals to 100–104 W
in the two slow intervals. Dedicated-memory counters decreased while shared
counters increased. These observations are compatible with memory pressure,
but do not prove physical WDDM paging, its cause or exclusive attribution.
The observer itself adds overhead; no matched observer-free comparison exists.

Next proposed correction is command-level: explicitly restore streaming CUDA
decode and use unbuffered phase logging, preserving allocator and budget
controls. Another CUDA execution requires separate command/criteria approval.
This interrupted attempt neither supersedes the completed latent gates nor
promotes a default or establishes full-video performance acceptance.

### Completed streaming-CUDA video-output gate — 2026-10-06

Separately approved single repetition restored explicit streaming CUDA decode
and process-scoped `PYTHONUNBUFFERED=1` (restored on exit). No observer ran in
this repetition. Native DLL, phase modulation, budgets and chunks were unchanged.
The approved command was:

```powershell
Set-Location E:\LTX-2
# Set PYTHONUNBUFFERED=1 for this invocation and restore its previous value afterward.
.\run_ddo_vmm_video_modular.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-joined-spans-cu128-20261006\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 8 --steps 8 --seed 43 --no-use-audio-condition --output-type video --stream-video-decode --decode-device cuda:0 --diagnose-denoise --ddo-vmm-resident-weight-budget-gb 2 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources --ddo-vmm-execution-auxiliary-ownership --phase-modulation
```

Completed with exit 0 and `status=completed`. Total elapsed: **697.7858 s**;
denoise CUDA phase: **612.076250 s**; host denoise phase: **612.1745 s**;
streaming decode/save: **45.0749 s**. CUDA steps were **56.3486 / 46.4514 /
65.2468 / 122.3519 / 64.3212 / 49.9341 / 91.3168 / 116.0213 s**;
steps 2–8 mean **79.3776 s**. This confirms completion, not stable approximately
49 s/step performance; the prior completed latent-only phase was 454.5071 s.
Different output/logging conditions and uncontrolled system/driver state prevent
assigning that increase to a particular code change or to decode execution,
which occurs after denoise. No allocator/default change or further run followed.

The runner checked 193 frames at 1280x704 and finite nonempty stereo audio
(384480 samples/channel at 48000 Hz), then saved a 1,361,896-byte MP4.
This is structural/output completion evidence, not visual quality or bit-exact
parity. The separate system `ffprobe` link pointed to an unavailable executable,
so independent container probing was not completed. The report's static
`validation` description still says CUDA execution pending; preserve that raw
field, but use this run's actual completion/timing evidence for its selected scope.

Final encoder/transformer selected fault/admission/pool/source-read failures and
quarantined bytes were zero; collected source tickets ended at zero. Native
timeline was complete with zero dropped/failed/pending records and
`event_cleanup=released`. Cumulative native activation-pool peak remained
5.14453125 GiB. Sampled GPU baseline-relative peak was 6.97265625 GiB
(baseline approximately 1.0269 GiB; sampled absolute approximately 7.9995 GiB),
and sampled process RAM peak was 29.3304863 GiB. Owned runner processes exited.

Evidence directory: `E:\LTX-2\outputs\ddo_modular_video_20261006_213214_265`;
report: `metrics\modular_run.json`; console:
`E:\LTX-2\outputs\phase_modulation_stream_video_console_20261006_213214_193.log`.
MP4: `ltx23_modular_1280x704_8.0s_seed43.mp4`, SHA256
`A5899F809920838EC36A7998F54ADDA614CD7EFCCED269AD26AF5698E0F0426C`.
DLL SHA256 remains
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Image, linear and protected NVMe paths were not altered or rebenchmarked.

### Selected 1.5 GiB transformer-budget comparison — 2026-10-06

Separately approved single full-video repetition used the preceding streaming
CUDA command, changing only `--ddo-vmm-resident-weight-budget-gb 2` to `1.5`.
DLL, encoder/staging/source ceilings, phase modulation, chunks, prompt, seed,
8 s/eight-step workload and process-scoped unbuffered logging were retained.
No observer ran. No further benchmark or default promotion followed.

Exit 0, `status=completed`: CUDA denoise **436.044750 s**, host denoise
**436.1425 s**, streaming decode/save **42.7006 s**, total **519.4920 s**.
CUDA steps: **48.2803 / 44.5408 / 57.5344 / 112.7347 / 46.4698 / 41.9453 /
42.1620 / 42.3261 s**; steps 2–8 mean **55.3876 s** includes the slow fourth
step. Denoise improved approximately 28.8% versus the preceding 2 GiB run
(612.076250 s). A single comparison with uncontrolled system/driver state
does not establish causality or stable performance; retain the slow-step evidence.

The ceiling was applied: final resident weights were 1,585,446,912 bytes
(approximately 1.4766 GiB), versus 2,143,289,344 bytes previously. Execution
buffer resident weights remained 1,547,698,176 bytes; arena resident weights
fell to 37,748,736 bytes. Cumulative native activation-pool peak remained
5.14453125 GiB. Sampled baseline-relative GPU peak remained 6.97265625 GiB
(sampled absolute approximately 7.9995 GiB), not a whole-run GPU peak reduction.
Sampled process RAM peak was 28.7613258 GiB. A user Task Manager capture
showed 7.3/8.0 GB dedicated and 26.4/32.0 GB shared; the earlier external
capture showed the same dedicated total and 25.2/32.0 GB shared. Instantaneous
totals do not identify physical page placement/migration or prove paging causality.

Runner checks confirmed 193 frames at 1280x704 and finite stereo audio. MP4
SHA256 matches the preceding 2 GiB run:
`A5899F809920838EC36A7998F54ADDA614CD7EFCCED269AD26AF5698E0F0426C`.
This establishes equality of those two saved artifacts, not general model parity.
Selected final fault/admission/pool/source-read failures and quarantined bytes
were zero. Encoder source tickets retired 8 to 0; transformer tickets ended
at zero. Native timeline completed with zero dropped/failed/pending records
and released events. No allocator code, defaults or frozen routes changed.

Evidence: `E:\LTX-2\outputs\ddo_modular_video_20261006_234635_397`, report
`metrics\modular_run.json`, output `ltx23_modular_1280x704_8.0s_seed43.mp4`;
console `E:\LTX-2\outputs\phase_modulation_budget15_console_20261006_234635_338.log`.

### User-run CUDA/WDDM trace inspection — 2026-10-07

User executed the separately proposed Nsight Systems 2025.6.3 capture at the
same 1.5 GiB ceiling, 8 s/eight-step streaming-CUDA workload, without recorded
standby-purge events. Report:
`E:\LTX-2\outputs\ddo_cuda_wddm_20261007_084526.nsys-rep`;
runner output: `E:\LTX-2\outputs\ddo_modular_video_20261007_084529_543`.
CPU-only analysis exported a separate SQLite copy without modifying the report.
A partial 3,471,679,488-byte copy on E: was removed after owned-export
termination to avoid filling that disk. The completed 6,197,682,176-byte copy
is temporary on C:, not a public deliverable. Original evidence remains intact.

Runner completed with total **430.2082 s**, CUDA denoise **346.27859375 s**,
decode/save **44.0518 s**. CUDA steps: **45.1554 / 43.5220 / 42.8971 / 42.9352 /
41.6315 / 42.4231 / 43.0541 / 44.6309 s**; steps 2–8 mean **43.0134 s**.
The slow interval did not reproduce. These are instrumented observations,
not matched performance acceptance or evidence that profiling/purge fixed it.
MP4 SHA256 matches the preceding two runs; selected final safety counters,
pending tickets and native timeline loss/pending/failure counts were zero,
and native diagnostic events were released.

The pasted console exceptions originated in PSReadLine cursor rendering while
pasting, not CUDA/DDO. Export confirms 377,561 CUDA kernel records attributed
to the runner Python process, plus WDDM paging/eviction events. Explicit
diagnostic messages reported no severity-3 errors or lost/dropped/overflow
events; this does not prove exhaustive trace completeness. Warnings about
no CUDA events belonged to other processes, not the kernel-producing worker.
The worker has a substantive compatibility warning: installed CUDA driver
13.4 is unsupported by this Nsight build, which collected with driver-13.2
libraries and software-instrumented tracing. Preserve that limitation.

System-wide WDDM records include approximately 1.24 million paging-queue start
events and 594,050 allocation-eviction events, even in this regular execution.
These global counts cannot be attributed wholesale to DDO or treated as
physical migration bytes/latency. The capture establishes observable CUDA and
WDDM activity, but cannot diagnose the earlier slow window because it did not
occur. No repeated CUDA run, code/default change or profiler installation followed.

### User-run standby-purge comparison — 2026-10-07

User supplied `C:\Users\pelol\AppData\Local\Temp\nsys-report-2946.nsys-rep`
(758,585,354 bytes). The corresponding completed runner report is
`E:\LTX-2\outputs\ddo_modular_video_20261007_090423_193\metrics\modular_run.json`.
Recorded runner controls match the preceding 08:45 run except output directory.
The event `purge_windows_standby_before_transformer` confirms helper execution,
2.9247 s, system-information class 80, command 4 and
`SeProfileSingleProcessPrivilege`, without a recorded error. This is separate
from the still-enabled PyTorch CPU allocator cache release.

| Observation | Without standby purge | With standby purge |
| --- | ---: | ---: |
| CUDA denoise phase | 346.27859375 s | 345.5176875 s |
| Mean CUDA steps 2–8 | 43.0134258 s | 43.0687796 s |
| Streaming decode/save | 44.0518 s | 43.6540 s |
| Full elapsed | 430.2082 s | 430.7291 s |

With-purge CUDA steps: 44.0125 / 43.5133 / 43.5378 / 42.5951 / 42.5408 /
42.5009 / 43.4857 / 43.3079 s. No meaningful gain is demonstrated by this
single user-run instrumented pair; neither run reproduced the earlier slow
window. Keep native standby purge disabled by default, retaining the optional
Windows helper and the independent PyTorch host-cache release. No further
repeat or removal is justified by these observations alone.

Both completed MP4s have SHA256
`A5899F809920838EC36A7998F54ADDA614CD7EFCCED269AD26AF5698E0F0426C`.
The with-purge runner checked 193 frames at 1280x704 and finite audio,
selected final safety counters were zero, source tickets ended at zero and
native diagnostic events were released with no pending/failed/dropped records.
The second large Nsight report was not exported: runner timing/event evidence
was sufficient for this comparison. Do not assert independently verified
trace completeness or identical profiler configuration for that second file.
No library code, defaults or frozen routes were changed.

### Feed-forward chunk 1024/2048 A/B — 2026-10-07

User approved two sequential fresh-process cases, stopping on failure or memory
anomaly: 1280x704, 8 s at 24 fps, 193 frames, seed 43, BF16, two steps, no
input audio conditioning and latent output. Native transformer/encoder ceilings
were 1.5/4 GiB, staging 1.5 GiB, persistent source cache/registration 24.625 GiB.
Phase-local modulation, RoPE reuse, execution auxiliary ownership and PyTorch
CPU-cache release remained enabled; normalization chunk stayed 1024. No Nsight,
decode or operator profiler was used. DLL SHA256 remained
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Requested controls differed only in FF chunk and output directory.

Evidence under `E:\LTX-2\outputs`:

- `ddo_modular_video_20261007_100103_955`: chunk 1024, total 136.0103 s,
  CUDA phase 97.595953125 s, steps 52.8862734375 / 44.68571875 s.
- `ddo_modular_video_20261007_100358_374`: chunk 2048, total 130.9887 s,
  CUDA phase 92.949265625 s, steps 50.279265625 / 42.64676171875 s.

Each directory retains `metrics/modular_run.json`, denoise snapshots and
`ltx23_modular_1280x704_8.0s_seed43.pt`. Both completed with exit 0.
Selected encoder/transformer fault, guard/CUDA/capacity admission, pool,
source-read and pressure-observation failures, quarantined bytes and ring
fallback bytes were zero. Timelines completed with zero dropped/failed/pending
records and released events. Sampled absolute GPU peak was 7.9995 GiB in both;
sampled RSS peaks were 28.5320 / 28.5106 GiB. These samples are not continuous
residency measurements or proof of spare VRAM.

CPU-only `torch.load(..., map_location="cpu", weights_only=True)` comparison
did not initialize CUDA. Metadata matched, and both video `[1, 22000, 128]`
and audio `[1, 201, 128]` FP32 latents were finite. Video maximum absolute
difference was 1.0112670660, RMS 0.0440976396, relative RMS 0.0481912903;
audio maximum absolute difference was 0.0933105946, RMS 0.0159603525,
relative RMS 0.0255889539. Outputs are not bit-exact; equivalence and visual
quality were not established. Do not attribute the numerical differences
solely to GEMM selection without an isolated operator comparison.

The second-step reduction was 4.5628829261%, below the predeclared 5% gate.
No numerical-equivalence tolerance had been approved. **Promotion rejected**;
retain the selected chunk 1024 and do not extend this sequence automatically.
No production code, defaults, image/linear/NVMe routes or DLL changed. A
single ordered two-step pair is not stable eight-step performance acceptance.
