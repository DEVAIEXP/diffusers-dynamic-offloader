# Native cold-source scheduling: corrective review (2026-10-02)

Latest multi-window gate completed: eight CUDA tests PASS (5.13 s), span PASS,
image identical and zero native errors. Retained window peak eight/limit eight,
final owned windows zero proves multi-window submission was used. Warm
1.423592 s/it, Generate 24.8827 s (FAIL), prepare 1.6399 s, first GPU step
6.071087 s. Source-ready wait still 4.974880 s. The whole-window barrier is
no longer an implementation gap; removing it alone was insufficient. No
promotion/retry/limit change; investigate remaining cold critical path before
another patch. Details: [native-source-flights-gate.md](native-source-flights-gate.md).

Pre-execution checkpoint: isolated multi-window candidate implemented after the failed unlocked-wait
gate: submission and retirement separated, at most eight retained flights
(bounded by existing workers/cache slots). No new memory ceilings/threads.
Native CPU completion polling, unordered retirement, partial-failure owners
and close drain remain outside runner code. ON/OFF compiled, 75 selected
Python CPU tests PASS in each; native flight/read-wait/scheduler/span/stream
contracts PASS. Optional public window snapshots added. No CUDA execution.
Next approval proposal: [native-source-flights-gate.md](native-source-flights-gate.md).
Do not reuse earlier approvals or claim measured improvement yet.

Latest isolated correction tested: source completion waits release the
global admission lock, with shared results, per-slot waiter retention and
close waiting for retirement. Dispatcher-OFF keeps locked consumable waits.
ON/OFF compiled; candidate 74 Python CPU PASS and 32 native lifecycle iterations
PASS; existing scheduler/span/stream CPU contracts PASS. Approved seven CUDA
cases and span probe PASS. One model run: warm 1.412124 s/it, Generate 24.4543 s,
prepare 1.6631 s; image identical, zero errors. Performance gate still FAILED.
See [native-source-unlocked-gate.md](native-source-unlocked-gate.md) for the
candidate/hash and completed gate. Earlier failed candidate and working milestone
remain preserved; no whole-phase win, promotion or retry. Source preparation
fell to 0.025003 s, but consumer readiness wait is still 5.093906 s. Code still
submits/waits/retires one whole window before dispatching the next; investigate
bounded multi-window submission with ownership-safe retirement, not new budgets.

Earlier locked-candidate status: scheduler connected to an isolated native source-dispatcher candidate;
compilation/CPU and approved small CUDA gates passed; full-model performance
gate FAILED (Generate 25.8173 s, prepare 7.5827 s, warm 1.390072 s/it).
No capacity adjustment, retry or default promotion. Preserve span-pool milestone
`8CAAD072...` and protected `4AC3D8B...`.

## Native connection checkpoint

Isolated CMake `DDO_VMM_NATIVE_SOURCE_DISPATCHER=ON` (default OFF), alongside
the existing span pool. `build-source-dispatcher-20261002` candidate SHA256
`24A2F4F7BEBBD27BC03EB93C9BB87E6582B3152A4C0A4BE59486C61ED06A82DD`.
Native dispatcher owns a bounded window-metadata queue (existing cache slot
count); the existing eight I/O workers and 256-chunk task queue are unchanged.
The dispatcher performs no CUDA calls, reads one admitted window through those
workers, drains all its chunks even on partial submission failure, then signals
readiness. Demand promotion occurs natively. Host storage stays cache-owned
until dispatcher join; immutable Entry owners retain file handles.

Source-only batch preparation bounds attempts by existing slot geometry and
stops on bypass. Actual consumer submission remains authoritative and uses
the existing cache/ring fallback. Source owners remain alive after batch/file
owners close. Clear rejects live tickets, stops/joins outside the cache mutex,
then releases storage/registrations; successful clear permits a fresh scheduler.
Completed scheduler leases are retired so they cannot block pressure trim or
associate old slot storage with a new file window. Errors never publish ready.

Optional public exports advertise capability and prepare immutable descriptors.
Older/disabled DLLs do not walk units or start preparation. The native adapter
provides generic unit descriptors after transient view setup exits; no runner
or architecture-specific scheduling code. Startup failure removes hooks and
drains the store only after that view scope has released its mappings.
Registration and DMA remain at demand transfer; existing pressure housekeeping
may unregister entries. Preparation performs no new CUDA registration.

CPU selected suite: 74 passed in the candidate and dispatcher-disabled build;
protected older DLL: 73 passed, one optional-export test skipped. This covers
bindings/metrics/profiles plus mocked look-ahead/layout/view ownership, not
real asynchronous I/O/CUDA safety. Earlier minimal-store mock failure was fixed
without changing protected semantics. Scheduler/span/stream CPU executables
also PASS. Three new CUDA cases plus the existing persistent-registration
recycling case passed (four PASS); the span probe also passed. One approved
model run produced the identical image and zero native errors, but failed
the whole-Generate performance criterion.
See [native-source-dispatcher-gate.md](native-source-dispatcher-gate.md) for the
executed sequence/results; its approval is exhausted.

The first GPU step fell to 1.693964 s, but source preparation consumed
6.627046 s and all source bytes had been read before denoise. Read-only review
identified `wait_file_read` holding the global ring mutex over `future.get()`;
foreground `commit_storage` needs that same mutex. Fix this ownership-safe
completion/admission serialization before another performance gate. Preserve
read/slot owners until completion, including concurrent waiters, partial
submission failure and clear/close; do not just move an unguarded future out
of the lock. This run proves no whole-phase gain.

## CPU implementation checkpoint

`ddo-vmm-allocator/native/include/ddo_vmm_source_scheduler.h` implements an
externally serialized metadata scheduler. Immutable window identity includes
file identity/version/range. Native-supplied queue/entry bounds include active
and settled owners; actual storage admission remains the coordinator's job.
Demand promotes existing work or displaces only unsubmitted speculation;
full required queues return deferred for the existing demand/fallback path.
No required request is dropped to admit another speculative request.

Work leases retain file/storage owners. Cancellation of an active read retains
ownership until all-chunk completion is acknowledged. New demand can reclaim
that cancelled-but-active read without duplicate I/O; completion failure stays
failed. Close rejects new work and remains undrained until active reads finish.
Foreign/stale tokens and duplicate completion are rejected. Failed entries
are not silently retried. Native integration must serialize operations and
retain the coordinator until acknowledgement; this header creates no workers,
executes no I/O/CUDA, and does not itself prove completion or driver safety.

Isolated `build-source-scheduler-cpu-20261002`: C++20/MSVC compilation PASS;
source scheduler **169 checks, zero failures** (includes repeated lifecycle
checks, not 169 independent scenarios). Existing span ledger **1,278 checks
PASS**, stream-dependency CPU contracts PASS. Only these CPU executable targets
were built/run; the VMM library was not rebuilt. Both milestone/protected DLL
hashes rechecked unchanged. Native connection and CUDA/performance gates remain.

## Evidence

The two unchanged repeatability cases passed at 1.412275 / 1.375524 GPU
steady-state s/it. First GPU steps were 6.540682 / 6.819729 s; DDO prepare
0.8337 / 0.7720 s; Generate 23.2125 / 23.2392 s.

Before denoise, source-read bytes and committed-cache bytes were both zero.
Afterward, source-read bytes were 25,846,148,360 and committed bytes
26,038,239,232 in both cases. Cumulative source-ready host wait was
5.719305 / 6.054061 s; registration 0.437358 / 0.411195 s; native batch
submission 6.346453 / 6.634525 s. These nested/process-lifetime counters
are not additive phases and do not prove every wait happened in step one.
They identify cold source readiness as the primary next investigation,
not steady-state disk rereads or registration alone.

`ddo_vmm_source_cache.cpp` requests source windows during batch submission,
then `ddo_vmm_source_ticket_copy_async()` calls `wait_entry()` before DMA.
The immutable batch is created lazily by native transport on first use.
Requests enqueue asynchronous I/O, but demand submission still waits for
source readiness. Current GPU look-ahead runs after the current unit has
enqueued compute; it is not an independent native source-fill pipeline.

`ddo_vmm.cpp` already has eight I/O workers, 2 MiB chunks and a bounded
256-task queue. Queue admission may block when full. Adding an eager
loop over the entire model on the caller would therefore not establish
asynchronous preparation, and could merely move the same delay.

## Intended correction and required contracts

Separate native source preparation from destination/DMA submission. Supply
immutable source descriptors once through the existing DDO/submodule boundary;
native code owns bounded read scheduling, deduplication and readiness. No
runner-managed windows, CPU tensor retention or Python residency heuristics.

Preparation must be asynchronous with explicit lifecycle/ownership, not
an unbounded caller-thread enqueue loop or an unconditional full-model wait.
Use the existing host capacity, pressure guards, worker/queue limits and lazy
commit policy. Do not reserve GPU slots or pin weight-arena pages merely to
prepare a source. CUDA registration remains at the owning-context transfer
boundary unless separately designed and tested; an I/O worker must not assume
the caller's CUDA context.

Required ordering: actual consumer demand remains correct if preparation
is incomplete, execution order changes, capacity is unavailable or optional
speculation is cancelled. Speculation cannot strand a required source behind
its own bounded queue or prevent ordinary ring fallback. Do not assume a
specific architecture or that registration order is forward order.

Descriptors/file owners/storage survive every queued read. Cancellation
prevents new scheduling but does not free bytes underneath workers or DMA.
Close drains outstanding ownership before releasing cache/ring/file handles.
Partial scheduling/read failures remain visible; no success/ready state may
be inferred from queue acceptance alone. Unregistering preserves native bytes.

Implement/test a CPU-only scheduler contract first: bounded progress and
backpressure, window deduplication, demand priority, changed execution order,
capacity refusal, partial enqueue/read failures, cancellation and close while
work is outstanding. Then connect native owners in an isolated candidate.
Compilation and mock tests are not CUDA/performance approval.

## Acceptance gate to propose after implementation

Request explicit approval for small CUDA source-lifecycle/demand/fallback
contracts followed by named unchanged eight-step cases. No run is approved
by this document. Stop at the first failed gate or memory anomaly.

Require identical image, zero new errors/quarantine/fallbacks, final tickets
zero and conserved source/registration/pool accounting. Preserve the observed
warm milestone (preapproved tolerance required) and evaluate both prepare and
Generate. A smaller first step with equally larger prepare is not a gain.
Compare whole Generate against 23.2125 / 23.2392 s, not only the original
24.9113 s observation. Record CPU source scheduling/ready wait separately
from CUDA registration, DMA and compute; do not change profiling overhead
silently. Text encoder expansion remains after transformer closure.
