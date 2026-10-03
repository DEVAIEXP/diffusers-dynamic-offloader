# Native allocation pool: split, reuse and ownership contract

Status (2026-10-02): connected native candidate passed CPU contracts, three
approved small CUDA gates and one model case (warm 1.387751 s/it). Not a default
promotion or proof of broader coverage/repeatability. Scope is the native PyTorch bridge
used by VMM-fast, not the model-weight arena, protected NVMe runtime or linear
presets. The protected binary remains the performance fallback.

## Evidence and intended correction

The footprint observer measured 625,065,660 bytes of granularity padding and
515,899,392 bytes of oversized reuse in the final live snapshot. Exact-size
cache selection removed oversized reuse but caused 3,182 eviction attempts and
12.709596 seconds of accumulated host eviction synchronization during denoise;
warm steps worsened to 1.950957 seconds. Do not promote that candidate.

The bridge currently gives each allocation sole ownership of a region, and
`AllocationRecord::reset()` drains its fence before destroying that region.
A returned interior pointer must never follow that destruction path as though
it owned the mapping. Suballocation therefore requires an ownership change,
not simply offset arithmetic or replacing the cache lookup.

The correction is to split logical spans inside retained, already mapped
native backing. No weight copy, CPU tensor cache, physical remap or driver
allocation is needed to reuse a suitable span. Mapping granularity and logical
allocation alignment are distinct. The ledger uses 512-byte logical alignment,
with checked rounding and the bridge's existing zero-request normalization;
nonzero requests currently round directly to driver granularity, not 512 bytes.
Validate
the interior-pointer contract on CUDA before claiming support.

## Backing and span ownership

- A backing owner has one region, device/context identity, mapped size and
  generation. Its address and physical mapping remain stable while any live,
  reserved, pending or quarantined span references it.
- Each allocation is a lease: backing owner, offset, requested bytes, aligned
  span size, generation and retained predecessor completion dependencies.
  Publication inserts the unique interior pointer in the native live ledger.
- A fit is selected natively by device and available span size, not module
  name, model architecture or runner hints. Splitting yields a leased span and
  a residual span; both inherit any predecessor fence covering their bytes.
- A cache miss creates only the minimum necessary mapping using the existing
  pressure/admission/retry path. Do not add larger speculative slabs, increase
  capacities, or evict live spans to compact memory.
- Lease destruction retires that span, not the backing owner. Releasing an
  allocation cannot invalidate any neighboring pointer or its queued work.

## Completion and transitions

Native states: available, reserved/unpublished, live, pending completion and
quarantined. State/generation checks reject duplicate free and stale tokens.

1. Reserve a suitable span transactionally. On failure, leave partition and
   ownership intact; do not publish a partially initialized lease.
2. Queue all inherited dependencies on the consuming stream. Preserve proven
   stream identity semantics; unknown identity requires the event dependency.
   Do not synchronize the host simply to make a span reusable.
3. Publish only after dependency submission succeeds. Keep predecessor event
   owners until publication/failure cleanup is safe. A failure does not return
   the span to an unguarded free list.
4. Free records a new consumer completion fence before returning the span to
   stream-ordered reuse. The new fence covers predecessor waits and consumer
   work. Preserve the bridge's existing caller stream-lifetime contract; do not
   infer that arbitrary additional tensor streams are tracked automatically.
5. Coalesce only adjacent spans of the same backing/generation whose completion
   dependencies are both proven complete, or whose identical retained fence
   safely covers the merged range. Never discard distinct pending fences to
   obtain a larger fit. Query-not-ready is not a failure or completion proof.
6. Event creation/record/query/wait failure retains affected ownership in a
   quarantined state. Mapping release requires positive completion proof for
   every span, not merely live count zero or a successful upload event.

Keep blocking destruction/driver calls outside the ledger lock. Pressure may
release only fully idle, completed backing owners. If all idle owners are
pending, use the existing explicit OOM/recovery semantics; do not invent a
background releaser or hide unbounded retired bytes outside accounting.

## Accounting and report compatibility

Count physical mapped backing once, including partially used owners. Separately
report requested bytes, aligned leased bytes, reusable span bytes, pending/
quarantined span bytes, and whole-idle backing retained in cache. Partition
bytes must sum to backing mapped bytes; requested bytes must not exceed their
leased spans. Physical mapping and cache admission must remain under existing
guards; splitting must not change physical accounting or registration ceilings.

The observer's v1 sum of per-request minimum VMM mappings is not physical
consumption for pooled leases. Do not reuse its field meanings or claim the
entire minimum-map sum was recovered. Introduce a versioned native pool report
and document its scope before integration; old DLLs omit unavailable metrics.
The public DDO report consumes these observations without runner internals.
Define treatment of legacy standalone records explicitly before switching
bindings, so a mixed allocator cannot silently double-count shared backing.

## Implementation and consolidated validation

`native/include/ddo_vmm_span_ledger.h` now implements the externally serialized
one-backing metadata ledger. It retains shared fence owners (no CUDA API calls),
best-fit splits available/pending spans, validates owner/generation/lease tokens,
and supports publication, unsubmitted rollback, guarded retirement, completion
observations and fail-closed quarantine. Coalescing preserves distinct pending
dependencies. Snapshots conserve one backing's partition instead of multiplying
physical mappings by the number of small leases.

Backing release is two-phase: begin blocks new reservations; actual destruction
will run outside the ledger lock; success commits once, failure quarantines the
backing. This handshake does not itself destroy a CUDA mapping. Successful
consumer retirement requires the caller to supply a fence covering all submitted
predecessor waits and consumer work; the CPU ledger cannot prove GPU submission.

Standalone `ddo_vmm_span_ledger_test` compiled with MSVC/C++20 and passed 1,270
checks, including repeated conservation checks, split/rollback, distinct-fence
non-coalescing, stale/foreign tokens, publication/query/retirement failures,
strong fence retention and pending/failed backing release. Existing footprint
and stream-dependency CPU executables also passed. No DLL was rebuilt or replaced
for this header-only, not-yet-integrated ledger. Protected and exact-cache DLL
hashes remain `4AC3D8B...` and `19AE9C2A...` respectively. No CUDA tests executed.

The isolated `build-span-pool-20261002` now compiles the connected candidate
(`DDO_VMM_NATIVE_SPAN_POOL=ON`, default OFF for existing builds). No new runtime
flag or Python residency policy: its PyTorch allocator entry points consume the
native pool. Protected binaries are not replaced. `ddo_vmm_torch_pool.cpp`
provides mapped backing and CUDA-event owners, context/device/stream validation,
transactional pointer publication, pressure/OOM recovery and whole-idle cache
retirement under the existing capacity setting. Cached spans inherit event
owners and enqueue consumer-stream dependencies without host synchronization.
Host completion waits occur only on whole-backing retirement, outside the
ledger lock, not on normal span reuse. This still needs performance validation.

Failed event/publication/retirement paths retain ownership; they do not switch
to a second allocator after partial submission. The original whole-region
route remains available in the existing build mode/protected binary, not an
automatic mid-process migration fallback. Do not promote without testing native
failure contracts and verifying the existing caller-level fallback route.

Native pool report schema v1 counts mapped backing once, partitions logical
leases/free/pending/quarantined bytes, and tracks retirement waits/failures.
The binding selects this schema instead of silently reinterpreting standalone
footprint or release counters. Public DDO JSON/profile integration and the
metric catalog are updated; older DLLs remain supported.

Current ledger test PASS: 1,278 checks; stream-dependency CPU test PASS. Selected
Python CPU telemetry/probe-option/profile suite: 32 passed with the candidate
and 32 with the protected DLL (no CUDA initialization). Read-only actual pool
export and invalid capacities/null pointers are included. Candidate SHA256
`8CAAD072B45302426EE8854591D9AB0AC893241D06A027FA0A6D8FB6B766021C`.
After those CPU checks, `vmm_span_pool_probe.py` and the two named allocator
gates were executed with approval and passed. One approved unchanged model
case then completed with identical PNG and no failures/quarantine, warm
1.387751 s/it, Generate 24.9113 s and total 45.7932 s. During denoise eviction
attempts/wait counters stayed at 53 / 0.1302295 s. Detailed partition/counters
are in the sidecar. Interior-pointer safety was exercised in these cases;
injected driver errors, broader pressure recovery and repeatability remain
unvalidated. No further run/default promotion authorized by that approval.

First implement a pure C++ span/lease ledger with mocked completion outcomes.
Cover partition conservation, alignment/overflow, best-fit splitting, device
isolation, generations, failed publication, multiple inherited fences,
coalescing eligibility, quarantine and single backing release.

Then connect backing/event owners and versioned accounting to the native
allocator in an isolated build. Preserve the old whole-region path as recovery
until the new route passes its gates; never switch the protected binary merely
because compilation or mocked tests pass.

Request one consolidated CUDA sequence after implementation: small allocation/
interior-pointer/alignment tests, adjacent-slice noninterference, cross-stream
pending reuse, failed-publication retention, last-slice backing retirement and
pressure/capacity accounting. Only if all pass, run the named unchanged eight-
step model case. Stop at first failure/memory anomaly. Require image equality,
consistent unique-backing accounting, no new fallback/failure, and substantially
less eviction/wait churn than exact-size rejection. Compare warm steps with the
protected 1.712871-second observation; no promotion based solely on memory saved
or total pipeline time. Performance claims need comparable evidence.
