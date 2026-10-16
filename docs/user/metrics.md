# Metrics reference

## Read a report

Use the public result returned by DDO instead of reading hook/planner attributes:

```python
import json

# result is returned by enable_offload(...).
report = result.metrics_snapshot()
print(report.format_profile())
print(json.dumps(report.as_dict(), indent=2))
```

The report includes its schema version and executed route. Fields depend on
the backend and enabled diagnostics; missing measurements are unavailable,
not zero. A snapshot does not by itself wait for pending CUDA work.

Model runners expose display/save controls in
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).
The catalog below defines units, collection boundaries and diagnostic meaning.

## SDNQ CPU preparation

The public `DynamicOffloadMetrics` report includes preparation accounting under
`runtime.state.sdnq_parameter_pinning` and capacity decisions under
`runtime.state.planner_decisions`. Runners render `format_profile()` or serialize
`as_dict()`; they do not inspect hook attributes. These fields apply to DDO's
`sdnq_runtime`, not official Diffusers leaf offload or native VMM. Older reports may lack already-CPU preparation accounting, so zero counts do not
prove that CPU source tensors were absent.

| Field | Unit / collection | Meaning / diagnostic action |
| --- | --- | --- |
| `sdnq_parameter_pinning.pinned_count`, `unpinned_count` | unique Parameter objects / attachment | CPU parameters after preparation; shared object identities count once, storage aliases are not deduplicated. |
| `sdnq_parameter_pinning.pinned_gb`, `unpinned_gb` | GiB / attachment | Packed tensor payload including scales/auxiliaries; despite the legacy `_gb` suffix these use 1024 cubed. Not GPU residency or a RAM peak. |
| `sdnq_candidate_parameter_bytes` | bytes / post-resident selection | Packed parameters eligible for CPU offload, excluding skipped/resident modules; not the entire checkpoint. |
| `sdnq_pin_parameter_budget_bytes` | bytes / pin selection | Resolved ceiling for new plus existing pins. Compare actual setup pins against this ceiling, not requested settings alone. |
| `sdnq_existing_pinned_parameter_bytes` | bytes / before new pins | Loader-owned CPU pins consuming the ceiling. DDO does not unpin them automatically. |
| `sdnq_existing_pinned_over_budget_bytes` | bytes / pin selection | Existing pins above the resolved ceiling; no new pins are admitted against that excess. |

The last four fields live in `planner_decisions`. Automatic planning uses the
same collector before resident selection, so its earlier `candidate_weight_gb`
can exceed the later offloaded candidate population. Byte ceilings and known RAM
admission remain separate from GPU workload residency profiling. Unsupported or
unmeasured tensor-subclass physical storage must not be inferred from logical
shape bytes. Setup pinning is not evidence of transfer overlap or speedup.

## Experimental component ownership (selected Windows/Linux workloads validated)

`DynamicOffloadApplyResult.metrics_snapshot()` exposes `runtime.component_ownership`
only for the opt-in component adapter. `format_profile()` marks it experimental;
normal runners do not inspect bindings/owners. Values are attachment metadata,
not GPU/RAM use or proof of numerical support.

The selected component contracts, real text-only encoder and staged pipeline
passed their Windows gates. Linux passed selected contracts and one staged
real-model generation; coverage is not identical and is not general encoder or
multimodal support. See [current usage and support](native-vmm.md).

| Field | Unit / collection | Meaning / action |
| --- | --- | --- |
| `schema_version` | version / attachment | Component metadata schema1 inside the versioned DDO report. |
| `parameter_groups` | identity groups / attachment | Unique native parameter bindings; shared embedding/output owners count once. |
| `buffer_groups` | identity groups / attachment | Separate device buffers with alias-preserving placement, not file-backed weights. |
| `payload_bytes` | bytes / layout | Aligned native parameter address span, not physical VRAM residency. |
| `buffer_bytes` | bytes / preflight | Logical once-copied buffer payload charged against the explicit buffer ceiling, not allocator physical overhead/peak. |
| `execution_units` | units / attachment | Disjoint fault/unpin scopes, not architecture detection or guaranteed forward order. |
| `lookahead` | boolean / attachment | False in the initial experimental component contract. Default validated Linear lookahead is unchanged. |

## Loader source association

`from_pretrained_with_dynamic_offload(...).metrics_snapshot()` returns a detached
public report, route `model_load`, with `setup.weight_source`. This reports the
metadata association after loading, not native attach/execution or measured
VRAM/RAM. It does not query CUDA. Use `format_profile()` or save `as_dict()`;
hosts do not need private registry attributes.

| Field under `setup.weight_source` | Unit / collection | Interpretation / action |
| --- | --- | --- |
| `bound` | unique pointer/byte-length groups / loader setup | Published CPU storage descriptors; shared Parameter aliases count once. Not full component coverage or selected module count. |
| `available` | keys / header discovery | Valid checkpoint keys before loader renaming. Duplicate shard keys cause the whole set to be refused (zero available). |
| `resolver` | label / setup | `exact_names` or `transformers_name_mapping` for supported name resolution; `unsupported_loader_mapping`, `unsupported_payload_transform`, `unsupported_quantized_payload`, `no_valid_source_metadata` indicate why association was refused. Preserve compatibility route; do not guess offsets. |
| `rejected_storage_groups` | groups / setup | Conflicting aliases/mapped sources or incompatible shape/dtype/bytes; investigate provenance, not cache capacity. No first-alias partial publication. |
| `missing_storage_groups` | groups / setup | Eligible CPU storage with no resolved source candidate. Check checkpoint coverage and loader renaming. |
| `ineligible_parameter_aliases` | names / setup | Non-CPU, noncontiguous or empty parameter aliases; no meta storage addresses are queried. |
| `header.invalid_files`, `header.invalid_entries` | counts / discovery | Invalid headers or malformed/out-of-file ranges/metadata were skipped. Inspect local checkpoint integrity. |
| `header.ambiguous_key_count` | keys / discovery | Duplicate conflicting keys across shards; entire source set rejected, independent of iteration order. |
| `diagnostic` | optional text / setup | Unsupported installed mapping API diagnostic; not a driver fault or an inference failure. |

Unknown non-Transformers loader mappings are refused, not emulated. Transformers
payload converters are conservatively refused even if a converter may be unused
by this checkpoint. Shape/dtype validation protects casts with equal byte sizes.
Registry ownership stays in DDO; source files are opened later by native transport.
No native encoder support is implied by a successful loader metadata report.

## Metadata-only component capacity inspection

`inspect_vmm_component(module, component="text_encoder")` returns the public
`DynamicOffloadMetrics` report, route `vmm_capacity_inspection`. It uses the
allocator's exact attachment selector without CUDA, DLL loading, tensor copies,
source binding or profile-file writes. CPU or `meta` modules are sufficient.
Selection is not execution support, especially for custom/quantized forwards.
The allocator package is optional and imported only when inspection is requested.

```python
report = inspect_vmm_component(component_model, component="text_encoder")
print(report.format_profile())
payload = report.as_dict()  # JSON-ready; no module/tensor owners retained
```

All fields below are under `setup.capacity_inspection`, inspection schema1.
They are collected once at inspection, not runtime VRAM/RAM or a residency budget.

| Field | Unit | Meaning / diagnostic action |
| --- | --- | --- |
| `selected_module_count`, `selected_modules` | count / metadata list | Exact dense-Linear selection, class and enclosing execution unit. Verify pattern, not model architecture heuristics. |
| `selected_aligned_payload_bytes` | bytes | Attachment-layout payload including alignment and each selected binding; not physical backing or source-cache capacity. |
| `unique_parameter_bytes` | bytes | Parameters deduplicated by object identity; different Parameters sharing storage are counted separately. |
| `selected_unique_parameter_bytes` | bytes | Identity-unique parameters with at least one selected owner. Does not imply all owners are covered. |
| `nonselected_unique_parameter_bytes` | bytes | Parameters with nonselected owners. Requires placement/lifecycle work before full-component support. |
| `shared_selected_nonselected_bytes` | bytes | Intersection of the previous two sets. Do not add selected/nonselected totals without subtracting this overlap; shared ownership needs explicit handling. |
| `parameters` | metadata list | Alias names, selected/nonselected owners, shape, dtype and bytes. No tensors, pointers, loaded-source binding or storage-alias inference. |
| `unique_buffer_bytes`, `buffers` | bytes / metadata list | Identity-unique named buffers and their metadata; placement and reconstructible/nonpersistent buffers require separate handling. |

`format_profile()` prints capacity in GiB and explicitly labels it metadata-only.
Checkpoint name/shape availability alone does not prove immutable native source
binding: loader renaming and payload transforms must be distinguished first.

## Native span-pool report (candidate; schema v1)

Optional `ddo_vmm_torch_pool_stats_v1` reports 15 uint64 values; status 2 means
unsupported build. Python exposes `fast_torch_allocator_pool_*`, with
`schema_version=1` and `enabled=1`. No CUDA calls occur during collection.
Do not combine with standalone per-request footprint or old release scopes;
the binding omits those when pool reporting is active. Ordinary/old DLLs retain
their previous report behavior. Compiled/CPU-tested is not a CUDA/performance gate.

| Suffix after `fast_torch_allocator_pool_` | Unit / collection | Meaning / action |
| --- | --- | --- |
| `mapped_bytes` | bytes / ledger snapshot | Physical backing size counted once across all leases; not WDDM-local usage or GPU total. |
| `requested_bytes` | bytes / snapshot | Raw requests of live/reserved leases. Compare with leased bytes, not per-request VMM minimum sizes. |
| `leased_bytes` | bytes / snapshot | Logical aligned live/reserved spans. Padding is leased minus requested. |
| `available_bytes` | bytes / snapshot | Unleased spans without pending completion; reuse still requires context/device admission. |
| `pending_bytes` | bytes / snapshot | Unleased spans with retained completion dependencies not yet reaped, plus backing being retired. A completed GPU event can still have pending metadata; not a measurement of outstanding GPU work or safely unmappable bytes. Snapshot does not query/reap events. |
| `quarantined_bytes` | bytes / snapshot | Ownership retained after failure; stop the gate and investigate. Never treat as reusable. |
| `live_allocation_count`, `reserved_allocation_count` | leases / snapshot | Published or unpublished logical spans; neither is a count of physical mappings. |
| `backing_count` | mappings / snapshot | Physical backing owners retained in the native pool. |
| `idle_backing_bytes`, `idle_backing_count` | bytes, mappings / snapshot | Whole backing without live/reserved/quarantined spans; completion can still be pending. Idle backing retention uses the existing cache ceiling. Partial backing remains charged in mapped bytes, not duplicated as cache. |
| `eviction_attempts` | attempts / process lifetime | Whole-idle backing retirement attempts. Compare interval deltas with prior exact-size churn. |
| `eviction_wait_seconds` | seconds / process lifetime | Host wall time in completion proof for eviction (C ABI nanoseconds converted to seconds). Not GPU engine execution or bandwidth. |
| `failures` | count / process lifetime | Context, event, admission/publication exception or retirement failure observations; nonzero blocks validation. |

Snapshot conservation: mapped = leased + available + pending + quarantined,
requested <= leased. Gauge partition is collected under one native ledger lock;
process counters can change independently outside that lock. Excludes model
arena/execution slots, external allocations and GPU driver overhead. Profile
prints GiB; JSON keeps byte integers. No additional runner metric-field handling.

## Optional native allocation-footprint snapshot

VMM-fast exposes `fast_torch_allocator_footprint_*` through the public report,
when the selected DLL supports `ddo_vmm_torch_allocation_footprint_v1`.
Older DLLs omit these fields, rather than reporting zero. Collection takes one
native ledger lock, without CUDA calls, synchronization or allocation-policy changes.

| Suffix after `fast_torch_allocator_footprint_` | Unit | Interpretation / diagnostic action |
| --- | --- | --- |
| `requested_bytes` | bytes | Sum of current live PyTorch requests, including the new request when backing is reused. Compare with mapped bytes. |
| `minimum_mapped_bytes` | bytes | Sum of minimum mapping sizes for current requests after allocator normalization and device-granularity rounding. |
| `mapped_bytes` | bytes | Actual mapping sizes of the same live records; excludes cached blocks. |
| `granularity_padding_bytes` | bytes | Minimum mapped minus requested; quantify rounding before considering native suballocation. Includes zero-request normalization. |
| `reuse_excess_bytes` | bytes | Mapped minus minimum mapped; quantify oversized cache reuse before proposing cache-selection changes. |
| `allocation_count` | records | Live records included in this atomic decomposition, not tensors or model parameters. |

All six values belong to one snapshot: requested + padding + reuse excess =
mapped. The separately collected legacy live/cache gauges need not match its
instant. Excludes model arenas, native execution slots, external allocations and
records retired outside the ledger; this is neither GPU total nor peak. Excess
is not automatically safely recoverable memory. `--show-profile` renders
requested/padding/reuse excess in GiB; JSON retains integer bytes. Post-phase
snapshots do not characterize transient forward peaks.

## Diagnostic compute-step boundary wait (not a benchmark)

`result.execution_timer(enabled=True, native_units=True,
diagnostic_step_wait=True)` is a causal submission-cadence control for an active
VMM-fast component, never a preset or official residency/transfer policy. Each
`mark_step()` waits only for its newly recorded compute event. No CUDA scalar
read, device synchronization or upload-stream drain is added. Default is false.
Host harness flag: `--ddo-diagnostic-step-wait`, requiring native/GPU timing.

Public `runtime.execution_timing.diagnostic_step_wait` fields: `enabled`,
`scope=compute_boundary_event_host_wait`, `calls` (attempted boundary waits),
`host_seconds` (wall time in those waits), `benchmark_eligible=false`. Collected
at timer snapshot; console warns “not a benchmark”. This measures changed
cadence, not saved GPU work. Compare engine overlap and launch backpressure in
otherwise identical diagnostic traces, not benchmark rankings. A negative
effect is a valid hypothesis result. Default timing retains no step waits.

## Optional native stream-window timeline (experimental)

```python
with result.execution_timer(enabled=True, native_units=True) as timing:
    # Existing step callback calls timing.mark_step().
    output = pipeline(...)
# Existing blocking readback/completion boundary, not a new per-step wait.
report = result.metrics_snapshot(collect_completed=True, execution_timing=timing)
tracker.record_ddo_metrics(report)
```

Requires an active `vmm_fast_runtime` component and a DLL with the optional
timeline-v1 exports. Older DLLs and protected routes remain usable without this
option; requesting it on an unsupported route fails explicitly. Host benchmark
flag: `--ddo-native-timeline`, requiring `--ddo-gpu-timing` and the public native
route. Both instrumentation and flag default off. Enabling them is not performance acceptance.

`runtime.native_timeline` is a separate versioned report. Capacity defaults to
4096 samples (`native_capacity` accepts integers 1..8192), preallocated as two
timing events per sample plus an epoch. No automatic growth or changes to GPU
weight/host source ceilings. Setup/cleanup overhead belongs to Generate/total;
measure a control run before treating instrumented timing as representative.
This pool is not the source-cache lifetime-event pool.

| Sample kind | Unit | Collection boundary / interpretation |
| --- | --- | --- |
| `upload_batch_window` | seconds | GPU markers at batch submission boundaries. Includes source preparation/host launch gaps; not pure DMA duration. |
| `compute_upload_dependency` | seconds | GPU markers surrounding the existing upload wait on compute. Includes any launch gap between markers; not CPU API cost. |
| `upload_slot_dependency` | seconds | GPU markers surrounding the existing last-consumer wait on upload. Indicates slot-recycle dependency, not eviction. |
| `execution_unit_window` | seconds | Markers around a generic unit forward after dependency/view setup and before cleanup/unpin. Includes kernel-launch gaps; not summed kernel duration. |
| `step_marker` | seconds | Native-clock boundary markers: unit 0 start, 1..N steps, UINT64_MAX end. Separate nearby markers from the public basic timer, not the same events. |

Samples carry `sequence`, `unit`, `stream`, `bytes`, `kind`, and `status`.
Complete samples alone expose signed `start_seconds`/`end_seconds` relative to
the CUDA epoch in the same context; negative offsets are preserved. `bytes=null`
means uncollected, not zero payload. Upload/dependency units are run-local native
identities; execution units have a name catalog. Do not compare native pointers,
unit IDs or stream handles as stable identities across runs.

`status` is complete/pending/truncated/failed; `capacity`, `dropped`, `failures`
and `pending` show observation limits. No duration zero is invented for pending
or failed work. Snapshot queries completion and computes elapsed time only for
ready events, with no synchronization. Only complete reports receive
`window_summary` and, with first-step/end markers, `warm_window_summary`.
Their `union_seconds` and `upload_execution_window_intersection_seconds` reduce
**stream windows**, not GPU engine activity. They cannot establish exact DMA/
kernel overlap or infer the percentage of hardware utilization. Native failure
counts and failed sample statuses are complementary; inspect both.

After sealing, complete events may be released nonblockingly (`event_cleanup`).
Pending/failed owners remain native-held, without dropping weight lifetimes.
An abandoned retained owner blocks a new diagnostic session in that context
until closed successfully or process exit. There is at most one owner/session
per context; concurrent inference in a shared context is not validated and may
contribute other native operations to this context-scoped observation.
No timing event replaces a lifetime event, is re-recorded for a later sample,
or introduces a cross-stream clock-alignment wait.

## Native temporary-allocator release observations

Available in `runtime.vmm` on the experimental fast route when the optional
native `ddo_vmm_torch_release_stats_v1` export is present. Older DLLs omit these
fields: absence is not zero. Snapshots read native atomics without CUDA queries,
waits, resets or policy changes. Values accumulate over the **process lifetime**,
including component loading and text encoding; use differences between snapshots
to isolate a phase. Concurrent snapshots are approximate, not a transaction.

All names below have prefix `fast_torch_allocator_`.

Four optional **gauges**, unlike the cumulative release counters below, are
read from existing native exports at the report snapshot. No CUDA query, wait,
allocator reset or new policy is added. They cover the process-wide PyTorch
pluggable allocator, not only the component reported by DDO.

| Suffix | Unit / collection | Meaning / diagnostic use |
| --- | --- | --- |
| `live_allocation_count` | allocations / snapshot | Records currently owned by live PyTorch allocations. Includes other pipeline components. |
| `live_mapped_bytes` | bytes / snapshot | Mapped size of those live records, including allocation rounding and reused oversized backing; not logical tensor size or peak. |
| `cached_allocation_count` | allocations / snapshot | Records currently retained in the native PyTorch reuse cache. |
| `cached_mapped_bytes` | bytes / snapshot | Mapped backing held in that cache. It may still have pending compute fences; a cache entry is not necessarily complete. |

Each gauge is read under an independent native ledger lock, so concurrent
values are not an atomic total. Model-weight arenas, execution slots, CUDA
context/workspaces not allocated through PyTorch, and records removed from
the ledger but still draining/retained after a failed release are excluded.
Do not subtract this snapshot from a GPU peak to infer free capacity, or add
it to a different-time weight snapshot as total physical GPU usage. Missing
exports omit the corresponding gauge rather than reporting zero.
`format_profile()` displays available mapped gauges as GiB with their scope;
JSON preserves byte values. Runners need no native attribute knowledge.

| Suffix | Unit | Meaning / diagnostic use |
| --- | --- | --- |
| `reset_sync_calls` | calls | Existing completion-event synchronization calls during allocation release; includes events already ready. |
| `reset_sync_seconds` | seconds | Host time inside those calls, not GPU kernel duration or necessarily extra idle time. Compare phase deltas. |
| `reset_failures` | failures | Completion proof failed or was already unproven; backing retained instead of unsafe unmapping. Any increase fails the safety gate. |
| `cache_eviction_attempts` | allocations | Cached records entering release, through capacity eviction or pressure/OOM drain. Does not imply successful unmapping. |
| `cache_eviction_attempted_bytes` | bytes | Mapped bytes in those attempts; includes failed completion proofs. Not current memory usage. |
| `cache_eviction_sync_seconds` | seconds | Subset of reset synchronization host time for cache eviction. Not additive to `reset_sync_seconds`. |

Collected when the public DDO report snapshots the active native component.
The profile formatter displays eviction synchronization time and attempt count;
JSON retains all fields. No runner-specific attribute access is required.
These observations do not measure event-record fallback stream synchronization,
unmap/free duration, event pooling, or compute-stream dependency wait duration.

## Optional compute-stream timing and completion collection

```python
with result.execution_timer(enabled=gpu_timing) as timing:
    # In the existing step callback: timing.mark_step()
    output = pipeline(...)
# Use the host's existing blocking readback/completion boundary here.
report = result.metrics_snapshot(
    collect_completed=gpu_timing, execution_timing=timing,
)
tracker.record_ddo_metrics(report)
```

Both options default off; existing reports and execution are unchanged.
Timing does not synchronize at step boundaries. `snapshot()` checks the final
event without waiting; pending timings omit durations rather than invent zeros.
An explicit `snapshot(wait=True)` waits for the final event only, for diagnostics.
Do not use it in step callbacks. The stream is captured when entering the timer;
work on unrelated streams is outside its scope unless dependencies join them.

| Field under `runtime` | Unit | Collection / backend | Interpretation and diagnostic action |
| --- | --- | --- | --- |
| `execution_timing.schema_version` | version | Timer report, any CUDA route | Timing sub-schema 1; additive to report schema 1. |
| `execution_timing.status` | label | Query after phase | `disabled`, `not_finished`, `pending`, `complete`; pending is not zero duration. Resolve after an existing completion boundary. |
| `execution_timing.phase_seconds` | s | Completed final compute-stream event | Start-to-end marker, including stream dependencies and host launch gaps; not kernel-only duration or proof of overlap. |
| `execution_timing.step_seconds` | s per step | Completed callback markers | Consecutive compute-stream intervals; first includes cold work. All callbacks must mark the same phase's steps. |
| `execution_timing.steady_state_seconds_per_step` | s/step | Mean excluding first marker interval | Defined only with at least two marked steps; compare equal workloads/settings. This is not iterations/s. |
| `execution_timing.tail_seconds` | s | Last callback marker to phase-end marker | Work not covered by callbacks; do not lose it from the phase total. |
| `execution_timing.stream`, `scope`, `clock`, `failed` | mixed | Timer report | CUDA stream handle (0 means default), not a profiler's stream ID; scope and failed execution. Do not promote timings from a failed run. |
| `completion_collection.tickets_before`, `tickets_after`, `tickets_retired` | count | Explicit phase boundary, native retained-source route | Native query/reap of batch-owned ready tickets only. After completed readback, remaining tickets need investigation; external owners are never stolen. |
| `completion_collection.blocking` | bool | Collection report | False: pending DMA is preserved, never waited on. Completed ticket close may use its already-ready event as a safety proof. |

Ordinary snapshots remain read-only: they do not reap or wait. Collection does
not change storage/registration/GPU budgets and does not claim full post-close
ledger validation. The optional native collection export requires a rebuilt
allocator; an old DLL remains usable unless this option is explicitly requested.
`fast_source_cache_events_created` is cumulative event creation, not a live-event
gauge. The native pool retains at most 256 idle ticket events; that is not an
in-flight limit. Never treat fewer snapshot tickets as proof of faster DMA.

DDO owns the telemetry schema. Hosts should obtain a snapshot from an
`enable_offload(...)` or `enable_dynamic_offload(...)` result instead of reading
hook fields or VMM implementation attributes.

```python
result = enable_offload(transformer, preset="auto", component="transformer")
report = result.metrics_snapshot()

if show_profile:
    print(report.format_profile())
if save_metrics:
    json.dump(report.as_dict(), output_file, indent=2)
```

`DynamicOffloadMetrics.as_dict()` has schema version `1` and contains:

| Field | Unit | Meaning |
| --- | --- | --- |
| `configuration` | mixed | Resolved DDO settings for the component. |
| `setup` | seconds/bytes | Work performed while the route was applied. |
| `runtime.state.setup_runtime` | seconds, GiB | Per-action setup work collected by DDO. |
| `runtime.state.copy_runtime` | calls, seconds, GiB | Per-copy-path transfer activity. |
| `runtime.state.planner_decisions` | mixed | Planner explanation; use for diagnosis, not host control flow. |
| `runtime.vmm` | bytes/counts/seconds | Present only for a VMM hook; native arena residency, materialization, fallback and transfer statistics. |

For `vmm_fast_runtime`, VMM-native counters are prefixed with `fast_`. The
transport fields have explicit scope: `fast_shard_read_*` measures synchronous
host reads from the mapped/overlapped shard into a bounded staging slot;
`fast_staging_source_copy_*` measures CPU-source to slot copies when used;
`fast_dma_enqueue_*` measures host time spent issuing the device transfer; and
`fast_staging_slot_wait_*` measures only waiting to reuse a still-busy slot.
`fast_compute_wait_events` counts stream dependencies queued before compute; it
is not a host-side wait. These are not end-to-end CUDA durations; use a
benchmark trace for that.

### Native pipeline (source ABI v11 / stats v8)

All fields below are observations in `runtime.vmm` snapshots, available through
`DynamicOffloadMetrics`. Hosts do not read native state to render them.
The source-cache fields carry the prefix `fast_source_cache_`.

| Suffix | Unit / collection | Interpretation and diagnostic action |
| --- | --- | --- |
| `request_total_seconds` | host seconds / source request calls | Entire valid request scope, including scheduling and lock acquisition. Compare with source readiness and misses. |
| `copy_total_seconds` | host seconds / direct ticket copy calls | Includes readiness, event acquisition, registration and enqueue. Not device DMA duration; compare nested scopes, do not sum them. |
| `close_total_seconds` | host seconds / ticket close calls | Wait, event retirement and bookkeeping. Inspect outstanding tickets and churn when high. |
| `event_acquire_seconds`, `event_retire_seconds` | host seconds / event-pool operations | Creation/reuse and recycle/destruction overhead. Compare event counts to uploads. |
| `registration_schedule_seconds` | host seconds / registration admission | Registration scheduling, recycling and nested registration/wait cost. Inspect registration churn rather than assume file rereads. |
| `batch_submit_seconds` | host seconds / native unit submission | Full native submission scope, including reap, requests, copies and bounded ring fallback. Compare with the broader `fast_dma_enqueue_seconds` scope. |
| `batch_reap_seconds` | host seconds / ticket retirement scans | Includes queries and ticket close; close-time draining can wait. Inspect with `close_total_seconds`. |
| `events_created`, `events_reused` | count / ticket event acquisition | Reuse occurs only after completion. These exclude persistent per-unit completion and execution-slot events. |
| `batch_submissions` | count / successful unit submission | Does not mean GPU completion. Compare with execution units; failed partial submission is excluded. |
| `batch_bypass_bytes` | bytes / uncached ring fallback submitted | Bytes sourced outside retained storage; unlike registered-source fallback, these can require file reads. Compare bypasses and source-cache capacity. |

`fast_execution_buffer_slots` is the native slot count;
`fast_execution_buffer_slot_bytes` is each slot's page-rounded capacity.
`fast_execution_buffer_capacity_bytes` is aggregate slot capacity (not per-slot).
`fast_execution_buffer_resident_bytes` reports materialized pool backing;
`fast_total_weight_resident_bytes` adds model-arena backing. Compare the latter
with the configured GPU ceiling; neither is total process VRAM or WDDM shared
residency. Snapshots do not synchronize GPU work.

These counters do not establish full-model transfer/compute overlap.

### Upload-boundary scopes (Python adapter, unchanged native ABI)

These additional host scopes partition adapter upload work. They add no CUDA
events, waits, transfers or stream changes beyond the existing context.

| Fields in `runtime.vmm` | Unit / collection | Meaning and diagnostic action |
| --- | --- | --- |
| `fast_native_transport_batch_transport_total_seconds` | host seconds / persistent batch dispatch | Full transport lookup, one-time creation and Python submit wrapper. Compare with broad upload, then partition into prepare/submit. |
| `fast_native_transport_batch_transport_prepare_seconds` | host seconds / dispatch preparation | Key/lookup, immutable source descriptors and creation on cache miss. Includes the create scope; do not add it twice. |
| `fast_native_transport_batch_transport_create_seconds` | host seconds / first creation of a unit batch | Includes source opening, metadata and the native create wrapper; subsequent reuse does not increment it. Compare cold and reused batches. |
| `fast_native_transport_batch_transport_submit_seconds` | host seconds / submit wrapper | Compare with bridge total and native submit; not device DMA time. |
| `fast_source_cache_batch_bridge_create_total_seconds`, `fast_source_cache_batch_bridge_create_ffi_seconds` | host seconds / batch construction | Total wrapper scope includes cache/source locks and descriptor conversion; FFI scope covers only the DLL create call and its boundary. Their difference is not a pure Python CPU measurement. |
| `fast_source_cache_batch_bridge_submit_total_seconds` | host seconds / submit wrapper | Includes lock acquisition, arguments, FFI and result handling. Captures failed submissions as well. |
| `fast_source_cache_batch_bridge_submit_lock_wait_seconds` | host seconds / lock acquisition | Python cache lock contention before submit; no CUDA wait is introduced. Compare with concurrent host callers. |
| `fast_source_cache_batch_bridge_submit_prepare_seconds` | host seconds / submit arguments | Destination ownership, handle/stream extraction and validation; retained ownership is unchanged. |
| `fast_source_cache_batch_bridge_submit_ffi_seconds` | host seconds / DLL submit call | Includes ctypes entry/return and potential GIL rescheduling outside the C++ timer. Compare with native `batch_submit_seconds` and matching call counts; do not attribute the difference without evidence. |
| `fast_source_cache_batch_bridge_submit_result_seconds` | host seconds / result handling | Status checking after return; errors remain errors, with ownership retained. |
| `fast_upload_stream_enter_seconds`, `fast_upload_stream_body_seconds`, `fast_upload_stream_exit_seconds` | host seconds / existing admitted-model-arena upload context | Separate PyTorch stream-context entry, transport body and restoration. Execution-pool uploads explicitly pass the stream and do not use this context. The body nests transport scopes; entry/exit are not DMA durations. |

Each transport/bridge timed scope also provides its name with `_calls` in
place of `_seconds`. Lock-wait has no separate count: use submit-total calls.
`fast_upload_stream_scope_calls` counts admitted-arena contexts, not all native
batches. Failed operations contribute elapsed time/calls; successful native
`batch_submissions` therefore need not equal bridge calls in a failed run.

Reconcile in order: broad upload versus transport plus stream entry/exit;
transport prepare versus one-time creation; transport submit versus bridge;
bridge FFI versus native submit. Scopes are nested, not independent durations.
Small bookkeeping/clock overhead remains outside partitions; timestamps are
host wall time, not CPU utilization, device timeline or proof of overlap.

Measured diagnostic: transport total 67.2670 s; bridge FFI 67.2392 s versus
native submit 13.4883 s, with 232 calls in both. Create/prepare/lock/result and
stream entry/exit are small. The dominant residual is localized to the DLL-call
interval; native timing does not include ctypes entry/return or re-entry into
Python. Background USS/CUDA memory monitoring is a concrete interference
hypothesis, not a proven cause. See the recorded benchmark for memory/counter
parity and observer-comparison caveats.

### Host memory observation: explicit full/RSS mode

Fast VMM's `setup_runtime` additionally separates:

| Setup action | Unit / collection | Meaning and diagnosis |
| --- | --- | --- |
| `vmm_fast_plan_metadata` | host seconds / hook initialization | Plan/cache metadata creation, not GPU residency calibration. |
| `vmm_fast_native_resources` | host seconds / before arena attach | Validation, native loader/imports, optional budget guard and transport configuration. Does not include arena attachment or auxiliary copies. |
| `vmm_fast_auxiliary_linears_to_device` | host seconds and bytes / auxiliary placement | Existing synchronous transfers of linears not attached to the arena; byte count excludes tensors already on the execution device. Compare with attach and small-tensor timings before calling the remaining setup time Python overhead. |

These scopes are disjoint from the existing arena-attach/root/small-tensor
actions. They do not imply a new optimizer, altered copy placement or a speedup.
Earlier reports lack these fields; absence is not zero. No field was added to
linear/NVMe setup contracts. Overall setup can still contain validation/reporting
outside these scopes; do not assume all durations reconcile exactly.

Experimental native setup also reports `fast_view_prepare_calls`,
`fast_view_create_calls`, `fast_view_release_calls` (attempt counts), and the
corresponding `fast_view_prepare_seconds`, `fast_view_create_seconds`,
`fast_view_release_seconds` (host wall seconds, failed attempts included).
Collection occurs during attachment, not denoising: prepare/release cover the
native transient bootstrap mapping, create covers CUDA/DLPack view construction.
These are nested parts of native attach, not DMA or resident-weight time.
Grouped setup should reduce prepare/release attempts while preserving tensor
view count and offsets. Failed prepare can fall back to individual preparation;
failed release prevents successful attach. Missing fields in earlier reports
mean unavailable, not zero. These fields belong only to the experimental
native adapter and do not alter linear/NVMe metric contracts.

The sibling host `RunTracker` now accepts `memory_monitor="full"` (unchanged
default) or `"rss"`. The VMM harness exposes `--memory-monitor full|rss`.
Full mode keeps Windows USS (`memory_full_info().uss`); RSS mode uses the cheap
resident working-set counter (`memory_info().rss`), avoiding repeated full-memory
scans. On Linux the existing full mode uses anonymous RSS with USS fallback.
This is host observation only: allocator budgets, lifetimes, native telemetry,
GPU query method and 0.1 s polling cadence are unchanged. Monitoring is not
silently disabled or made faster by dropping the GPU peak observation.

RSS includes resident shared/mapped pages and is not USS. The legacy
`global_peak_ram_gb`/`peak_ram_gb` fields remain numeric GiB but their definition
is explicit in top-level `memory_monitor.ram_metric`, each phase's `ram_metric`
and console `Peak RAM (rss|uss)` labels. Do not directly compare RSS peaks
with previous USS peaks or interpret changing definitions as reduced RAM use.
Native committed/registered-byte counters and capacity guards are unaffected.

| Host field | Unit / collection | Interpretation |
| --- | --- | --- |
| `memory_monitor.mode`, `ram_metric` | enum / tracker setup | `full` preserves the legacy definition; `rss` selects working-set RAM. Labels are required when rendering peaks. |
| `memory_monitor.gpu_metric` | string / setup | Existing `cuda_mem_get_info_used_minus_baseline`; GPU observation is unchanged. |
| `memory_monitor.gpu_absolute_metric`, `memory_unit` | string / setup | `cuda_mem_get_info_used`, `GiB`; additive labels for absolute device observation. Not process-only or shared-WDDM usage. |
| `vram_baseline_gb` | GiB / tracker setup | Exact existing device-used query after CUDA initialization, before tracker timing starts. Do not substitute rounded console baseline when checking a memory gate. |
| `steps[].peak_vram_absolute_gb` | GiB / phase end | Absolute peak from the same samples as legacy delta `peak_vram_gb`; no added device queries. Compare absolute GPU limits against this field. |
| `global_peak_vram_absolute_gb` | GiB / tracker setup, sample and phase end | Maximum of baseline and completed/explicit samples. Not an instantaneous cap; includes other allocations visible through device-used queries. Legacy global delta retains its existing semantics. |
| `memory_monitor.interval_sec` | seconds / setup | Poll delay after the previous query completes, not guaranteed sampling period. |
| `steps[].memory_monitor.samples` | count / completed samples | Includes the initial sample and explicit tracker samples. A sampled peak is not an exact device timeline maximum. |
| `steps[].memory_monitor.ram_query_seconds`, `gpu_query_seconds` | accumulated host wall seconds / query calls | Cost of observation, including failed attempts; not CPU utilization or the exact delay imposed on inference. Can overlap inference. |
| `steps[].memory_monitor.error_type` | nullable string / phase finish | Failed background observation is surfaced as an error; incomplete peaks must not be reported as a passing benchmark. |

Missing absolute fields in a report mean unavailable, not zero. Keep baseline,
absolute driver usage and RAM measurement mode distinct when comparing reports.

The sibling native harness now saves `text_encoder_native_vmm` as the requested
opt-in. Existing `text_encoder_group_offload`/`text_encoder_dynamic_offload`
booleans are requested preset controls, not evidence of the selected native
override. The console labels them `_requested`; its route includes that override.
`text_encoder_offload_route` comes from the public apply result; the executed
backend and component ownership are in the component's public DDO report.
Do not infer executed support from a requested flag, preflight or partial run.

RSS and USS have different scope and sampling cost; do not compare their peaks
as interchangeable measurements. Monitoring overhead is reported separately
and can affect a workload. Model timing comparisons belong to
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).

The opt-in native retained-source micro route adds `fast_source_cache_*`
fields. `registered_bytes` and `process_staging_registered_bytes` are current
registered host bytes in the cache and ring; `process_limit_bytes` is their
combined cap. `retained_bytes` is useful cached source content, while `hits`,
`misses`, `bypasses`, `source_read_bytes`, and `dma_bytes` describe cache use.
These are native host-cache counters, not WDDM non-local VRAM usage or an
inference-speed result. If `bypasses` rise while `registered_bytes` stays at
capacity, inspect slot size and the aggregate host cap before changing GPU
residency. This experimental route is disabled by default. ABI v6 keeps local
cache capacity at 128 MiB and permits a separately configured ring-plus-cache
cap up to 1 GiB. Its CUDA compatibility and 768 MiB host-registration capacity
contracts passed; inference measurements remain a separate gate.

`fast_source_cache_process_registered_bytes` sums registered cache bytes across
live cache owners in this DLL. Add `process_staging_registered_bytes` to obtain
the tracked ring-plus-cache consumption. Both are instantaneous bytes sampled
at the runtime snapshot; registrations by other libraries are excluded.

When native admission is denied, `fast_fault_failures` gives the frequency;
`fast_guard_admission_failures`, `fast_cuda_admission_failures`, and
  `fast_capacity_admission_failures` identify the known cause; and
  `fast_temporary_tensors`/`fast_temporary_bytes` report the short-lived CUDA
  fallback volume.

`fast_capacity_event_waits` counts the narrow event waits used when all pages
inside the resident capacity are still protected by submitted CUDA work. Each
wait retires only the oldest protected module range, allowing the native LRU
to reuse its physical pages; it is not a device-wide synchronization. A rising
value with low fallback volume is expected for a capacity-constrained run. A
rising value together with high fallback volume indicates that the active
range is larger than the safely reusable resident set.

VMM `build_profile=True` does not tune a resident-module budget. It saves a
capacity observation for the component: native virtual/page/resident bytes,
weight-source mix, platform, fault/signature/upload activity, and fallback
volume. The transformer is the first validated component; the same contract
is intended for text encoders after its own CUDA validation.

The runner may merge this payload into its own run tracker, but it must not
depend on `DynamicOffloadHook.state`, `planner_decisions`, or VMM-specific
attributes. `format_profile()` is the supported compact display for
`--show-profile`.

For the experimental VMM fast backend, the only bootstrap required from a
runner is the first executable statement, before importing Torch or Diffusers:

```python
from ddo_vmm_bootstrap import ensure_vmm_fast_process

ensure_vmm_fast_process()
```

The parent transparently runs the same command in a child process after the
native allocator is installed. The child continues through the runner normally.

## Route arguments

Hosts add the standard arguments once and pass the resolved route to DDO. They
do not need a `DDO_BACKEND` environment variable or backend-specific branches:

```python
from diffusers_dynamic_offloader import add_dynamic_offload_arguments, resolve_dynamic_offload_backend

add_dynamic_offload_arguments(parser)
args = parser.parse_args()
route = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset)
config = route.build_config_from_args(args)
result = enable_offload(transformer, config=config, use_environment=False)
```

`linear/auto` remains `one_shot_fast`. `vmm/auto` selects `vmm_fast`; only
`vmm/nvme_baseline` selects the protected NVMe fallback. The runner passes
`config` to `enable_offload`; it never selects `execution_mode` directly.

The experimental `vmm_fast` route accepts
`--ddo-vmm-native-transport` to select its registered-host ring independently
of retained-source caching. Add `--ddo-vmm-native-source-cache-slot-mb` and
`--ddo-vmm-native-source-cache-slots` together to enable retention; that also
selects the ring. Both options default to zero. Cache slots are currently at
most 64 MiB each and 128 MiB in total; ring plus cache is limited to 1 GiB.
These limits describe the legacy explicit micro-cache configuration, not the
full-source route. `--ddo-vmm-source-cache-gb` selects larger retained storage;
its separate registration ceiling is described below. Do not extrapolate
micro-cache measurements to full-model coverage.

`--ddo-vmm-staging-buffer-gb` means **total** staging capacity. The native route
rounds it up once to whole windows of at most 64 MiB, not once per logical
slot. `runtime.state.planner_decisions.vmm_native_transport_registered_capacity_bytes`
reports that effective capacity in bytes at attach;
`vmm_native_transport` reports the effective transport selection. These setup
fields diagnose allocation/configuration; they are not measured DMA duration
or cache occupancy. Cache-off snapshots omit `fast_source_cache_*` counters.

Keeping the ring equal is necessary for a cache-off/on comparison, but is not
enough to isolate retention cost: the current cache-enabled adapter suspends
speculative ring prefetch to avoid rereading hits. Record that behavior in the
benchmark manifest. The default public PyTorch staging path and protected
NVMe runtime are unchanged. The explicit cache-off/on native selection passed
its small CUDA gate (`6 passed in 4.77s`) after correcting read-ticket slot
self-reuse and explicitly activating the primary context for the test-only
post-close ledger probe. Numerical results, repeated passes, source alias,
retention/bypass, JSON and zero registrations after close were checked.
Argument/configuration CPU contracts have passed. This is not a performance
measurement or validation of full-model retained-source coverage; harness
capacity/page/execution-unit parity still needs to be resolved before its gate.
### Persistent host storage (experimental ABI 8)

`vmm_source_cache_gb` now selects config v5 for the public retained-source
route. It limits committed native host storage, not GPU or CUDA-registered
bytes. `vmm_source_registered_gb` limits registration within that storage;
zero preserves the same ceiling, positive values must be whole 64 MiB
segments and no larger than retained capacity. The host runner can pass
`--ddo-vmm-source-registered-gb`; no memory ceiling grows automatically.
Legacy explicit cache-slot configurations retain versions 1-4 behavior.

Snapshots export these additional `fast_source_cache_*` metrics. Sizes are
bytes, counts are session totals; collection does not synchronize GPU.

| Suffix | Meaning / diagnostic interpretation |
| --- | --- |
| `committed_bytes`, `peak_committed_bytes` | Actual retained-storage allocation and peak, including segment padding. Separate from registered bytes and valid payload. |
| `process_committed_bytes` | Total native cache storage across components; add staging registered bytes for aggregate host consumption. Excludes external allocators. |
| `registered_limit_bytes` | Effective local registration ceiling, not retained-storage ceiling. |
| `registrations`, `unregistrations` | Registration churn. Unregistration in config v5 leaves native contents intact. |
| `registration_waits` | Local DMA event/stream waits needed to reuse registration capacity; not device-wide synchronization. |
| `ring_fallback_bytes` | DMA bytes served through existing native staging after direct registration refusal. Not file-read bytes. |
| `host_copy_bytes` | Retained-source to staging copies in that fallback. Normal registered-source DMA needs no host-to-host copy. |

Storage is filled once per admitted immutable file segment. Registration is
acquired for DMA and protected until its own completion; recycling it must
not increase source-read bytes. Source leases protect contents independently
of registration. Host pressure can still discard unleased ready storage, so
zero rereads is conditional on coverage and available memory, not guaranteed
for a partial buffer. Counters do not establish a performance guarantee for partial buffers or arbitrary models.

### Native model-source windows and GPU look-ahead

The persistent-source execution-buffer connection additionally reports these
`fast_` metrics for `vmm_fast_runtime` (zero when the route is inactive):

| Name | Unit / collection | Meaning and diagnostic use |
| --- | --- | --- |
| `execution_buffer_uploads` | units / cumulative uploads | Execution units uploaded to reusable native backing, not exceptional fallbacks. |
| `execution_buffer_bytes` | bytes / cumulative uploads | H2D payload through that buffer; it is traffic, not retained capacity. |
| `execution_buffer_capacity_bytes` | bytes / snapshot | Page-aligned reusable GPU capacity reserved inside the configured weight-residency ceiling. |
| `execution_buffer_resident_bytes` | bytes / snapshot | Currently materialized buffer backing, kept pinned against eviction. |
| `total_weight_resident_bytes` | bytes / snapshot | Model-arena residency plus execution-buffer residency. Compare this sum, not just arena residency, with the configured ceiling. Activation/other allocator usage is excluded. |

These counters do not prove full overlap or a particular throughput.

Source-cache ABI v10 / stats v7 additionally exports the following cumulative
host-wall scopes (`fast_source_cache_` prefix). Old stats v1–v6 prefixes remain
compatible; old DLLs omit fields they cannot measure rather than invent zeros.

| Suffix | Unit / collection | Meaning and diagnostic use |
| --- | --- | --- |
| `source_ready_seconds` | seconds / direct retained-source upload | Wait/check for immutable source I/O readiness before direct upload. Successful readiness checks entering upload are counted; explicit source `wait()` calls and fallback are excluded. High cold-pass cost is not proof of repeated reads. |
| `dma_submit_seconds` | seconds / native direct upload call | Host duration of the arena async-copy helper, including context handling and the driver enqueue call, not device-event DMA duration. Includes failed calls. Investigate blocking enqueue/backpressure when this grows. |
| `dma_event_seconds` | seconds / direct-upload completion-event record | Host event-record call cost, including failures. Excludes event creation, queries and stream dependency waits. |
| `ticket_close_wait_seconds` | seconds / submitted-ticket close | Host event/stream synchronization already required to safely close a ticket, including failures; no new synchronization is added. This can expose retirement waits hidden by a broader upload scope. Teardown costs after the pre-cleanup snapshot are not present in that snapshot. |

These operation scopes exclude cache-mutex acquisition time and Python
descriptor scheduling/query work. They are nested within broader upload or
cleanup scopes and are not a device transfer/compute breakdown.

Source-cache ABI v9 adds stats v6 timing suffixes, retaining v1–v5 snapshot
prefixes. `fast_source_cache_registration_seconds` and
`fast_source_cache_unregistration_seconds` are cumulative host wall seconds
around the persistent-source register/recycle helpers, including their local
ring lock/driver-call cost. Failed attempts also contribute. Legacy coupled
admission and teardown are excluded; zero is not a claim that they were free.
`fast_source_cache_registration_wait_seconds` measures only blocking event or
stream waits requested to recycle source registration capacity. It excludes
ticket-close waits and nonblocking completion queries.

These are not CUDA device DMA durations. They add no synchronization and
snapshots do not drain CUDA. Registration timing is nested in the existing
host DMA-enqueue scope, so do not add the scopes as independent durations.
Compare timing with registrations/unregistrations and denoise time to assess
whether registration churn is materially expensive. Host registration timings
are not device DMA bandwidth or proof of overlap.

Full-unit GPU look-ahead reports four counters under VMM fast:

| Metric | Unit / collection | Meaning / diagnostic action |
| --- | --- | --- |
| `fast_gpu_prefetch_units` | units / successful speculative enqueue | Entire next units faulted/pinned and necessary H2D submitted; GPU completion remains asynchronous. Not proof of transfer/compute overlap. |
| `fast_gpu_prefetch_hits` | units / consumer hook | Consumer inherited pins and waits for upload dependencies without repeating fault/upload. Compare with enqueued units. |
| `fast_gpu_prefetch_capacity_skips` | attempts / native admission refusal | Native capacity prevented look-ahead; demand path remains available. Investigate pressure/lifetimes rather than increasing budgets by default. |
| `fast_gpu_prefetch_cancelled_units` | units / cancellation | Look-ahead was not consumed in the expected pass/order. Native execution slots are released behind their upload fence; model-arena ranges still drain the owned stream before unpin. Investigate changing execution order. |

GPU look-ahead runs after current-unit compute enqueue. Pins belong to the
prefetched unit until consumption, then to its compute completion event.
Cancellation and upload-error cleanup may drain the owned upload stream;
normal consumption uses stream dependencies, not device-wide synchronization.
The older read-only look-ahead counters below describe the diagnostic source
prefetch path, not the full-unit GPU look-ahead now used by native transport.

`fast_prefetch_resident_skips` (count, VMM fast) records tensor candidates
whose last upload generation still matches the C++ arena's read-only residency
observation, so no speculative source read was issued. This counts tensors,
not execution units. It is not a pin or promise of future residency: consumer
fault/generation checks still trigger demand reads if eviction happens later.
Use alongside prefetch attempts/hits/skips and source-operation bytes when
diagnosing unnecessary speculation; do not treat it as avoided H2D bytes.

`vmm_native_source_cache_slot_mb` and `vmm_native_source_cache_slots` report
effective window geometry, not the zero-valued legacy arguments when GiB
capacity is selected. `vmm_source_cache_gb` is the requested ceiling;
`vmm_native_source_cache_model_windows` identifies the opt-in file-window mode.
The registered-byte snapshot is actual consumption, not requested capacity.

In this mode source-read bytes include the full aligned file window (EOF
clamped); DMA bytes include only requested tensor ranges. A small tensor can
therefore cause a larger cold read, while neighbors reuse those retained bytes.
Do not interpret registered host bytes as WDDM shared GPU residency, or cache
hits as evidence that disk traffic was eliminated for the entire model.

The isolated source-dispatcher candidate adds optional observations:

| Metric | Unit / collection | Meaning / diagnostic action |
| --- | --- | --- |
| `fast_source_cache_source_dispatcher_enabled` | boolean / cache snapshot | Source-only native dispatcher is selected; not proof that all model bytes are ready, registered or GPU resident. Older/disabled DLLs omit this field. |
| `fast_source_cache_source_dispatcher_owned_windows` | windows / optional native snapshot | Submitted window leases not yet retired by the dispatcher; may include completed I/O awaiting retirement. Not GPU residency, active chunk count or unread bytes. Expect zero after full source drain. |
| `fast_source_cache_source_dispatcher_peak_windows` | windows / peak since cache clear | Highest number of submitted window leases retained concurrently. Greater than one verifies multi-window admission, not simultaneous disk activity or speedup. Must not exceed window limit. Clear resets it. |
| `fast_source_cache_source_dispatcher_window_limit` | windows / optional native snapshot | Native metadata flight bound: minimum of admitted cache slots, existing I/O workers and eight. No extra storage/registration/GPU ceiling. Zero until dispatcher starts or after clear; older DLLs omit these three fields. |
| `fast_native_transport_source_prepare_seconds` / `_calls` | host seconds/count / unit descriptor preparation | Includes immutable metadata and native host admission, not background I/O completion. Include in total prepare/Generate; compare whole phase to avoid claiming a time shift as a gain. |
| `fast_source_cache_batch_bridge_source_prepare_total_seconds` / `_calls` | host seconds/count / native prepare API | Nested within transport source preparation; do not add these scopes as independent time. No DMA/new registration in source-only preparation. |

Background source filling contributes to existing source-read byte counters;
those can now be nonzero before denoise. `source_ready_seconds` measures consumer
wait for readiness, not total background I/O duration or exclusive disk time.
Registration still occurs on demand; source preparation does not establish
WDDM residency. Windows-only candidate; no Linux/WSL/macOS support claim.

## Same-model generation observations (host diagnostics)

The staged host harness can persist `generation_observations` for each denoise
generation. `repeat_index` is zero-based; `elapsed_seconds` measures latent
preparation through the existing CPU latent readback, excluding shared model
setup, subsequent report snapshot, final cleanup and VAE decode. It is not
whole Generate or full-pipeline time. `ddo_report` is a detached public report
with that generation's execution timer when GPU timing is enabled, otherwise
null. Snapshotting adds no explicit CUDA wait or intermediate ticket collection.

Native counters remain cumulative, so compare equivalent before/after reports
for deltas. Missing/pending timing is not zero; registered host bytes are not
WDDM residency. Same-model repeats do not validate unload/reload, numerical
parity of every latent, or long-run stability. See
[the bounded reuse gate](../developer/validation.md) for validation scope.

## Source event ownership (optional export)

`fast_source_cache_transfer_owned_events`, `fast_source_cache_pooled_events`
and `fast_source_cache_batch_completion_events` are counts at a locked native
metadata snapshot, not cumulative creations. They cover tracked DMA transfer
events, free pooled events, and immutable batch completion events respectively.
They exclude unsubmitted ticket events and all other allocators; do not sum them
as a global live CUDA-event count. There is no CUDA query or explicit wait,
although taking metadata locks can contend with an operation already running.

Expect transfer events to reach zero after successful final collection, batch
events after batch close, and free pooled events <=256 until cache close.
Batch completion events can legitimately remain while reusable batches exist.
Older DLLs omit these fields rather than report false zeros. Failed cleanup
requires retained ownership, not forceful counter reset. See
[the isolated lifecycle gate](../developer/validation.md).
## Native source-cache pressure observations (candidate v1)

An opt-in completed-phase PyTorch host-cache release has a separate public report
with before/after allocator and Windows commit observations. See the
[boundary API and metric catalog](../developer/pytorch-host-cache-boundary.md). It does not
clear this native cache or change admission policy.

Available on `vmm_fast_runtime` when the DLL exports the optional
`ddo_vmm_source_cache_pressure_snapshot_v1`. Older binaries omit these fields:
absence means unavailable, not zero. The existing cache stats ABI is unchanged.
JSON fields below live in `runtime.vmm` with prefix
`fast_source_cache_pressure_observation_`. `format_profile()` renders episode
counts and the last cause; hosts only consume the public DDO report.

Fixed-size observations reuse existing pressure samples and locks, without new
OS queries, CUDA waits, capacity changes or runner policy. Counts are cumulative
per cache lifetime, not per denoise step. Last-episode fields overwrite previous
episodes; this is not a timestamped event trace.

| Suffix | Unit / collection | Interpretation / diagnostic action |
| --- | --- | --- |
| `schema_version` | version / snapshot | 1; separate from main report/cache stats versions. |
| `valid_observations`, `failed_observations` | count / existing pressure polls | Valid/failed OS samples; failure suspends growth rather than inventing headroom. |
| `entries`, `recoveries` | count / suspension transitions | Pressure entries/resumption; a final resumed state does not erase earlier entries. |
| `ram_limited_entries`, `commit_limited_entries`, `both_limited_entries`, `unknown_entries` | count / entry | Entry causes, not every blocked request; compare with bypass and eviction counters. |
| `last_reason` | enum / last entry | 0 no entry, 1 RAM, 2 commit, 3 both, 4 unavailable observation. Does not identify the consuming external process. |
| `last_entry_ram_bytes`, `last_entry_commit_bytes` | bytes / last entry | Available physical RAM/commit; zero with reason 4 means unavailable. Windows commit is `ullAvailPageFile`, not disk free space. |
| `last_entry_committed_bytes`, `last_entry_registered_bytes` | bytes / last entry | Native source commitment/registered portion before reclaim; not process RSS or shared VRAM. |
| `last_recovery_ram_bytes`, `last_recovery_commit_bytes` | bytes / last recovery | Headroom when growth resumed; meaningful only if recoveries is nonzero. |
| `min_ram_bytes`, `min_commit_bytes` | bytes / valid samples | Sampled minima, not continuous peaks; meaningful only if valid observations is nonzero. |
| `low_bytes`, `high_bytes` | bytes / poll | Suspend/resume hysteresis thresholds; zero before first poll. Do not automatically change budgets. |
| `last_episode_reclaimed_bytes` | bytes / reclaim | Commitment released in the latest episode, including an active episode; zero before any entry. |
| `blocked_reclaims`, `failed_reclaims` | count / reclaim attempts | Busy ownership prevented discard, or reclaim failed. Busy never authorizes freeing live data. |

Observe RAM versus commit causes before changing policy. Bypass bytes do not
prove physical disk reads. These diagnostics do not establish speedup or CUDA
safety; see the [candidate gate](../developer/validation.md).

# Loaded component preflight

`DynamicOffloadLoadResult.inspect_vmm_component(component=..., buffer_limit_bytes=...)`
returns route `vmm_component_preflight`. It requires actual contiguous CPU
parameters and immutable descriptors, but does not load a DLL or initialize CUDA.
`setup.weight_source` retains loader diagnostics; `setup.component_capacity`
contains detached ownership/execution geometry. Units are bytes unless noted:

- `unique_parameter_bytes`: unique parameter identity groups, excluding buffers.
- `execution_slot_payload_bytes`: largest execution unit packed at 512-byte
  parameter alignment; not page-rounded committed GPU memory.
- `execution_slot_count`: existing native execution pool slot count (two).
- `execution_slot_total_payload_bytes`: slot payload times slot count; not
  total runtime VRAM, admission or a budget recommendation.
- `units`: names, parameter-group counts and packed payload per execution unit.
- `native_admission_validated`: always false at this report-only boundary.

Use these observations to prepare an explicit capacity gate; never interpret
them as successful CUDA execution or automatically increase a residency ceiling.

## Read-only host phase observations

`inspect_host_memory(module=None, phase="", max_storage_samples=16)` returns a
`DynamicOffloadMetrics` report with route `host_memory_observation` and
`setup.host_memory`. It never initializes/waits for CUDA, copies tensor data,
collects garbage, changes placement or adjusts budgets. Call only at a
nonconcurrent component boundary. The caller can omit the module for a host-only
snapshot. `format_profile()` renders the phase and available private/storage
bytes; normal trackers can print/save the unchanged public report.

| Field | Unit / collection | Interpretation / diagnostic action |
| --- | --- | --- |
| `phase`, `status`, `elapsed_seconds` | label/status/seconds per call | `observed` or `partial` storage inspection; host-sampling errors remain explicit inside `host`. Not continuous monitoring or proof of native safety. |
| `host` | bytes / adjacent CPU snapshot | Same RSS/private/physical/available-commit sampler as the host-cache boundary helper. Missing fields are unavailable, not zero. |
| `cpu_storages.cpu_tensor_references`, `cpu_tensor_reference_bytes` | count/bytes / registered CPU parameters and buffers | Counts named references, including aliases; byte sum is not deduplicated physical storage. |
| `unique_storage_count`, `unique_storage_bytes` | count/bytes / deduplicated CPU backing extents | Deduplicate address plus storage length. Tiny views can retain a whole shard. Extents are not proven private commitment or RSS. |
| `non_cpu_tensor_references` | count / registered tensors | Skipped CUDA/meta/other-device references; no data/statistics are queried on those devices. |
| `storage_samples` | bounded largest-storage records | Storage bytes, reference count, up to three tensor-name examples, and shard paths resolved by DDO's source descriptors when available. No tensor/storage object escapes the report. |
| `omitted_storage_count` | count / sampled records | Storage groups excluded by the configurable sample limit; aggregate totals still include them. |
| `inspection_errors`, `errors` | count/bounded records | Per-tensor storage/provenance failures; a partial census is not a zero-retention result. |

This is a census only of the passed module's registered parameters/buffers. It
does not enumerate closures, other components, arbitrary Python tensors, native
allocations, driver backing or system/external-process consumers. Host and
storage samples are sequential, not an atomic process-wide ownership census.
In particular, safetensors/PyTorch storage slices can expose a per-tensor
`nbytes()` while retaining a larger parent file mapping through hidden ownership.
The reported extent sum is not a census of those parent mappings; combine shard
provenance with an independent mapping/lifetime check before claiming release.

## Opt-in residual CPU source detachment

`DynamicOffloadApplyResult.detach_auxiliary_cpu_sources(enabled=False,
max_copy_bytes=16 * 1024**2)` returns the public report route
`vmm_auxiliary_source_detachment`. Fields below live in
`setup.auxiliary_source_detachment`. Call only after fast-native attachment,
before any forward, with exclusive component ownership. This diagnostic is not
a residency policy and is disabled by default. Unsupported ownership, partial
aliases, prior execution or insufficient copy allowance raise before admission;
no automatic ceiling increase is performed.

| Field | Unit / collection | Interpretation / diagnostic action |
| --- | --- | --- |
| `status`, `elapsed_seconds` | status/seconds per boundary | `disabled`, `detached`, or `no_source_backed_cpu_tensors`; not a performance acceptance result. Repeated successful calls return the original report without copying again. |
| `copied_bytes`, `copied_storage_groups` | bytes/count | Deduplicated source-backed CPU payload cloned once. Not shard size, released RAM or recovered commit. |
| `copied_tensor_owners`, `copied_named_references` | counts | Distinct registered tensor objects and named references, respectively; exact compatible aliases share the replacement storage. |
| `unbound_cpu_storage_groups` | count | CPU spans without source provenance, left untouched. Does not enumerate arbitrary escaped tensor references. |
| `source_shard_count`, `source_shard_samples` | count/up to 16 paths | Provenance of selected residual views; mapping release still requires separate evidence. |
| `host_before`, `host_after` | bytes / adjacent CPU samples | Same best-effort sampler as the host-cache helper. Samples are not atomic; missing fields are unavailable, not zero. Compare private/available commit without assuming a decrease. |
| `gpu_placement_changed`, `commit_recovery_proven` | booleans | Both false: no GPU placement is changed and the ownership operation alone does not prove OS commit recovery. |
| `scope` | label | Registered source-backed CPU tensors only, after native attachment. Other components, closures, external storage references and native allocations are outside the census. |

The operation preserves registered Parameter/Tensor identity, values, device,
dtype and admitted exact aliases. It refuses pinned, subclassed, noncontiguous
or incompatible aliases and whole-component ownership. CPU provenance is
rebound transactionally without changing immutable managed-source descriptors.
No CUDA initialization, device synchronization or garbage collection is added.

## Opt-in native allocation diagnostics

`result.allocation_diagnostics(enabled=False)` creates a single-use scope on an
active fast-native component. Disabled mode never accesses native/CUDA state.
Enabled mode requires the separate `DDO_VMM_ALLOCATION_DIAGNOSTICS=ON` build,
acquires exclusive process diagnostic ownership and releases it on exit. Older
or nondiagnostic binaries are refused, not interpreted as zero measurements.
It does not change allocation, cache, ceilings, residency or event ownership.

Use `scope.mark_step()` at the host's normal step callback, alongside
`result.execution_timer(enabled=True, native_units=True).mark_step()`. Attach
the finished diagnostic scope to the ordinary public metrics report:
`scope.attach(result.metrics_snapshot(execution_timing=timer))`. Runners print
or save that report; they do not read native or hook attributes. Resolve GPU
markers only after the normal phase-completion boundary, never by adding a
wait per step. This instrumentation is diagnostic, not performance evidence.

Fields live in `runtime.allocation_diagnostics` (schema 1):

| Field | Unit / collection | Interpretation and diagnostic action |
| --- | --- | --- |
| `status`, `failed`, `error` | scope status / CPU snapshot | Disabled, not finished, complete or failed; diagnostic failures must not mask an existing model error. |
| `host_scope_seconds` | wall seconds / context entry to exit | Includes host launch work and phase readback; not kernel-only time. |
| `steps[].step_index`, `host_submission_seconds` | zero-based index / seconds between CPU boundaries | No CUDA wait at the boundary; in-flight work can cross snapshots. Pair with the compute-stream event intervals, which include dependencies and host launch gaps. |
| `steps[].driver_delta`, `total_delta` | cumulative counter differences / boundary and scope | For each of `create`, `map`, `access`, `unmap`, `release`, `reserve`, `address_free`: `_calls`, `_host_seconds`, `_failed_calls`. Times cover calls issued by this DLL only, including blocking inside the driver, not all CUDA calls made by other libraries. Failed calls include retries and do not necessarily mean an unrecovered inference failure. |
| `pool_reuse_calls`, `pool_new_backings` inside deltas | successful reuse/new backing counts | Compare early versus late allocation churn. Reuse is counted after successful publication; new backing is counted even if its publication later fails. |
| `pool_observed_peak_mapped_bytes` | bytes / new pool-backing insertion | Process diagnostic high-water mark for pool-owned backing; no reset per step, no model arenas, no absolute GPU/shared-memory peak. Zero means no backing-creation observation, not proven zero GPU use. |
| `native_schema_version` | integer / scope snapshot | Native schema 1 lacks retry/fault observations; schema 2 adds them. Absent fields in older reports are unavailable, not zero. Public scope schema remains 1 with additive fields. |
| `access_retry_sync_calls`, `access_retry_sync_host_seconds`, `access_retry_sync_failed_calls` inside deltas | count / seconds / non-success count | Schema 2 times only the existing context synchronization in the model-arena access retry. No additional waits. A non-success synchronization return is not equivalent to an existing final fault counter. |
| `fault_range_calls`, `fault_range_host_seconds` inside deltas | count / seconds, entry through exit | Schema 2 measures enclosing native model-arena fault calls, including lock acquisition, materialization and retry waits. Overlaps driver and retry timers; never sum them as independent costs. Fault failed-call count is not supplied; use existing final native safety metrics. |
| `access_return_success_calls`, `access_return_not_ready_calls`, `access_return_out_of_memory_calls`, `access_return_other_calls` inside deltas | return-category counts / driver calls | Schema 2 categorizes this DLL's access returns. Other is a category, not an exact error code. Combined with retry counters, determines whether the suspected retry occurs; does not identify WDDM paging. Concurrent reads may split category/timing updates. |
| `boundary`, `scope`, `benchmark_eligible` | fixed labels / scope | CPU submission without CUDA wait; process native driver host calls; benchmark eligibility false. Concurrent calls may straddle reads; counters are not an atomic allocation census or an additive partition of GPU duration. |

No WDDM paging, kernel-only compute, cuBLAS autotuning or driver-loss diagnosis
is implied by these counters. Use the native unit timeline to locate slow
windows; do not attribute unexplained residual time to the cache by subtraction.

## Sampled operator diagnostics

`result.operator_diagnostics(enabled=False, module_path=None, invocations=(0, 2),
output_dir=None, use_cuda=True)` creates an explicitly diagnostic scope. The
host supplies a module path; DDO contains no model-specific class or path names.
Disabled mode installs no hooks and does not inspect the model, filesystem or
CUDA. Enabled mode admits one module and up to four distinct invocation indices
in 0..31. Selected forwards use PyTorch operator profiling with direct-child
labels; shapes, stacks and memory recording are disabled. Model inputs/outputs
are not retained by this scope. Existing forward functions remain unchanged.

Profiling can synchronize at start/stop and perturb host submission. CUDA
activity support must be available; an absent activity trace refuses capture
rather than silently substituting legacy CUDA timing. See the
[PyTorch profiler activity documentation](https://docs.pytorch.org/docs/2.14/profiler.html#torch.profiler.profiler.supported_activities).
Use single-threaded, nonrecursive inference without another profiler. Remove
diagnostic hooks before component retirement and report failures rather than
retrying or tuning memory automatically. No trace overwrites are permitted.

```python
with result.operator_diagnostics(enabled=True, module_path=selected_module,
                                 output_dir=diagnostic_directory) as operators:
    run_pipeline()
report = operators.attach(result.metrics_snapshot())
print(report.format_profile())  # Or pass report to the host's existing consumer.
```

For the existing `RunTracker` integration use
`tracker.record_ddo_metrics(report, component="transformer_diagnostics")`;
no metric internals are required. Attach accepts `DynamicOffloadMetrics`.

Fields in `runtime.operator_diagnostics` (schema 1):

| Field | Unit / collection | Meaning and diagnostic action |
| --- | --- | --- |
| `status` | scope exit | Disabled/not-finished/failed/incomplete/complete; missing requested captures are not passing evidence. CUDA activity absence stops capture. |
| `module_path`, `requested_invocations`, `observed_invocations` | path / zero-based indices / count | Invocations are module calls, not necessarily sampler steps or CFG passes. One sampled block is not representative proof for every block. |
| `benchmark_eligible`, `may_synchronize`, `scope` | fixed diagnostic labels | Not performance evidence; profiler cadence differs from normal execution. |
| `samples[].host_forward_seconds`, `profiler_stop_seconds` | wall seconds / selected forward and stop | Forward host duration may include blocking launches. Stop can drain work already queued, including earlier work; not independent kernel compute or an additive phase partition. |
| `samples[].device_activity_events`, `trace_path` | count / file path | Activity count validates observed CUDA capture; inspect the Chrome trace to distinguish launch gaps and device activity. Not WDDM paging or hardware utilization proof. |
| `samples[].top_host`, `top_device`, `child_ranges` | operator aggregates / microseconds normalized to seconds | Rows have name, calls, self/inclusive host and device seconds. Self excludes nested operators; inclusive rows overlap. Top lists retain at most 40 rows; full trace is the evidence for remaining events. Direct-child ranges label host-provided module structure. |
| `samples[].operator_count`, `invocation_index` | distinct aggregate names / zero-based index | Check coverage and compare selected calls; do not extrapolate missing steps. |

Missing or zero device activity is not evidence of zero GPU work. CUDA kernels,
runtime events and host operators may overlap; do not subtract their aggregate
durations from step time to claim driver paging. The normal allocator/residency
policy and protected runtimes are unchanged by this diagnostic feature.

## Execution auxiliary ownership candidate

`runtime.execution_auxiliary_ownership` is present only for the opt-in native
execution auxiliary adapter. Its `schema_version` is 1. `parameter_groups` is
the count of identity-unique auxiliary Parameter groups attached; `parameter_bytes`
is their unaligned payload in bytes, sampled at attachment. Neither is GPU
residency or driver usage. `budget_policy="existing_native_ceiling"` records that
these bytes enter existing native admission; `residency_cache=false` records that
no additional Python residency cache exists. Use native aligned slot/capacity
and transfer counters to diagnose actual admission and materialization.
`preserved_small_tensor_threshold_bytes` is the unchanged existing DDO small
placement threshold, in bytes; it is not a new residency ceiling. These metrics describe ownership, not performance acceptance. See
[validation](../developer/validation.md) for scope.

## Opt-in partial metrics journal

```python
with result.metrics_journal(output_dir / "denoise_snapshots", enabled=True) as journal:
    journal.record()  # before_denoise.json
    # Inside an existing host step callback:
    journal.record(step_index)  # zero-based; step_0000.json etc
```

Default-off journals do not inspect the result or create directories. Enabled
journals persist `result.metrics_snapshot(collect_completed=False).as_dict()`
without resolving CUDA timing or collecting pending events. Hosts do not access
native attribute names. The report is nested under `metrics` in a wrapper:

| Field | Unit / collection | Interpretation |
| --- | --- | --- |
| `schema_version` | version / publication | Journal schema 1; nested metrics retains its own report schema. |
| `observation` | label / publication | `partial_host_step_boundary`; not final acceptance. |
| `step_index` | zero-based index / callback | Null before denoise; scheduler submission boundary, not GPU completion. |
| `host_elapsed_seconds` | seconds / monotonic host clock | Time since journal construction; includes host and diagnostic I/O overhead, not GPU step time. |
| `gpu_completion_proven` | boolean / publication | Always false; pending counters remain meaningful and must not be silently treated as retired. |
| `completion_collection_requested` | boolean / publication | Always false; observation does not request resource collection. |

Records are detached and individually published without overwriting existing
files. File flush/publication is CPU I/O, not free benchmark instrumentation.
Unsupported filesystem publication or serialization errors propagate; previous
records remain intact. Abrupt termination can leave an unpublished `.tmp` file.
The journal scope drops its result reference on exit, avoiding retention of an
unloaded model. Final metrics, latent validation and cleanup gates remain
separate requirements.

## Experimental FP8 component-lifetime root state

These additive fields apply only when explicit native FP8 ownership includes
bounded plain parameters at the component root. They are not component buffers
or streamed unit parameters. Missing fields on other/historical routes are
unavailable, not evidence of zero memory use.

| Field | Unit / collection | Interpretation and diagnostic action |
| --- | --- | --- |
| `component_lifetime_parameter_groups` | count / capacity preflight or component ownership report | Root identity-unique parameter groups. Cross-scope aliases are refused. |
| `component_lifetime_parameter_bytes` | bytes / capacity preflight or component ownership report | Exact root payload, excluding page padding. Root payload plus buffers must fit the explicit component limit. |
| `component_lifetime_native_capacity_bytes` | bytes / component ownership publication | Native single-buffer capacity, including page rounding; not sampled total GPU usage. |
| `component_lifetime_budget_charge_bytes` | bytes / component ownership attachment | Capacity reserved from the caller's existing total weight ceiling. Must match native buffer capacity. |
| `declared_total_weight_ceiling_bytes` | bytes / component ownership attachment | Unchanged caller ceiling; it is not measured residency. |
| `arena_weight_ceiling_bytes` | bytes / component ownership attachment | Declared ceiling minus lifetime backing charge. Small remaining values can refuse streaming admission. |
| `component_lifetime_uploads` | count / component ownership publication | One successful setup upload; no root transfer per denoising step. |
| `fast_component_lifetime_capacity_bytes` | bytes / native runtime snapshot | Capacity of the currently retained lifetime backing. |
| `fast_component_lifetime_resident_bytes` | bytes / native runtime snapshot | Actual mapped physical residency of that native backing. |

Runtime fields live under `runtime.vmm`; ownership fields live in the public
component ownership report, and capacity fields in component preflight.
`fast_total_weight_resident_bytes` includes model arena, streamed execution
slots and component-lifetime backing. `fast_file_backed_bindings` and
`fast_file_backed_bytes` include the root immutable sources. These measurements
are not absolute driver GPU usage or process RSS. CPU contract acceptance does
not establish native CUDA admission, generation quality or performance.
