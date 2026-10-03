# Native VMM component expansion checkpoint

## Transformer gate and profile identity

The transformer real-model reuse regression passed three eight-step generations
at warm 1.401990/1.397915/1.394595 s/it. This permits preparing the next component,
not automatic DLL promotion or a declaration that the text encoder is supported.
See [executed gate](native-hardened-reuse-gate.md).

Explicit VMM component profiles now include their component in identity version3.
`enable_offload(..., component="transformer")` and `component="text_encoder"`
propagate that identity through DDO, without runner profile-key construction.
Same class/profile name in different components does not share capacity records.
Component labels are arbitrary strings, not architecture detection or model
class conditions. Generic component="model" VMM identity remains version2;
existing linear/SDNQ identities remain unchanged. Existing profile files are not
deleted or rewritten, and ambiguous legacy generic records are not silently
assigned to a newly explicit component. The standalone loader retains its
existing generic identity unless a subsequent explicit component attachment is
used. No residency budget is inferred or changed from VMM capacity observations.

Validation: 23 CPU profile/backend contracts PASS in 5.47 s, including separate
persisted transformer/text-encoder observations, unchanged legacy identity,
public component propagation and invalid blank component rejection. CUDA calls
in profile measurements are mocked; this is not an encoder CUDA gate.

## Outstanding text-encoder work, in order

### Phase-staged full-pipeline gate PASS (2026-10-03)

The explicit host opt-in passed one real-prompt BF16 generation1280x704/seed43,
8steps,no PAG: native encoder unload before transformer, normal VAE and visually
inspected image. Warm transformer CUDA1.395602s/it stays below1.46; first step
6.729011s, full CUDA denoise16.498256s, RunTracker total38.5135s.
Public pre-denoise process source commit/registration0; transformer completion
model/process26,038,239,232 bytes below24.625GiB, with separate1.5GiB staging.
Encoder/transformer collected tickets38/736→0, transfer owners0; no fault,
fallback, denial/bypass or pool quarantine. Sampled RSS27.3479GiB and approximate
absolute GPU6.97GiB satisfy gates; no post-unload global driver gauge.
Default encoder compatibility, protected runtimes, budgets and DLLs unchanged.
Full staged integration for this selected case is closed. Next consolidated
robustness/protected-route regression and reporting/docs closure, not automatic
default promotion or another benchmark. [Executed gate](native-staged-encoder-gate.md).

### Actual encoder native numerical gate PASS (2026-10-03)

Approved sequential reference/native processes passed for the real cat prompt,
BF16,1024 tokens. Two native embeddings agree with group-offload reference within
rtol/atol0.016, masks exact;1065 file-backed bindings and zero CPU-source bindings,
original loader Parameters released. Native parameters CUDA and unload retires to
meta preserving embedding/head identity. Fixed limits/monitoring passed; source
reads/commit/registration stable between calls, collected transfers/tickets zero,
no faults/fallbacks/denials/bypass or quarantine. No post-unload global gauge.
First/second native encoding phases6.4763/2.4827s include CPU readback/comparison,
not pure GPU inference; reference16.2952s includes readback/artifact writing.
The selected real text-only prompt gate is closed; host defaults unchanged.
Phase-staged host opt-in integration subsequently passed as recorded above,
without observed overlapping encoder/transformer source caches. Arbitrary input lengths,
multimodal execution and real-model interruption remain outside this evidence.
[Executed gate and limits](native-real-encoder-gate.md).

### Actual loaded encoder source preflight PASS (2026-10-03)

Approved CPU loader gate passed: all 1,065 parameter groups bound through loader
name transforms, no missing/rejected/ineligible groups or header errors, shared
embedding/head identity intact. BF16 parameter bytes 24,374,650,080; generic
layout 82 execution units/six buffers (34,818 bytes), largest execution slot
payload 2,013,757,440 bytes and two-slot payload 4,027,514,880 bytes.
CUDA uninitialized; no native admission/forward. Post-load RSS 719,892,480 bytes
is one sample, not evidence of every safetensor byte read/resident. The loaded
source prerequisite is closed, but actual native attachment, numerical prompt
embeddings and performance remain pending. [Executed CPU gate](native-real-encoder-gate.md).

### Real encoder preparation: nested-owner geometry (2026-10-03)

Repeated local CPU/meta audit reports exactly one direct parameter-owning
parent/child pair, in the multimodal projector and its norm. Name transforms
cover all 1,065 parameter groups with zero shape/dtype mismatches; CUDA remains
uninitialized. This is metadata coverage, not actual loaded-source binding.
The earlier disjoint-only component preflight would refuse the full encoder,
even though this branch is not necessarily executed for text-only input.

Allocator opt-in geometry now encloses nested non-root owners in the outer
parameter-owning execution unit, deduplicating bindings and preserving native
lifetimes. No architecture detection, branch omission, Python residency/cache
policy or ceiling adjustment. Root direct-owner overlap remains refused.
153 CPU contracts PASS in 6.23 s. Subsequent approved consolidated CUDA gate
passed seven tests in 5.78 s: nested parent projection and child norm actually
execute, reference BF16 outputs match, and RuntimeError/KeyboardInterrupt
interruption/resume/immediate unload pass alongside previous component/default
regressions. Same DLL/ceilings, no promotion or performance run. The small
nested-owner prerequisite is closed; actual encoder payload validation is not.

Then verify actual loaded source binding and execution-unit capacity, with
explicit ceilings including the shared 2,013,757,440-byte embedding/head and
native slot geometry. Do not infer capacity from the metadata totals or increase
budgets automatically. Compare the actual prompt embeddings/attention mask
against sequential group-offload reference execution in a separate fresh process;
never retain two full encoders simultaneously. Keep transformer/VAE out of this
component gate, and preserve the existing host encoder default. Exact commands
and resource limits must be proposed and approved before CUDA/payload runs.

### Metadata audit completed (2026-10-02; no CUDA)

Public `inspect_vmm_component` returns `DynamicOffloadMetrics` and calls the
allocator's exact Linear layout selector. Reports retain only metadata, include
nonselected/shared owners and buffers, and change no module/profile/budget.
CPU/meta contracts cover aligned selector parity, aliases, root/empty selection,
JSON, detached lifetime, no tensor copy/pointer access/native load, and
Parameter-identity versus storage identity. Metric definitions: [catalog](metrics.md).
Final consolidated CPU suite: 30 PASS in 4.81 s (capacity inspection, named
profiles, profile persistence and backend contracts); initial 28-test suite
also passed before adding storage/shared-module alias cases. No CUDA gate.

Local BF16 encoder audit using cached config, meta construction and safetensors
headers: 499 selected Linears, aligned payload 24,353,223,936 B (22.6807 GiB),
identity-unique model parameters 24,374,650,080 B. Nonselected-owner parameters
2,035,186,176 B (1.8954 GiB), of which 2,013,757,440 B (1.8755 GiB) are shared
with a selected output Linear; these sets overlap, not additional model storage.
Other nonselected weights total 21,428,736 B. Buffers total 34,818 B, including
generated/nonpersistent buffers not necessarily checkpoint-backed. Full config
includes vision/projector weights; metadata does not identify executed branches.
No native attachment or source binding, CUDA remained uninitialized.

The initial exact-name metadata comparison FAILED: all 1,065 Parameter groups
had no matching checkpoint name. The installed Transformers loader renames keys
at loading. A second diagnostic using that installed loader's name transforms
matched every group in shape/BF16, including the embedding/output shared alias.
At that checkpoint this successful name-only comparison was NOT a passing DDO
descriptor-binding gate: the binder still used exact names. The CPU association
work below supersedes that limitation, not the pending real-encoder CUDA gate.
Do not hardcode encoder prefixes/classes or bind meta storage/pointers to files.

Executed diagnostic (DDO Python, sibling host, local metadata only):

```powershell
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
& 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe' .\inspect_text_encoder_vmm_metadata.py 'F:\hf_cache\hub\models--elismasilva--ltx2.3-image-distilled-1.1\snapshots\5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9\text_encoder'
```

The host diagnostic depends on the installed Transformers name-conversion API;
it is not library policy, a loader replacement or a supported native encoder.
Existing group-offload loading/default and transformer DLL/runtime stay intact.

### Remaining implementation gates

1. Validate actual loaded encoder source binding in the later approved component
   gate. Generic descriptor association CPU contracts now pass (below), but no
   real encoder payload was loaded/native-attached in this preparation.
2. Establish DDO-owned placement/lifetime of embeddings, normalization and other
   non-targeted weights. Native dense-Linear coverage is not full-component
   support. Quantized/custom forward semantics must remain intact; unsupported
   cases keep compatibility routes rather than pretending complete VMM support.
3. Use DDO's weight-aware loader/descriptors and generic component lifecycle,
   replacing the host's direct encoder loading only in an opt-in test route.
   Do not build Python copies/caches/admission logic alongside native sources.
4. Validate a generic small component with embeddings, block list and outputs
   comparable to the unmodified forward: multiple inputs/lengths, shared
   parameters, cancellation/reuse and unload. Propose exact commands/limits and
   obtain approval before CUDA execution; no unapproved budget calibration.
5. Only after small contracts pass, validate the actual prompt encoder output
   and full staged workload against its existing group-offload route. Account
   for prompt/tokenization/component timing separately from transformer denoise.
   Keep group offload default until the new component gate is approved.

No LTX-specific code is added to DDO; workload orchestration stays in the sibling
host. Global allocator/bootstrap and source ownership remain generic. Windows
first; this does not add WSL/macOS or unvalidated Linux support.

## Loader association CPU checkpoint (2026-10-02)

DDO now uses the installed Transformers loader's ordered name-renaming API and
the same optional `key_mapping` passed to `from_pretrained`. No model class or
prefix is hardcoded. Other loaders retain exact-name association; their unknown
mapping semantics are not guessed. Unknown Transformers mapping APIs,
quantized payloads and any non-renaming converter disable direct file bindings
conservatively; they do not change normal loader or compatibility forward logic.
This is optional source metadata, not support for arbitrary payload transforms.

All parameter aliases are considered. Descriptors are published transactionally
per CPU contiguous pointer/byte-length group, counting storage bindings once.
Different source ranges converging on a target/shared storage, dtype/shape/byte
mismatches and invalid ranges are refused. Duplicate shard keys refuse the entire
source set. Existing live descriptor registries cannot be overwritten by rebind.
No meta pointer access, CPU tensor copies/cache, open source handles or retained
loader tensors are introduced. Backward-compatible manually built descriptors
can omit shape/dtype; parsed checkpoint descriptors always carry both.

`DynamicOffloadLoadResult.metrics_snapshot()` exposes detached loader diagnostics
as the public report, `setup.weight_source`; no runner access to private registry
attributes or CUDA queries is needed. See [metric catalog](metrics.md).

CPU gates: initial 56 PASS in 5.21 s; expanded protected preset/profile/backend
and source contracts 88 PASS in 5.47 s, then 88 PASS in 5.43 s after header-report
diagnostics. These are CPU contracts, not a benchmark or native encoder gate.
Final detached-report regression also passed 88 tests in 5.43 s; changing the
returned nested report cannot mutate module diagnostics. Default DLL hash stays
4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3.
Group-offload encoder loading/default, placement, budgets and DLLs unchanged.

## Alias-preserving ownership contracts (2026-10-02; CPU only)

Allocator public `capture_model_tensor_owners`/`TensorOwnerGroup` provides generic
direct parameter/buffer ownership groups, including repeated module paths.
Capture does not copy/read payload, access storage addresses, mutate the model
or initialize CUDA. Torch remains a call-time optional allocator dependency.
Parametrized modules and tensor subclasses are refused rather than substituting
backend-specific semantics. Parameter groups precede buffer groups deterministically.
Standalone allocator import with Torch deliberately unavailable also PASS;
the added exported helpers do not make Torch mandatory at import time.

Publishing an externally supplied view replaces every owner with one shared
Parameter/buffer object. The prepared loader tensor reference is dropped only
after every alias commits. Failed partial publication retains both references
until explicit rollback; successful publication cannot be rolled back without
the native lifetime gate. Shape/dtype/contiguity and concurrent owner changes are
checked before publication. Original buffer persistence flags are preserved.

Retirement retains the supplied final-consumer event and view while pending or
after event-query/assignment failure, refuses replacing a retained event and
permits retry. Only a completed event permits invalidating all aliases with
shared meta placeholders. This means the component is unloaded/unusable, not
that CPU weights have been restored. External alias changes block retirement;
native backing ownership/close remains the caller adapter's responsibility.
This helper does not record events, validate a real CUDA context, fence uploads,
fault weights or allocate an execution slot. CPU fake events are not CUDA proof.

Initial CPU attempt FAILED on parameter/buffer enumeration order; no CUDA ran.
Deterministic capture fix passed 102 contracts in 5.45 s. Final expanded suite
passed 105 in 5.53 s, including shared embedding/Linear numerical comparison on
CPU, original Parameter weak-reference release, shared module paths, buffer
persistence, invalid views, external mutation, pending/failed events, failed
partial publication/rollback and partial retirement retry. Existing profile,
source, backend and protected preset contracts were included.

At this ownership-helper checkpoint it was not connected to production
attachment. That is now superseded by a separate opt-in component adapter;
the validated default Linear adapter still keeps its previous attachment path.
The new path publishes one Parameter per group, maps all aliases/normalizations
to native execution units and owns bounded buffers. CPU suites passed 124/144/
146 contracts during integration; final immutable-inference contracts passed
148 in 6.07 s. Subsequent route/boundary regression passed 152 CPU contracts
in 6.12 s. The consolidated small CUDA gate passed five cases in 5.74 s on
2026-10-03, including BF16 shared embedding/head, normalization/buffer ownership,
RuntimeError/KeyboardInterrupt resume/unload and default-route regression.
Opt-in units use a BaseException-aware boundary around unchanged forward logic;
default Linear hooks are unchanged. See the exact integration restrictions,
command, limits and criteria in [component ownership gate](native-component-ownership-gate.md).
No blanket GPU embedding copy, cache/residency heuristic, DLL rebuild/promotion,
default change or full encoder placement support claim.

## Remaining support limitations

Actual loaded-source, native numerical and selected full staged encoder gates
passed as recorded above. General input/multimodal support, real-model interruption
and expanded profile-native feature observations remain pending. Driver-fault
recovery and second-live-context/multi-GPU hardening are still outside proved
support. Final documentation/publication audit and any history cleanup require
their existing scope/approval gates; profile identity work does not authorize
them or remove any protected runtime/compatibility route.
