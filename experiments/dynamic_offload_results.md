# Dynamic Offload Experiment Notes

`diffusers-dynamic-offloader` (DDO) is a model-agnostic offload router for Diffusers-style pipelines. It provides one preset-driven API that can route each component to DDO's dense-linear dynamic offload path, official Diffusers group offload, or a no-op/manual path.

DDO is designed for low-VRAM inference where full model residency is not possible or not desirable. Its dynamic path keeps selected modules resident on the accelerator, streams standard dense linear weights at runtime, optionally pins CPU weights when system RAM has enough headroom, and leaves quantized/backend-specific module forwards intact for compatibility.

This file is the report-ready benchmark sidecar. It summarizes what DDO is trying to improve, where it wins, where official Diffusers group offload is still preferable, and which results are closed benchmarks versus exploratory probes.

## Native VMM development gate

### Linux file-source/bootstrap groundwork (2026-10-03; native Linux pending)

Linux-only positioned file I/O and source C ABI metadata/shared-handle lifecycle
added in allocator; Windows branches preserved. Regular-file/CLOEXEC/range/EOF
validation, chunked pread/EINTR handling; immutable caller source contract, not
an external-writer lock or CPU tensor cache. DDO bootstrap now discovers Linux
libddo_vmm.so without falling back to a Windows DLL.209 Python/ABI CPU tests
PASS in5.76s with mocked relaunch, existing Windows ABI candidate unchanged.
Candidate/default hashes intact; no compilation/CUDA/Linux execution or promotion.
One standalone POSIX CPU contract prepared but not compiled/run. Native Linux
ring/cache/storage/admission/batches still require porting; no ready `.so` or
Linux support claim. [Implementation checkpoint](../docs/native-linux-port-plan.md).

### Lean Windows acceptance and next Linux port (2026-10-03)

User accepts the selected performance/critical robustness milestones without
further open-ended model/environment/long-run testing or chasing the remaining
small timing difference. Native text-only encoder is validated opt-in; defaults
and protected routes stay unchanged. Other models/environments are user-tested,
with targeted fixes/contracts when defects are reported.
Next block is native Linux `.so` portability, not only compilation: source I/O,
ring/cache/host admission and batch submission still have Windows-only branches;
implicit bootstrap library discovery also assumes the DLL path. Plan records
the concrete gaps and one bounded build/smoke sequence for later approval.
No source/DLL/default edit, compilation, CUDA run or history/publication action
in this checkpoint. [Linux port plan](../docs/native-linux-port-plan.md).

### Public reporting consistency (2026-10-03; CPU validated only)

Sibling RunTracker persists exact baseline and sampled absolute phase/global GPU
use alongside unchanged legacy deltas, without extra device queries or monitor
mode/cadence changes. Native host console route recognizes the explicit encoder
override and labels legacy booleans requested controls; JSON adds native opt-in
flag, public apply route/report remains executed-route evidence. Runtime, budgets,
allocator and DLLs unchanged.82 CPU contracts PASS in5.09s with mocked GPU queries
and AST-only runner tests; source syntax compilation PASS. No new CUDA/benchmark
or rewriting of historical artifacts. README/AGENTS/metric catalog link current
support/limitations; release validation and publication audit still pending.
[Support checkpoint](../docs/native-support-status.md).

### Consolidated robustness CUDA PASS (2026-10-03)

Approved one-shot sequence:11PASS in5.63s, no skips, exit0. Four ordinary/nested
component exception variants, two default interrupted-forward variants,
128-forward/four-reload source-generation contract, missing-current-context
owner retention, persistent registration recycling, GPU eviction/source reuse
and staging admission/refund all passed their existing numerical/lifecycle/
resource assertions. Intentional staging denial once is expected; healthy model
cases have no fault/fallback/source denial/bypass or collected transfer-owner leak.
Inspected reload reports: source reads8448/18816/8448/18816 bytes stable within
each32-forward cycle; source commit/registration64MiB. Interrupted default
warm/resumed reads12,672 bytes stable, tickets/transfer owners0, CPU bindings0.
Candidate/default DLL hashes unchanged. No rerun, benchmark, production patch,
budget/default change or promotion. Small suite has no independent process-wide
peak memory monitor; driver-loss recovery/WDDM migration are not proved.
Selected robustness gate closed; staged1.395602s/it remains previous performance
evidence, not remeasured. Protected linear/NVMe routes have CPU contract coverage
here, not a new performance gate. Public reporting/docs consistency remains.
[Executed command and scope](../docs/native-final-robustness-gate.md).

### Historical robustness preparation (2026-10-03; superseded by PASS above)

200 CPU contracts PASS in5.71s: protected preset/backend/profile routing,
source/capacity metadata, component ownership, fake event/cancellation/upload
failure retention, timing and source-lifecycle ABI. First attempt198PASS/1FAIL
in6.73s stopped on missing explicit DLL selection; corrected environment passed,
no production code change. CPU ABI-null-handle contract confirms CUDA uninitialized.
No new CUDA/benchmark execution, DLL rebuild/promotion or budget/default change.
Prepared one-shot11-case small CUDA regression for component/default interruptions,
128-forward source reload, missing-current-context owner retention, registration
recycling, GPU eviction and staging admission/refund. Deliberate staging denial
is an asserted case, not a normal-forward fault. No full-model rerun proposed.
[Exact command, fixed limits and acceptance](../docs/native-final-robustness-gate.md).

### Phase-staged native full-pipeline PASS (2026-10-03)

Approved one fresh-bootstrap real-prompt BF16 generation1280x704/seed43/8steps,
no PAG, DDO .venv-vmm; encoder native → unload → transformer native → VAE/image.
Saved1280x704 image visually inspected. CUDA first transformer step6.729011s,
warm mean1.395602s/it (strict<=1.46 PASS), CUDA phase16.498256s; prepare-latents
through CPU readback16.517878s. Transformer load0.3965s/prepare0.6261s.
Encoder phase12.6624s, Generate24.2141s, VAE1.5570s/save0.0760s;
RunTracker total38.5135s excludes bootstrap/import/startup. Different workload
from historical three-generation run; no total-time speedup comparison.

Encoder1065file/0CPU source bindings; source read/commit/registration
23,555,211,264 bytes. Before denoise process source commit/registration0;
transformer final model/process both26,038,239,232 bytes below24.625GiB,
with separate1.5GiB staging. No observed encoder cache overlap. Transformer
1152file/0CPU; source read25,846,148,360 bytes, DMA187,588,136,576 bytes.
Final collection encoder38→0/transformer736→0 tickets, transfer owners0;
fault/admission/read failures, fallback/denial/bypass/mismatch/quarantine0.
Sampled peak RSS27.3479GiB, GPU delta5.9403GiB plus rounded baseline1.03GiB
gives~6.97GiB absolute, below gates; monitor errorsnull. Not shared WDDM or
instantaneous caps. Standby purge skipped safely for missing privilege.
Candidate/default DLL hashes unchanged; no rerun/promotion/default/budget edit.
Actual native encoder public report supersedes legacy preset booleans in JSON.
Selected staged case closed; robustness/regression/publication work remains.
[Executed command, metrics and limitations](../docs/native-staged-encoder-gate.md).

### Historical phase-staged host preparation (2026-10-03; superseded by PASS above)

Sibling runner adds default-off --text-encoder-native-vmm; loaded-source/public
metrics and finally-based encoder unload before next component loading. Encoder
config is separate; transformer/default group-offload path unchanged. No resource
ceiling recalibration, native allocator/DLL edit or runtime promotion.
52 CPU contracts PASS in5.77s; syntax compiled only. Proposed one-generation
real-prompt1280x704/seed43/8step gate preserves transformer warm<=1.46s/it and
fixed memory/source ceilings. [Exact proposed command and acceptance](../docs/native-staged-encoder-gate.md).

### Real encoder native numerical PASS (2026-10-03)

Approved sequential fresh reference/native processes exited0, one local encoder
each, BF16/real cat prompt/1024 tokens/seed43. Native bootstrap precedes Torch.
Both native embedding outputs match group-offload reference at rtol/atol0.016,
masks exact/finite outputs. Native source coverage1065, CPU bindings0, loader
Parameters released; native unload retires parameters and preserves shared alias.

Phase wall observations: reference load/setup2.2579s, encode16.2952s; native
load/setup4.7261s, encodes6.4763/2.4827s, explicit unload2.3310s. Encode phases
include readback/artifact or CPU comparison/metrics; not exclusive GPU inference
or a controlled throughput benchmark. Reference unload0.0s marker does not
destroy its model and is not comparable to native explicit unload.
Source read/committed/registered23,555,211,264 bytes stable across both calls;
staging1,610,612,736 bytes. DMA25,545,825,792 then51,091,651,584 cumulative;
cache reuse does not eliminate DMA. Actual two-slot capacity4,030,726,144 bytes.
Fault failure/fallback/source denial/bypass/quarantine0; collected tickets38/47
to0, transfer owners0. No post-unload global resource gauge.
Reference peak RSS26.8873GiB, native27.7958GiB; native peak GPU delta5.0332GiB
above~1.03GiB baseline, not absolute GPU/shared-WDDM accounting. Monitor errors
null and approved sampled limits respected. Candidate/default DLL hashes unchanged.
No rerun, promotion, default migration, budget change or denoising.
The selected text-only prompt passes; general/multimodal/long-run support and
full staged performance remain pending. [Executed command](../docs/native-real-encoder-gate.md).

### Encoder-only numerical CUDA sequence prepared (2026-10-03; not executed)

Sibling standalone harness prepared: sequential ordinary group-offload reference
process, then fresh native bootstrap process with two real-prompt encodes and
embedding/mask comparison. DDO Python, local checkpoint/tokenizer, BF16,1024
tokens, fixed real prompt and seed43. Existing encoder/transformer defaults unchanged.
Native limits fixed at4GiB active,512MiB allocator cache,1.5GiB staging,2MiB pages,
24.625GiB source/registration,1MiB buffers. RunTracker RSS/GPU sampling retained;
sampled RSS40GiB/GPU7.5GiB checks at phase boundaries, not instantaneous caps.
Script syntax compiled only; no CUDA run or performance result.
[Exact two-process proposal and acceptance](../docs/native-real-encoder-gate.md).

### Actual encoder loaded-source preflight PASS (2026-10-03)

Approved one-encoder CPU command exited successfully. Source coverage 1,065/1,065;
zero missing/rejected/ineligible groups and header errors; shared embedding/head
preserved. BF16 unique parameter bytes 24,374,650,080; aligned payload
24,374,652,672; 82 execution units and six buffers (34,818 bytes).
Largest execution slot payload 2,013,757,440 bytes, two slots 4,027,514,880 bytes;
this is geometry, not page-rounded native residency/admission.
Host loader/preflight elapsed 0.5996301 s excludes imports/startup; single post-load
RSS 719,892,480 bytes with 47,895,617,536 bytes available before. Lazy mapped CPU
payload means low RSS does not establish complete reading/residency. CUDA remained
uninitialized; no native attachment, numerical output, benchmark or DLL/budget/
default change. [Executed command and scope](../docs/native-real-encoder-gate.md).

### Real encoder loaded-source gate prepared (2026-10-03; payload pending)

Public loader result now exposes detached immutable-source/component preflight
through DynamicOffloadMetrics, including 512-aligned execution-unit payload and
two-slot geometry. No CUDA/DLL/attachment/admission or budget policy added;
existing native execution code unchanged. 154 CPU contracts PASS in 6.26 s.
Sibling one-encoder CPU diagnostic prepared and syntax compiled; actual BF16
encoder payload NOT loaded. Exact proposed command, RAM preconditions and
acceptance: [real encoder gate](../docs/native-real-encoder-gate.md).

### Nested-owner consolidated CUDA PASS (2026-10-03)

Approved seven-case sequence: 7 PASS in 5.78 s, DDO .venv-vmm and lifecycle OFF
DLL SHA256 755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729.
Two nested-parent cases actually execute direct projection and child norm;
reference BF16 outputs match across input lengths/reuse. Twelve file-backed
parameter groups/five execution units, shared embedding/head aliases, zero CPU
source bindings, stable warm source reads, no fault failure/temporary fallback/
source denial/bypass or collected transfer-owner leak. RuntimeError and
KeyboardInterrupt resume and immediate unload pass, with registry close and
checkpoint deletion. The five earlier component/default CUDA cases also PASS.
Ceilings remain 0.07 GiB active, 2 MiB staging/pages, 64 MiB source cache and
registration, 1 MiB component buffer. No rebuild/DLL promotion/benchmark.
Real encoder loaded sources, capacity and numerical outputs remain pending.
[Executed command and criteria](../docs/native-component-ownership-gate.md).

### Encoder preparation: nested-owner CPU checkpoint (2026-10-03)

Local encoder meta/header audit: one nested direct-parameter owner pair;
1,065 groups covered via loader name transforms, zero shape/dtype mismatches,
CUDA uninitialized. This is NOT loaded-source binding or capacity validation.
Generic opt-in allocator execution geometry now encloses non-root nested owners
in their outer parameter-owning unit; root overlap remains refused. No class
conditions, branch pruning, cache policy, DLL/budget/default change.
153 CPU contracts PASS in 6.23 s. At this CPU checkpoint geometry was CUDA
PENDING; subsequent seven-case PASS above closes the small nested-owner gate,
not actual encoder payload validation. [Component plan](../docs/native-component-expansion-plan.md).

Prepared seven-case consolidated CUDA proposal: two new nested-owner interruption
variants plus five previous cases. Nested parent projection/child norm actually
execute, with 12 unique source groups/five units; same DLL, ceilings and BF16
tolerance. Initially syntax compiled only; subsequently approved and executed
with seven PASS in 5.78 s. [Exact command/acceptance](../docs/native-component-ownership-gate.md).

### Component ownership consolidated CUDA PASS (2026-10-03)

Approved corrected five-case gate: 5 PASS in 5.74 s, DDO .venv-vmm and existing
lifecycle OFF DLL SHA256
755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729.
Small BF16 component preserves numerical outputs and shared embedding/head
aliases, native normalization parameters and bounded buffers, with no original
CPU Parameter retention. Nine file-backed/zero CPU-source bindings, stable warm
source reads, zero fault failures/temporary fallback/source denial/bypass and
zero collected transfer owners; interruption/resume and immediate unload pass
for RuntimeError and KeyboardInterrupt. Registry closes and checkpoint deletes.
Existing two interrupted-forward cases and 128-forward/four-reload default-route
regression also PASS. No limit change, rebuild, DLL promotion or performance run.
Real encoder loading/capacity/numerical validation remains pending; small CUDA
contracts are not a full encoder support claim. [Executed command and criteria](../docs/native-component-ownership-gate.md).

### Component interrupt boundary correction (2026-10-03; CUDA retry pending)

Second approved five-case CUDA attempt stopped with 1 PASS / 1 FAIL in 5.73 s.
RuntimeError native component case passed numerical BF16, shared aliases,
source reuse, interruption/resume and immediate unload. KeyboardInterrupt left
a unit active: installed Torch always_call hooks catch Exception, not
BaseException; resume was rejected as reentry, and unload refused backing
release. Three default-route CUDA cases not executed; consolidated gate FAILED.
Opt-in component units now use a BaseException-aware boundary around unchanged
forward logic, retaining default-route hooks and all memory ceilings/DLLs.
152 CPU contracts PASS in 6.12 s, including exception fencing/resume/removal.
No encoder benchmark or native capacity/support claim. Corrected consolidated
CUDA retry requires fresh approval: [gate command](../docs/native-component-ownership-gate.md).

### Component CUDA fixture route correction (2026-10-02; retry pending)

Approved consolidated CUDA sequence stopped on its first case: 1 failed in
5.16 s at old loader-Parameter release, before native numerical execution.
The new fixture used `component='text_encoder'` with the default preset, which
selected Diffusers group offload instead of the intended native component route.
Fixture corrected to request `route='dynamic_offload'` and assert route/hook;
149 CPU contracts PASS in 5.98 s, including explicit encoder-route dispatch.
The remaining four CUDA cases were not run. No allocator/default preset, DLL,
budget or benchmark change; protected DLL SHA256 unchanged. Corrected five-case
CUDA retry requires fresh approval: [gate command](../docs/native-component-ownership-gate.md).

### Opt-in native component integration CPU PASS (2026-10-02; CUDA pending)

Public allocator dispatch now connects ownership to a separate opt-in component
adapter: one native range/shared Parameter per group, direct aliases including
embedding/output consumers and block norms, bounded device buffers, native
fault/unpin/event/slot execution. Full selection/eval/immutable-source/disjoint-
unit preflight; no speculative component lookahead or CPU weight cache.
Protected default Linear attachment/lookahead and sibling encoder remain unchanged.

CPU suites 124 PASS in 5.75 s; 144 in 5.83 s; 146 in 5.80 s; final expanded
regression 148 in 6.07 s. Includes source/profile/preset/backend/layout and pending
unpin/prefetch-cancel contracts, shared binding publication, lifetime retry and
reentry/output-view escape guards. New CUDA source syntax compiled, NOT run.
No C++ rebuild, budget change, model benchmark or DLL promotion. Exact proposed
five-test CUDA command and acceptance: [component ownership gate](../docs/native-component-ownership-gate.md).

### Generic tensor-owner lifecycle contracts CPU PASS (2026-10-02)

Prepared allocator alias-preserving publication/retirement bookkeeping for
parameters and buffers. Shared embedding/Linear owners receive one Parameter,
successful publication drops the prepared loader Parameter reference, and
retirement requires final-consumer completion before invalidating aliases.
Event/query/assignment failures retain view/owner state for retry. No event is
recorded by this helper and CPU fake events do not prove native completion.

Initial CPU attempt FAILED on capture ordering; deterministic parameter-before-
buffer order then passed 102 in 5.45 s, final expanded regression 105 in 5.53 s.
Includes source/profile/backend/protected preset regressions. The helper is not
connected to production attachment/placement yet; no encoder CUDA, benchmark,
copy/cache policy, budget or default/DLL change. Next step is native execution-unit
integration, not a permanent GPU embedding copy. [Component checkpoint](../docs/native-component-expansion-plan.md).

### Generic source-association CPU checkpoint (2026-10-02)

DDO source binding now follows installed Transformers pure name transforms and
the loader's optional key_mapping, includes shared aliases, validates shape/dtype
and bytes, and refuses ambiguous/conversion/quantized sources. Other loaders keep
exact-name behavior without architecture conditions. Optional immutable descriptor
shape/dtype fields preserve manually constructed legacy descriptors. No CPU tensor
retention/copies or native policy change. Public loader metrics expose diagnostics.

Initial source/profile/backend CPU suite: 56 PASS in 5.21 s. Expanded protected
preset regression: 88 PASS in 5.47 s, repeated 88 PASS in 5.43 s after diagnostic
report additions. Final detached-report regression: 88 PASS in 5.43 s, no
source-owner retention. No CUDA/native encoder attachment or performance run;
placement, group-offload default, budgets, protected runtimes and DLLs remain unchanged.
Actual encoder loaded-source validation remains pending along with generic
embedding/nonselected-weight lifetime support. Details: [component checkpoint](../docs/native-component-expansion-plan.md).

### Encoder metadata coverage audit (2026-10-02; no CUDA/performance claim)

Public report-only inspector shares native attachment selection, preserves
nonselected/shared owners, no DLL/CUDA/payload copies. Actual local meta BF16
encoder: 499 selected Linears, aligned payload 24,353,223,936 B, identity-unique
parameters 24,374,650,080 B. Nonselected-owner bytes 2,035,186,176 B include
2,013,757,440 B shared embedding/output Linear; other nonselected bytes
21,428,736 B, buffers 34,818 B. These are layout counts, not VRAM peaks/budgets.
Consolidated CPU contracts: 30 PASS in 4.81 s, using DDO `.venv-vmm`.

Initial exact-name source comparison FAILED for 1,065 groups. Installed loader
renaming then matched all groups in shape/BF16 in metadata only. DDO's existing
exact-name source binder still needs generic loader-name provenance handling;
no native encoder attachment/source-binding gate was executed. Group offload,
protected routes, DLLs and approved transformer performance remain unchanged.
Details: [component expansion checkpoint](../docs/native-component-expansion-plan.md).

### VMM component profile preparation CPU PASS (2026-10-02)

23 profile/backend CPU contracts PASS in 5.47 s. Explicit VMM component identity
version3 prevents same-name/class transformer and text-encoder capacity records
from colliding; public enable_offload propagates the component. Generic VMM and
linear/SDNQ identities remain unchanged; no file rewrite, budget, routing or
native-policy change. Profile CUDA measurement calls were mocked. No actual
encoder VMM execution or new performance gate. Next source/layout/placement
requirements: [component expansion checkpoint](../docs/native-component-expansion-plan.md).

### Hardened real-model reuse regression PASS (2026-10-02)

One explicitly approved bootstrap process, real prompt/BF16, 1280x704, seed43,
three eight-step generations; exit zero. DDO `.venv-vmm`, sibling host, lifecycle
OFF candidate hash verified. All three complete CUDA reports passed the unchanged
strict warm <=1.46 s/it criterion:

| Generation | First GPU step (s) | Warm (s/it) | GPU denoise phase (s) | Generation readback wall (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7.528940 | 1.401990 | 17.342902 | 17.356360 |
| 2 | 1.395665 | 1.397915 | 11.181079 | 11.196348 |
| 3 | 1.390656 | 1.394595 | 11.152851 | 11.166465 |

Source reads stable 25,846,148,360 B; committed/registered stable 26,038,239,232 B,
388 registrations. Unique pool mapped 2,543,845,376 B constant, quarantine zero.
Transfers/tickets 736/704/704 at deliberately uncollected generation snapshots;
final collection 704->0, transfer owners zero, free event pool 256 and immutable
batch event owners 103 while still attached. No fault/temporary fallback,
mismatch, source denial/bypass/pressure eviction or monitor error. No retained
CPU source bindings. DMA still 184,900,627,456 B for each additional generation.

Transformer offload prepare 0.7483 s, load 0.3657 s separately. Whole Generate
48.0874 s covers THREE denoises; text encode 20.2336 s, total 69.9961 s. Peak
GPU delta 6.3214 GiB (not absolute use), RSS 29.5053 GiB. Existing CLI staging
0.5 GiB is per slot: THREE slots/1.5 GiB effective DDO total, confirmed by report.
Other limits unchanged: 4 GiB weight, cache/registered 24.625 GiB, allocator
cache 512 MiB. Safe standby purge skip due to privilege, no runtime escalation.

Candidate/default/milestone DLL hashes unchanged. No promotion/rebuild/rerun;
approval exhausted. Prior reuse FAIL remains FAIL; one passing candidate run
does not prove causal speedup or performance variance, full-model cancellation,
driver failure recovery or numerical parity across repeated generations.
Details/artifact: [executed hardened reuse gate](../docs/native-hardened-reuse-gate.md).

### Upload first-error diagnostics checkpoint (2026-10-02; CUDA regression PASS)

70 selected CPU contracts PASS in 5.05 s. Native-buffer upload bridges now keep
the original upload exception when fence/release also fails, attach secondary
diagnostics and retain unsafe-close refusal. Nine new fake contracts; no C++,
budget or protected-runtime change. Approved six-case CUDA regression executed
once: four existing execution-buffer cases and two interrupted-forward variants,
existing case limits. Six PASS, no skips, 5.16 s, exit zero; all existing
numeric/resource/reuse/cleanup assertions passed. Lifecycle OFF DLL hash verified.
Approval exhausted; no rerun/rebuild/benchmark/promotion. Combined upload/fence
failure diagnostics are CPU fake proof, not real driver fault validation.
Details: [lifecycle checkpoint](../docs/native-lifecycle-hardening.md).

### Partial look-ahead cancellation correction (2026-10-02; CUDA regression PASS)

61 selected CPU contracts PASS in 5.13 s. Fast-VMM cancellation tracks confirmed
unpin progress so a retry does not repeat successful releases; failed upload
fence/lease cancellation retains owners. Eight new fake contracts, no residency
policy/ceiling change. Approved CUDA regression executed once: existing two
cancellation variants plus 128-forward reload case, unchanged limits. Three
PASS, no skips, 5.24 s, exit zero; all numeric/source/resource/cleanup assertions
passed. Lifecycle OFF DLL hash verified. No rerun, benchmark, rebuild or promotion;
approval exhausted. Real driver failures remain unproven. Details:
[lifecycle checkpoint](../docs/native-lifecycle-hardening.md).

### Interrupted-forward resume/unload gate PASS (2026-10-02)

Two explicitly approved CUDA variants (RuntimeError/KeyboardInterrupt) PASS,
no skips, 4.86 s, exit zero. Lifecycle OFF DLL hash verified before execution.
Three-block generic model: warm, interrupt after first block, resume, interrupt
again and immediately unload without intervening readback/metrics collection.
Both resumed outputs matched reference. Each case observed one look-ahead
cancellation by resumed boundary; source reads unchanged at 12,672 bytes, zero
tickets/transfer events and stable committed/registered source storage at 64 MiB.
Pool stable within each case: 2/6 events respectively, bounded values not a leak
claim. Final registry closure/hook removal/source-file deletion passed; no global
post-unload gauge. No driver fault, benchmark, rebuild or promotion. Approval
exhausted. Details: [executed cancellation gate](../docs/native-forward-cancel-gate.md).

### Pending-unpin ownership correction (2026-10-02; CUDA regression PASS)

51 selected CPU contracts PASS in 5.13 s. Experimental VMM adapter now retains
queued completion events/ranges after query/wait/unpin failure and commits only
successful releases. Ten new fake-event contracts include partial release retry
and KeyboardInterrupt. No native policy/budget change, rebuild or promotion.
CUDA regression executed once after explicit approval: context recovery plus
existing 128-forward tiny public-model reload gate, unchanged case parameters.
Two PASS, no skips, 5.47 s, exit zero; all numeric/resource/cleanup assertions
passed. Lifecycle OFF DLL hash verified, no rebuild or promotion. Approval
exhausted. No performance, poisoned-driver recovery or full-model cancellation
claim. Details:
[lifecycle checkpoint](../docs/native-lifecycle-hardening.md).

### Missing-current-context recovery gate PASS (2026-10-02)

One explicitly approved CUDA test PASS, no skip, 4.68 s, exit zero. Lifecycle OFF
DLL hash checked before execution, no rebuild or promotion. After real tiny DMA,
ticket close/cache clear/cache close without a current context rejected with
status 3 and preserved handles/arena ownership. Restored owner context allowed
cleanup and another numerically correct upload without file reread (2 MiB source
read total). Completed boundaries: zero live tickets/transfer events; clear left
zero committed/registered bytes; source file deletion after close passed.
No full-model benchmark or parameter change. Missing context is not another live
context, another GPU or a lost device. Approval exhausted. Details:
[executed context gate](../docs/native-context-recovery-gate.md).

### DLL compatibility CPU checkpoint (2026-10-02)

35 CPU contracts PASS in 5.06 s. Required exports and loader failures now report
the selected DLL with actionable diagnostics; optional exports remain optional.
Bootstrap malloc/free preflight occurs before Torch import/CUDA availability
query. Protected default, span-pool milestone and lifecycle OFF candidate passed
separate load/export checks, unchanged hashes, no CUDA calls or promotion.
Export presence is not ABI-signature proof. Context/device ownership source audit
found existing fail-closed guards; real-driver wrong-context/multi-GPU validation
remains pending. Details: [lifecycle checkpoint](../docs/native-lifecycle-hardening.md).

### Generic-model reload gate: 128 checked forwards PASS (2026-10-02)

One generic DDO public-route test prepared: four A->B->A->B model attachments,
32 numerically checked forwards each, same-path source replacement, hook/source
cleanup and public resource snapshots. Fixture/configuration CPU suite 31 PASS
(5.19 s). Explicitly approved CUDA test 1 PASS, no skip, 5.25 s, exit 0:
128 outputs matched CPU reference; no unexpected native failures/fallback.
Source reads 8,448 B for A /18,816 B for B, no rereads within attachments.
Source committed/registered bytes stable at 67,108,864 B, tickets and transfer
events zero after per-forward collection; two batch completion owners while
attached. Pool 2 events for first A, 4 thereafter, stable within each attachment.
Aggregate source committed/registered bytes zero before every new first forward;
registry/hook cleanup and same-path file deletion/replacement passed.
Isolated lifecycle OFF DLL, one 64 MiB source
window, 2 MiB staging, 0.07 GiB weight ceiling; these are small-contract limits,
not changed model benchmark parameters. No runtime policy patch or promotion.
Tiny models plus per-forward collection do not prove large-model pressure,
cancellation or production custom-allocator longevity. OFF hash verified;
milestone/protected DLL hashes unchanged. No rerun/promotion; approval exhausted.
Executed command and coverage: [generic lifecycle gate](../docs/native-model-lifecycle-gate.md).

### Native lifecycle injection: CPU and three small CUDA contracts PASS (2026-10-02)

Isolated ON/OFF builds compiled; test hooks OFF by default, absent from OFF
exports. One-shot native copy/close refusal does not trigger a driver fault.
Optional snapshot exposes tracked transfer/free-pool/batch event ownership, not
global live CUDA handles. ON/OFF selected CPU suites 48 PASS each. After explicit
approval, three small CUDA cases passed, no skips, 4.80 s, exit 0: refusal after
one actual DMA retained partial batch/cache owners; one failed close retained
owners until successful retry; 16 repeated batches stabilized tracked event
ownership, source-read and committed/registered bytes with correct payloads.
ON hash verified before run; milestone/protected hashes unchanged afterward.
No model run, promotion, actual driver-failure or performance claim. Approval
exhausted, no rerun. Exact builds and coverage limits:
[native lifecycle gate](../docs/native-lifecycle-cuda-gate.md).

### Lifecycle bridge hardening: CPU PASS, CUDA pending (2026-10-02)

Reviewed ticket/batch/cache close and partial submission ownership; added 13
fake-ABI lifecycle cases. Cache/ticket context-manager cleanup now preserves
the original operation exception and adds failed-close details as a note;
explicit close still raises and failed close retains handle/dependencies.
Selected CPU/file-owner/timing/bootstrap suite: 44 PASS (4.73 s), including
the fresh-process guard proving native file contracts do not initialize CUDA.
No C++ policy/DLL/budget change, no CUDA run or performance claim. Real native
failure injection, live-event accounting and long-process lifecycle proof remain
pending. See [checkpoint](../docs/native-lifecycle-hardening.md).

### Same-model generation reuse: source reuse PASS, strict warm gate FAIL (2026-10-02)

One explicitly approved process completed three real-prompt BF16 generations,
eight steps each, 1280x704/seed43 with the existing advancing generator. DDO
Python, span-pool milestone `8CAAD072...`; protected `4AC3D8B...` unchanged.
Weight/staging/source/registration/cache ceilings unchanged. Encoder once, only
the final latent decoded. No runtime allocator patch, intermediate collection,
extra wait or retry. New per-generation observations passed 19 CPU tests first.

| Observation | Generation 1 | Generation 2 | Generation 3 |
| --- | ---: | ---: | ---: |
| First GPU step s | 7.627204 | 1.479055 | 1.420242 |
| Warm GPU steps 2-8 s/it | 1.473632 | 1.437565 | 1.408667 |
| Complete GPU phase s | 17.942639 | 11.542035 | 11.280940 |
| Latent prepare through CPU readback s | 17.957989 | 11.556646 | 11.295934 |
| Cumulative source-read bytes | 25,846,148,360 | 25,846,148,360 | 25,846,148,360 |
| Cumulative readiness host wait s | 6.782640 | 6.833092 | 6.881722 |
| Live source tickets before final collection | 736 | 704 | 704 |

No additional source file reads or registrations after generation 1. Committed
and registered source storage constant at 26,038,239,232 B, registrations 388;
unique pool mapped backing 2,543,845,376 B, pending metadata 2,086,775,808 B;
pool evictions 53 and host eviction wait 0.465545 s remain unchanged at these
boundaries. Weight backing 3,858,759,680 B. No native faults/admission failures,
temporary fallback, pressure, mismatch, quarantine, bypass or monitor error.
Final existing nonblocking collection: tickets 704 -> 0.

DMA remains necessary: cumulative 187,588,136,576 / 372,488,764,032 /
557,389,391,488 B, i.e. 184,900,627,456 B each subsequent generation. Reuse
eliminates source file reads, not accelerator transfers; no WDDM residency claim.
Event creation count 736 -> 1,216 -> 1,696 is cumulative, not live handles.
Source review shows a 256-event free pool with excess destroyed on retirement;
live event accounting still needs explicit hardening proof, not a leak conclusion.

Strict per-generation warm <=1.46 FAILED on generation 1 (1.473632); generations
2/3 passed. Source reuse/correctness observations passed, but no full acceptance,
causal regression or variance claim from this one process. DLL unchanged, no
automatic promotion/rerun. First-step penalty largely disappears with retained
source: GPU phase falls by 6.400604 / 6.661698 s versus generation 1. These
in-process comparisons include different random latents and warming effects.
Readiness counters are nested host observations, not additive/exclusive I/O time.

Shared DDO prepare 0.8471 s. Whole Generate (THREE denoises) 49.6829 s; total
71.7061 s, text phase 20.3610 s. Do not compare with one-denoise Generate or
divide to claim full-pipeline latency. Peak CUDA-used-minus-baseline 6.7182 GiB
(not absolute VRAM), RSS 29.5659 GiB. Only final PNG saved, so no per-generation
image parity proof. Artifacts: sibling host
`outputs/ltx_image_modular_vmm/generation_reuse_gate`. Approved sequence exhausted.
Next: prioritize retained-model lifecycle/long-run resource proof; cold source
remains a first-use cost. Further cold optimization requires a critical-path
mechanism, not another speculative concurrency change. See
[reuse gate](../docs/native-generation-reuse-gate.md).

### Multi-window gate: concurrency PASS, performance FAIL (2026-10-02)

Approved sequence completed once: eight CUDA tests PASS (5.13 s), span probe
PASS, one unchanged real-prompt/BF16 eight-step model run, exit 0 with DDO
`.venv-vmm`. Candidate `A958C53E...` selected/hash verified. Output under sibling
`outputs/ltx_image_modular_vmm/source_flights_20261002`; identical PNG SHA256
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.

| Metric | Eight-window candidate | Single-window candidate | Preserved milestone repeats |
| --- | ---: | ---: | ---: |
| GPU warm s/it | 1.423592 | 1.412124 | 1.412275 / 1.375524 |
| First GPU step s | 6.071087 | 6.082628 | 6.540682 / 6.819729 |
| DDO prepare s | 1.6399 | 1.6631 | 0.8337 / 0.7720 |
| Host denoise s | 15.3484 | 15.3018 | 15.7681 / 15.8085 |
| Generate s | 24.8827 | 24.4543 | 23.2125 / 23.2392 |
| Total s | 46.1006 | 45.5254 | 44.2206 / 43.5638 |

Actual retained flight peak/limit = 8/8; owned windows 8 before denoise -> 0
after. Confirms multi-window submission, not simultaneous disk activity or
speedup. Warm threshold passed, Generate <=22.5 FAILED. No observed whole-phase
win; single runs do not isolate variance/causal regression. No promotion,
rerun or parameter change. Protected/milestone hashes unchanged.

Source preparation 0.028299 s; before-denoise completed source bytes
4,353,687,552, committed 26,038,239,232 bytes with zero registration/DMA.
Final read 25,846,148,360 bytes, source-ready wait 4.974880 s versus prior
5.093906 s, registration 0.517319 s, unchanged DMA 187,588,136,576 bytes and
weight backing 3,858,759,680 bytes. Nested cumulative timers are not additive.
Tickets 736 -> 0 nonblocking; zero native errors/fallback/pressure/mismatch/
quarantine/pool failures. Evictions 53 -> 53, wait 0.1299724 s unchanged;
pool accounting conserved. Pending is unreaped metadata, not measured active
GPU work. Peak CUDA-used-minus-baseline 6.5024 GiB, RSS 29.5051 GiB, monitors
without errors. Budgets unchanged, no full NVMe/linear GPU baseline rerun.

The hypothesis that removing whole-window serialization alone would reach the
target was not supported. That barrier is now implemented/validated, not an
unimplemented gap to diagnose repeatedly. Identify the remaining cold critical
path before another patch; this completed approval authorizes no new runs or
flight/memory increases. Full gate in `docs/native-source-flights-gate.md`.

### Bounded multi-window native source candidate (2026-10-02; pre-execution checkpoint)

Isolated `build-source-flights-20261002` SHA256
`A958C53EB1FFC4CD2B243923373C72C918EA8341CB198C0353BF7A05941EA8EC`.
Dispatcher now submits multiple windows before completion, with fixed retained
flights bounded by existing slots/workers and eight. No new workers/source
storage/budgets or runner policy. Existing chunk queue remains 256. Ready
windows retire independently; native CPU checks at most eight futures with
up to 1 ms idle polling, not GPU synchronization. This latency still needs
performance validation. Partial enqueue/exception/cancel retain ownership until
all chunks drain; close joins before storage release. Dispatcher default OFF.

ON/OFF compiled; selected Python CPU 75 PASS (4.76/4.73 s). Native flight CPU
test: 32 lifecycle iterations plus eight-flight bound, zero failures. Existing
read-wait 32 iterations, scheduler 169 checks, spans 1,278 checks and stream
dependencies PASS. Tests cover metadata/proof callbacks, not real failure
injection or CUDA safety. Window-owned/peak/limit optional public metrics added
without struct ABI changes; normalization/older-export absence CPU checked.
Protected `4AC3D8B...` and milestone `8CAAD072...` hashes unchanged.

Eight CUDA cases collected only, including a new bounded ten-window payload/
clear/recycling case. Proposed consolidated gate in
`docs/native-source-flights-gate.md`: eight small cases, span probe, one unchanged
real-prompt eight-step model run; require actual retained peak >1 <=8, zero final
owned windows/tickets, warm <=1.46 and Generate <=22.5 plus existing safety/image
criteria. No CUDA execution, default promotion or measured speedup yet.

### Unlocked source wait gate: correctness PASS, performance FAIL (2026-10-02)

Approved sequence completed once with DDO `.venv-vmm`: seven CUDA cases PASS
(4.94 s), span probe PASS, one unchanged BF16/real-prompt eight-step model run,
exit 0. Selected DLL/hash/capability verified; output under sibling host
`outputs/ltx_image_modular_vmm/source_unlocked_20261002`. Identical PNG SHA256
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.

| Metric | Unlocked wait | Earlier locked dispatcher | Preserved milestone repeats |
| --- | ---: | ---: | ---: |
| GPU warm s/it | 1.412124 | 1.390072 | 1.412275 / 1.375524 |
| First GPU step s | 6.082628 | 1.693964 | 6.540682 / 6.819729 |
| DDO prepare s | 1.6631 | 7.5827 | 0.8337 / 0.7720 |
| Host denoise s | 15.3018 | 10.7700 | 15.7681 / 15.8085 |
| Generate s | 24.4543 | 25.8173 | 23.2125 / 23.2392 |
| Total s | 45.5254 | 46.6713 | 44.2206 / 43.5638 |

Warm <=1.46 passes; Generate <=22.5 FAILS, and is above accepted milestone
repeats. Global admission lock serialization is fixed, but this single run
does not establish a whole-phase gain or quantify variance. No promotion,
repeat, capacity or default change. Approval exhausted.

Source preparation 0.025003 s (earlier 6.627046), before-denoise completed
source bytes 4,278,190,080; committed storage 26,038,239,232 bytes with no
source registration/DMA yet. Final source bytes 25,846,148,360, source-ready
consumer wait 5.093906 s, registration 0.430762 s; DMA 187,588,136,576 bytes
and weight backing 3,858,759,680 unchanged. Do not sum nested counters.
Tickets 732 -> 0 nonblocking; zero native failures/fallback/pressure/mismatch/
quarantine/pool failures. Evictions 53 -> 53, wait 0.1350962 s unchanged;
pool accounting conserved before/after. Pending is unreaped metadata, not a
measurement of unfinished GPU work. Peak CUDA-used-minus-baseline 6.3424 GiB,
RSS 29.5031 GiB, monitors without errors. Protected/milestone hashes unchanged.

Read-only review confirms `dispatch_reads` still submits one whole window,
waits for it and retires it before taking another. Chunk-level workers alone
do not establish multi-window concurrency. Next design should separate bounded
submission/retirement with retained owners; no attribution of all remaining
cost to this barrier or performance guarantee. Gate details in
`docs/native-source-unlocked-gate.md`; no new runtime patch in this execution.

### Source read wait serialization corrected (2026-10-02; pre-execution checkpoint)

Isolated `build-source-unlocked-20261002` candidate SHA256
`DC4FAA4DE93B13F059E097F08EEA393163D394EFBE8E3EF0AA010D64589F828B`.
Dispatcher wait now releases global ring admission mutex while completion is
pending. Shared completion and active-waiter counts guard slot reuse/unregister/
release; destroy waits for all waiters before freeing storage and its mutex.
Both successful and exceptional completion retire safely. Callers must stop
new submissions before closing a raw handle. No ceilings/runner/worker changes.
Dispatcher-OFF retains original consumable future/locked wait; ON/OFF compiled.

Candidate Python CPU: 74 PASS (4.79 s), dispatcher-OFF 74 PASS (4.69 s);
native read-wait test 32 concurrent
success/failure/close/restart lifecycle iterations, zero failures. Scheduler
169 checks, span ledger 1,278 checks and stream dependency CPU contracts PASS.
Protected `4AC3D8B...` and milestone `8CAAD072...` DLL hashes unchanged.
No CUDA/model run or speedup claim. Proposed seven small CUDA cases, span probe
and one unchanged model run in `docs/native-source-unlocked-gate.md` require
new confirmation; prior failed sequence does not authorize this execution.

### Native source dispatcher gate: correctness PASS, performance FAIL (2026-10-02)

One explicitly approved consolidated sequence completed with DDO `.venv-vmm`:
four source CUDA cases PASS (4.73 s), span probe PASS, then one unchanged
real-prompt BF16 1280x704/seed43/eight-step model run, exit 0. Candidate
`24A2F4F7...`; selected DLL/capability verified. Output under sibling host
`outputs/ltx_image_modular_vmm/source_dispatcher_20261002`; PNG SHA256
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`
is identical to the preserved milestone.

| Metric | Dispatcher | Preserved milestone repeats |
| --- | ---: | ---: |
| GPU warm s/it | 1.390072 | 1.412275 / 1.375524 |
| First GPU step s | 1.693964 | 6.540682 / 6.819729 |
| DDO prepare s | 7.5827 | 0.8337 / 0.7720 |
| Host denoise s | 10.7700 | 15.7681 / 15.8085 |
| Generate s | 25.8173 | 23.2125 / 23.2392 |
| Total s | 46.6713 | 44.2206 / 43.5638 |

Warm <=1.46 passed; whole Generate <=22.5 FAILED. No promotion or repeat.
Peak CUDA-used-minus-baseline 6.2692 GiB (not absolute VRAM), RSS 29.6251 GiB.
Tickets collected nonblocking 732 -> 0; zero faults/admission failures,
temporary tensors, execution-plan mismatches, quarantine and pool failures.
Pool evictions stayed 53 -> 53; wait stayed 0.1262315 s. Mapped accounting
conserved before/after: leased + available + pending + quarantine = mapped.
Pending is unreaped metadata, not evidence of unfinished GPU work.

Before denoise, source read 25,846,148,360 bytes, committed 26,038,239,232,
source registrations/DMA zero. Source preparation took 6.627046 s; final
consumer source-ready wait 0.053206 s and registration 0.818015 s. Final DMA
187,588,136,576 bytes and weight backing 3,858,759,680 bytes unchanged.
These nested timers are not additive. Work moved into prepare; a faster first
step is not an end-to-end improvement.

Code inspection found global ring mutex held over `wait_file_read`'s blocking
`future.get()`; `commit_storage` for the next admission needs that mutex.
This is concrete serialization consistent with the preparation delay, not
exclusive timing attribution. Next correction must separate completion wait
from global admission locking while preserving slot/read lifetime through
partial failures, concurrent waits and clear/close. No CUDA rerun authorized.
Milestone `8CAAD072...` and protected `4AC3D8B...` hashes verified unchanged.

### Native source dispatcher connected (2026-10-02; pre-execution checkpoint)

Isolated candidate `build-source-dispatcher-20261002`, SHA256
`24A2F4F7BEBBD27BC03EB93C9BB87E6582B3152A4C0A4BE59486C61ED06A82DD`.
CMake source-dispatcher option default OFF; span-pool milestone `8CAAD072...`
and protected `4AC3D8B...` unchanged. Generic unit descriptors now reach a
source-only native preparation export when that optional capability is enabled.
Native metadata dispatcher feeds existing eight I/O workers/256 chunk queue;
window queue/attempts are bounded by existing host-cache geometry. No CUDA calls
in the dispatcher, new registration, GPU speculation policy or budget increase.
Actual demand retains cache/ring fallback and registration/DMA lifetimes.

Clear stops/joins outside the cache lock before storage/file release, including
partially submitted I/O. Completion signals retain entry ownership; settled
scheduler references retire before pressure reuse. Startup cleanup occurs only
after temporary view setup exits. Native optional exports/report fields keep
older DLLs usable without eager unit preparation or new runner code.

Compiled ON and OFF candidates. CPU suite: 74 passed with dispatcher ON and OFF;
protected older DLL 73 passed / one optional-export skip. Scheduler 169 checks,
span ledger 1,278 checks and stream-dependency CPU contracts PASS. New small
CUDA source-only/fallback/clear-restart tests: three cases collected, NOT run.
One minimal-store mock regression was corrected; no failing CPU contract left.
CPU/mocks/compilation do not prove asynchronous device safety or faster Generate.

Proposed gate in `docs/native-source-dispatcher-gate.md`: four source cases,
span probe, then one unchanged eight-step model case only after all small PASS.
Warm <=1.46 s/it and whole Generate <=22.5 s proposed; require identical image,
zero errors/fallbacks/quarantine/tickets and no memory anomaly. No CUDA execution
or default promotion yet; no measured performance improvement claimed.

### Cold-source scheduler CPU contract implemented (2026-10-02)

Pure C++ metadata scheduler in `native/include/ddo_vmm_source_scheduler.h`:
immutable-window deduplication, bounded admission, demand priority, explicit
backpressure, stale/foreign-token guards, failure visibility and strong read
ownership through cancellation/close. Required demand may reclaim a cancelled
active read without duplicate scheduling. The coordinator must prove all-chunk
completion and host admission; this scheduler performs no I/O or CUDA and is
not yet connected to the source runtime.

Isolated CPU build `build-source-scheduler-cpu-20261002`: scheduler 169 checks
PASS, existing span ledger 1,278 checks PASS, stream-dependency contracts PASS.
Only CPU executable targets built/run. Milestone `8CAAD072...` and protected
`4AC3D8B...` DLL hashes unchanged. No new benchmark, budget change, runtime gain
or CUDA-safety claim. Next step is the native lifecycle/dispatcher connection
described in `docs/native-cold-source-plan.md`, then separately approved gates.

### Native span-pool repeatability gate passed (2026-10-02)

Two explicitly approved fresh-process model repeats completed, exit 0, using
the unchanged `8CAAD072...` candidate and DDO `.venv-vmm`. Same real prompt,
BF16, 1280x704, seed 43, eight steps, no PAG, RSS/basic GPU timing, weight
ceiling 4 GiB/effective staging 1.5 GiB/source-registration 24.625 GiB/cache
512 MiB. Only output directories changed: sibling host
`outputs/ltx_image_modular_vmm/span_pool_repeat_1` and `span_pool_repeat_2`.

| Observation | Repeat 1 | Repeat 2 |
| --- | ---: | ---: |
| Warm GPU s/it | 1.412275 | 1.375524 |
| First GPU step s | 6.540682 | 6.819729 |
| GPU phase s | 16.426617 | 16.448408 |
| Host denoise s | 15.7681 | 15.8085 |
| Transformer load s | 0.3572 | 0.2840 |
| DDO prepare s | 0.8337 | 0.7720 |
| Generate s | 23.2125 | 23.2392 |
| Total s | 44.2206 | 43.5638 |
| CUDA-used-minus-baseline peak GiB | 6.2856 | 6.1370 |
| RSS peak GiB | 29.5029 | 29.5039 |

Both pass the preapproved <=1.46 s/it repeatability threshold. Including the
initial case, observed warm range is 1.375524–1.412275 s/it, still above the
1.32–1.33 s/it reference; this is limited local reproducibility, not a
statistical/platform/architecture guarantee. Total includes the text phase;
CUDA peak is a sampled delta from baseline, not absolute GPU usage/headroom.

Both PNG hashes equal the initial `877876A4...` image. Selected DLL checked
in each report. GPU timing complete, source tickets 724 -> 0, zero native
fault/admission/source/pool failures, temporary tensors, execution mismatches,
pressure evictions or quarantine. Pool conservation holds before and after;
mapped backing remains 2,543,845,376 B. Pool eviction attempts 53 -> 53 in
both; lifetime eviction wait stays unchanged across denoise at 0.1336463 s
and 0.1351797 s respectively. Monitor errors null; no memory anomaly observed.
These runs do not inject driver failures or establish pressure recovery.

Protected DLL SHA256 `4AC3D8B...` rechecked unchanged. No code, budgets,
default promotion or new CUDA runs: the two-run approval is exhausted.
Next optimization targets remain cold materialization/prepare and the small
warm gap; dedicated native text-encoder integration follows transformer closure.

### Native span-pool consolidated CUDA/model gate passed (2026-10-02)

Approved sequence completed once. Three fresh-process contracts PASS:
interior pointers/512-byte alignment/shared mapping/neighbor payloads;
16 MiB cache-512 MiB cross-stream lifetime + nonblocking renewed-fence reuse;
16 MiB cache-disabled whole-backing retirement (zero live/cached mappings).
Small-span final snapshot: one 2 MiB backing, zero live/reserved/quarantined
leases, zero failures, 1,536 bytes retaining unobserved completion metadata.
This is not evidence of GPU work still running after the synchronization.

Then one real-prompt BF16, 1280x704, seed 43, eight-step/no-PAG/local-files
model run with selected candidate `build-span-pool-20261002`, SHA256 `8CAAD072...`.
Requested weight budget 4 GiB, effective staging 1.5 GiB, source/registration
24.625 GiB, cache 512 MiB; RSS/basic GPU timing, no profiler/fine timeline/step
waits. Exit 0; identical PNG SHA256
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Artifacts: sibling host `outputs/ltx_image_modular_vmm/span_pool_20261002`.

| Observation | Protected control | Exact-size rejection | Native span pool |
| --- | ---: | ---: | ---: |
| Warm GPU s/it | 1.712871 | 1.950957 | **1.387751** |
| First GPU step s | 7.359793 | 8.545900 | 7.481182 |
| GPU phase s | 19.349891 | 22.203549 | 17.195465 |
| Host denoise s | 17.7442 | 22.2041 | 16.5393 |
| DDO prepare s | 0.8657 | 0.8606 | 0.8120 |
| Generate s | 27.3821 | 29.7937 | 24.9113 |
| Text phase s | 19.8818 | 19.1968 | 19.3302 |
| VAE phase s | 1.5518 | 1.6960 | 1.4728 |
| Total s | 48.8927 | 50.7639 | 45.7932 |
| CUDA-used-minus-baseline peak GiB | 6.4694 | 6.0387 | 6.2091 |
| RSS peak GiB | 29.5052 | 29.5045 | 29.5029 |

Observed warm reduction 18.98% relative to protected control; still 4.34–5.13%
above the 1.33–1.32 s/it reference range, not target closure. This is one case,
not statistical repeatability, platform/architecture coverage or automatic
default promotion. First step and prepare remain separate improvement targets.

| Atomic pool gauge | Before denoise | After readback/retirement |
| --- | ---: | ---: |
| Unique mapped backing bytes | 2,543,845,376 | 2,543,845,376 |
| Raw requested bytes | 471,933,252 | 457,064,772 |
| Logical leased bytes | 471,934,464 | 457,065,984 |
| Available bytes | 77,312 | 3,584 |
| Pending completion-metadata bytes | 2,071,833,600 | 2,086,775,808 |
| Quarantined bytes | 0 | 0 |
| Live / reserved leases | 311 / 0 | 308 / 0 |
| Backing count | 4 | 4 |
| Whole-idle backing bytes / count | 134,217,728 / 1 | 134,217,728 / 1 |
| Pool eviction attempts (process lifetime) | 53 | 53 |
| Pool eviction host wait s (process lifetime) | 0.1302295 | 0.1302295 |
| Pool failures | 0 | 0 |

Both partitions conserve unique mapped backing. Logical alignment padding is
1,212 bytes, not 0.58 GiB of per-request VMM rounding. However, total backing
is 2.369141 GiB, with 1.943461 GiB pending/unobserved metadata after readback;
partially occupied backing cannot be unmapped or promised as recoverable VRAM.
Do not claim physical memory reduction by comparing only logical leased bytes
with the old per-record footprint. Snapshot collection does not reap CUDA events;
pending describes retained proof metadata, not measured outstanding GPU work.

No eviction/retirement-wait delta during denoise (53 -> 53 / unchanged 0.1302295 s),
unlike exact-size rejection's 3,182 attempts/12.709596 s delta. No native
fault/admission/source/pool failures, fallback tensors, pressure evictions,
execution mismatches or quarantine. Completion tickets 736 -> 0, GPU timing
complete, monitor errors null. Source reads 25,846,148,360 B, H2D 187,588,136,576 B
and weight backing 3,858,759,680 B unchanged; improvement is not an observed DMA
byte reduction. Existing standby privilege warning remains.

Protected DLL `4AC3D8B...` unchanged. Local rollback milestone saved without
overwriting any binary: `build-milestone-span-pool-8caad072/Release/ddo_vmm.dll`,
same full SHA256 `8CAAD072B45302426EE8854591D9AB0AC893241D06A027FA0A6D8FB6B766021C`.
Native/source/script/CMake snapshot ZIP beside it, SHA256
`0E3671C6C35AB0208E647F192F5C9D18FB7C8497848ECE2AC91780BC383EF342`.
These local checkpoints are not a published package or complete host/Python
environment snapshot. No default/path promotion, budget change or further run.
Transformer persistent-source route is validated in this case; text encoder
still uses host group offload, despite the process-wide allocator also servicing
its ordinary CUDA allocations. Dedicated native text-encoder weight route and
capacity profile remain pending. Driver-error injection/pressure recovery,
broader model/platform support and reproducibility are not closed by these gates.

### Connected native span-pool candidate compiled (2026-10-02; CPU only)

Isolated `build-span-pool-20261002`, CMake `DDO_VMM_NATIVE_SPAN_POOL=ON` (default
OFF), SHA256 `8CAAD072B45302426EE8854591D9AB0AC893241D06A027FA0A6D8FB6B766021C`.
Native allocator entry points now consume shared backing/span leases in this
candidate. Context/device/stream checks, retained CUDA-event ownership,
transactional pointer publication, native pressure/OOM retries and whole-idle
cache retirement implemented. No automatic ceiling increase or runner policy.
Normal reuse queues event dependencies; whole-backing retirement may wait for
completion outside the ledger lock. Failure paths retain/quarantine ownership;
there is no automatic cross-allocator migration after partial submission.

Optional native pool schema v1 and public report/catalog integration distinguish
unique physical backing from logical leases and cached/pending/quarantined bytes.
Standalone footprint/release scopes are omitted, not invented as zero. Actual
export read/invalid capacity tests run without CUDA initialization. Ledger CPU
PASS (1,278 checks), stream dependency CPU PASS, selected Python CPU suite
32 PASS on candidate and 32 PASS on protected DLL. Interior-pointer CUDA probe
prepared, not executed; pressure/failure/reuse performance gates remain pending.
Protected `4AC3D8B...` unchanged; no model run or promotion in this step.
Default-OFF control also compiled in `build-span-pool-control-20261002`;
17 CPU telemetry tests passed there, checking unsupported-pool omission and
legacy report behavior without replacing any saved binary. Pending consolidated
commands/criteria: [native span-pool gate](../docs/native-span-pool-gate.md).

### Native span-ledger CPU contracts completed (2026-10-02)

Header-only `ddo_vmm_span_ledger.h` and standalone C++20 test implement/test
one-backing logical split/reuse metadata, inherited strong fence ownership,
best-fit, guarded publication/retirement, unsubmitted rollback, safe coalescing,
fail-closed quarantine and unique-backing partition accounting. Two-phase backing
release prevents allocation during pending physical retirement; failed retirement
cannot make partially destroyed backing available. Caller must eventually prove
real CUDA stream/event submission; this ledger has no CUDA API calls.

CPU executable PASS (1,270 checks, including repeated conservation checks),
existing footprint and stream-dependency CPU executables PASS. Only this new
standalone target was built; protected DLL `4AC3D8B...` and exact-size candidate
`19AE9C2A...` remain unchanged. No new CUDA/model case, no performance claim.
Not connected to allocator or public reports yet. Real backing/event ownership,
pointer publication, cache/pressure policy and versioned physical-accounting
integration remain necessary before requesting the consolidated CUDA gate.

### Exact-size reuse correctness passed; performance gate failed (2026-10-02)

Approved sequence executed once with `build-exact-cache-20261002` (`19AE9C2A...`):
16 MiB allocator/cache 512 MiB cross-stream lifetime probe PASS, then one
real-prompt BF16, 1280x704, seed 43, eight-step model case, same capacities/RSS/
basic GPU timing as the footprint observer. Exit 0. Report verifies candidate
DLL selection. PNG hash matches `877876A4...`; protected DLL `4AC3D8B...` intact.
Artifacts: sibling host `outputs/ltx_image_modular_vmm/exact_cache_20261002`.

| Observation | Footprint observer (larger reuse) | Exact-size candidate |
| --- | ---: | ---: |
| Warm GPU s/it | 1.756785 | 1.950957 |
| First GPU step s | 7.429325 | 8.545900 |
| GPU phase s | 19.726818 | 22.203549 |
| Host denoise s | 18.1177 | 22.2041 |
| Generate s | 27.4644 | 29.7937 |
| Total s | 48.4887 | 50.7639 |
| Final live mapped bytes | 1,598,029,824 | 1,082,130,432 |
| Final oversized-reuse excess bytes | 515,899,392 | 0 |
| Final cached bytes | 222,298,112 | 534,773,760 |
| Cache eviction attempts before / after | 363 / 363 | 384 / 3,566 |
| Cache eviction host sync s before / after | 0.153293 / 0.153293 | 0.147922 / 12.857517 |

Exact-size eliminates oversized reuse in both snapshots, but introduces 3,182
eviction attempts and 12.709596 s accumulated host eviction synchronization
between snapshots. This is an observed churn/wait regression, not a kernel
time measurement. Reuse savings of 0.480469 GiB in live records are partly
offset by larger retained cache; GPU-used-minus-baseline peak 6.0387 GiB vs
6.4501, not proof of safe new residency budget. Padding remains 625,065,660 B
after denoise. Requested bytes and live count unchanged: 457,064,772 B / 308.
Atomic decomposition consistent; RSS peak 29.5045 GiB, monitor errors null.

Transformer load 0.2710 s, prepare 0.8606 s, text 19.1968 s, VAE 1.6960 s.
No faults/admission/reset/source failures, temporary fallbacks, pressure
evictions or execution mismatches; completion tickets 32 -> 0, GPU timing
complete. Source reads 25,846,148,360 B, DMA 187,588,136,576 B and total weight
backing 3,858,759,680 B unchanged. Fewer pending tickets do not prove more
efficient streaming: synchronous retirement can reduce the pending count.

Not promoted: correctness and excess-zero pass do not outweigh worse throughput
(also above protected 1.712871 s/it). Do not grow the cache/budget or rerun variants
to mask the result. Next design must preserve larger-backing reuse through native
split/suballocation and per-slice completion ownership, rather than exact-size
rejection; safe split/coalescing/release contracts require implementation and
new approval. The prepared exact-size section below is historical, not policy
recommended for production. Approved CUDA sequence is exhausted.

### Exact mapped-size cache candidate prepared (2026-10-02; CPU only)

Changed native cache selection from unbounded larger-block reuse to matching
mapped size and device exactly. Larger cached mappings remain available to
their own size class, not assigned whole to smaller requests. No mapping
splitting/trimming, capacity change, automatic budget increase, new Python
policy or fence removal. A miss uses the existing allocation/pressure/OOM retry
path; this may increase allocation/eviction churn and has no performance approval.
Granularity padding remains unresolved; this change addresses only reuse excess.

Isolated Release `build-exact-cache-20261002`, SHA256
`19AE9C2A84542340E9368F62652966452122DCC55C18D3A395C1448405F49DF9`.
Three C++ CPU contracts PASS (exact cache/device selection, footprint,
stream dependencies). 28 Python CPU tests passed in 5.33 s, including actual
read-only DLL telemetry; no CUDA context initialized by these tests. Protected
DLL `4AC3D8B...` unchanged. The isolated build retains the previous unpromoted
same-stream change and footprint observer. No new CUDA/model run executed.
Pending gate: small allocator reuse followed, only on PASS, by one otherwise
unchanged eight-step model case; require zero reuse excess, consistent report,
correct image and no memory/failure anomaly. Timing/churn must be checked;
zero excess alone does not justify promotion or prove target throughput.

### Allocation-footprint CUDA/model observation completed (2026-10-02)

Approved sequence: fresh-process allocator probe, 16 MiB allocation, 512 MiB
cache, `--verify-stream-lifetime`: PASS, stable address reused, payload correct,
zero live allocations after release. Then exactly one real-prompt BF16 model
run with the isolated footprint DLL `79D58198...`: 1280x704, seed 43, eight
steps, no PAG, local files, RSS monitor and basic GPU timing. Requested weight
budget 4 GiB, effective staging 1.5 GiB, source/registration 24.625 GiB unchanged.
Output directory: sibling host `outputs/ltx_image_modular_vmm/allocation_footprint_20261002`.
Exit 0; PNG SHA256 remains
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Protected DLL hash `4AC3D8B...` unchanged; candidate is not promoted.

| Atomic live-ledger observation | Before denoise (bytes) | After readback/retirement (bytes) |
| --- | ---: | ---: |
| Current requested | 471,933,252 | 457,064,772 |
| Minimum mapped | 1,101,004,800 | 1,082,130,432 |
| Actual mapped | 1,616,904,192 | 1,598,029,824 |
| Granularity padding | 629,071,548 | 625,065,660 |
| Oversized-reuse excess | 515,899,392 | 515,899,392 |
| Live records | 311 | 308 |

Both snapshots satisfy requested + padding + reuse excess = mapped. Final
snapshot: requested 0.425675 GiB, padding 0.582138 GiB, reuse excess 0.480469 GiB;
combined excess 1.062607 GiB (71.4% of live mapped bytes). Cache is separate:
39,845,888 / 222,298,112 bytes before/after. These are not forward peaks and
do not establish safely recoverable capacity, driver-local residency or speedup.
Next native correction candidates are oversized cache admission/selection and
small-request suballocation, with alignment/stream ownership preserved; neither
is implemented or approved for GPU validation by this observation alone.

Warm GPU 1.756785 s/it vs protected control 1.712871 (+2.56% observed), first
7.429325 s, GPU phase 19.726818 s, host denoise 18.1177 s. Transformer load
0.3108 s, DDO prepare 0.8667 s, Generate 27.4644 s, text 19.3092 s, VAE 1.6391 s,
total 48.4887 s. Lower total does not cancel slower warm steps. The build also
contains the unpromoted same-stream change; no pure-observer causal timing claim.
Peak CUDA-used-minus-baseline 6.4501 GiB (not absolute GPU total), RSS 29.5056 GiB.
No fault/admission/reset/source failures, execution mismatches, temporary
fallbacks or arena/source pressure evictions; completion tickets 1324 -> 0,
GPU timing complete, memory-monitor errors null. Same source read bytes
25,846,148,360 and H2D bytes 187,588,136,576 as protected control; weight backing
3,858,759,680 bytes. Existing standby privilege warning remains.
Approved sequence ended; no further CUDA run or budget increase authorized.

### Allocation-footprint observer prepared (2026-10-02; CPU only)

Added optional atomic live-ledger decomposition: current requested bytes,
minimum mapping, actual mapping, granularity padding, oversized-reuse excess
and record count. It does not alter allocation sizing, cache selection, capacity
or fences. Public report integration supports older DLLs by omitting unavailable
fields. Isolated Release build: `build-allocation-footprint-20261002`, SHA256
`79D58198CA1692A0DECF3934E09DC272D3745FA1BB15FE21A7CA6D411E17192A`.
Both standalone CPU contracts passed (footprint and stream dependency);
13 Python CPU tests passed, including actual export validation and invalid output
capacity/null-pointer checks without CUDA initialization.
Against the protected DLL, 25 CPU tests passed (telemetry and profiles),
confirming omission of the unsupported observer without breaking old reports.

Protected DLL SHA256 remains
`4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.
The isolated build also contains the existing unpromoted same-stream candidate;
it is not a pure instrumentation timing control and is not promoted. No CUDA or
performance gate executed here. Actual model padding/reuse excess and safely
recoverable capacity remain unmeasured; the accepted 1.712871 s/it observation
is unchanged. No inference-performance improvement is claimed.

### Memory-ledger full-model control completed (2026-10-01)

Exactly one approved unprofiled run, DDO `.venv-vmm`, RSS explicit, basic GPU
timing enabled; no fine native timeline, diagnostic step waits or profiler.
Protected DLL `4AC3D8B...` selected in report and hash unchanged. Real prompt,
BF16, 1280x704, seed 43, eight steps/no PAG/local files; requested 4 GiB weight,
effective 1.5 GiB staging, source/registration ceilings 24.625 GiB unchanged.
Exit 0. PNG matches protected SHA256
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.

| Observation | Prior protected control | Memory-ledger control |
| --- | ---: | ---: |
| Transformer load (s) | 0.3464 | 0.3230 |
| DDO prepare (s) | 0.9104 | 0.8657 |
| First GPU step (s) | 7.676571 | 7.359793 |
| GPU steady-state (s/it) | 1.718794 | 1.712871 |
| GPU phase (s) | 19.708131 | 19.349891 |
| Host denoise submission (s) | 18.1559 | 17.7442 |
| Generate (s) | 27.7763 | 27.3821 |
| Text phase (s) | 19.8476 | 19.8818 |
| VAE phase (s) | 1.7319 | 1.5518 |
| Full run (s) | 49.4374 | 48.8927 |
| GPU used-minus-baseline peak (GiB) | 6.7146 | 6.4694 |
| Process RSS peak (GiB) | 29.5661 | 29.5052 |

Warm -0.005923 s/it (~0.345%), Generate -0.3942 s, full run -0.5447 s.
Single observations, not statistically demonstrated gain/non-regression.
This validates report integration and no observed deterioration in the selected
run, not achievement of the final target. No allocator algorithm/DLL changed.

All four gauges present in before/after public snapshots:

| Process PyTorch ledger gauge | Before denoise | After readback/retirement |
| --- | ---: | ---: |
| Live allocation count | 311 | 308 |
| Live mapped bytes | 1,616,904,192 | 1,598,029,824 |
| Cached allocation count | 13 | 29 |
| Cached mapped bytes | 39,845,888 | 222,298,112 |

Post live = 1.488281 GiB; cached = 0.207031 GiB. These are not transient peaks,
atomic physical totals or component-only allocation counts. Direct native
weight backing reaches 3,858,759,680 B separately. The observed residual cache
alone is not evidence of multiple GiB recoverable capacity. Per-allocation VMM
granularity rounding is a concrete source-code path to inspect, but requested
versus mapped-byte waste and recoverable headroom have not been measured.
Do not increase residency based on post-run gauges or baseline-subtracted peak.

Fault/admission/source failures, reset failures, temporary tensors/bytes,
execution mismatches and pressure counters zero. Source final tickets zero
(1319 retired), eight GPU steps complete, all memory monitors without errors.
Committed/registered source 26,038,239,232 B; source read 25,846,148,360 B;
H2D 187,588,136,576 B; total mapped weight backing 3,858,759,680 B preserved.
RSS query time for Generate 0.021686 s. Known standby privilege/vision-layer/
Hub warnings unchanged. GPU/host phase times are not additive.
Artifacts: sibling `outputs/ltx_image_modular_vmm/memory_ledger_control_20261001`
metrics/PNG. Approved sequence ended; no automatic rerun or capacity change.

### Public native PyTorch memory-ledger gauges prepared (2026-10-01)

Follow-up real-export CPU contract on the protected DLL: **10 passed in
4.55 s**, including all four memory gauges and release-export output capacity.
No CUDA initialization or inference. Existing host snapshots are before
denoising and after readback/retirement, not transient peak observations;
no claim of recovered/safe weight capacity from those gauges alone.

Existing native exports for live/cached allocation counts and mapped bytes
connected to the native Python binding, VMM report and compact profile. No
C++/ABI, native DLL, allocator scheduling, budget or host-runner change.
Missing exports omit gauges; query exceptions propagate rather than becoming
zero. Per-export locks, no CUDA query/wait/reset. Gauges are process snapshots,
not component-only, peak or physical GPU totals; direct arenas/slots, external
workspaces and records retiring outside the ledger are excluded. Definitions
and limitations in `docs/metrics.md`.

CPU contracts/accounting/library-selection/offline-analysis regression:
**38 passed, 1 skipped in 4.64 s**. Optional real-DLL test skipped. Initial
launch import/tempdir errors resolved with explicit source PYTHONPATH and new
workspace basetemp, not product/test-policy changes. No CUDA or benchmark;
protected active DLL unchanged. GPU performance/non-regression gate not closed.

Saved protected control's mapped weight backing: 3,858,759,680 B (2,717,908,992 B
model arena + 1,140,850,688 B execution slots); eight-step H2D 187,588,136,576 B.
23,448,517,072 B/step is the aggregate mean including cold first step, NOT an
observed warm-step payload. Investigate execution/cache/auxiliary allocations
before reconsidering weight capacity; no larger ceiling or GPU run authorized.

### Event-completion capture — usable timestamps, timing comparison invalid (2026-10-01)

One approved Nsight capture completed; no rerun or native DLL change. CUDA
event-completion tracing enabled, CUDA/NVTX tracing, CPU sampling/context-switch
tracing disabled; protected active DLL SHA256 unchanged (`4AC3D8B...`). DDO
`.venv-vmm`, real prompt/BF16/1280x704/seed 43/eight steps/no PAG; 4 GiB weight
ceiling, effective 1.5 GiB staging and 24.625 GiB source/registration ceilings.
PNG matches protected SHA256 `877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Fault/admission/source failures, temporary fallback, pressure and execution-order
mismatches zero; final tickets zero (160 retired). Monitor errors null;
eight GPU steps complete; native timeline 2444 samples complete/released,
zero dropped/failures/pending. Source committed/registered 26,038,239,232 B,
source read 25,846,148,360 B and H2D 187,588,136,576 B/11,168 copies preserved.

**Command omission:** proposed/approved command omitted `--memory-monitor rss`.
The runner defaulted to full/USS, whereas the prior control used RSS. Generate
RAM-query host time accumulated 52.209703 s versus prior RSS 0.028245 s; this
counter is not additive/exclusive phase cost. Event tracing itself also warns
of overhead/artificial cross-stream dependencies. Thus this is NOT a matched
timing comparison, native performance regression, benchmark or promotion gate.
No single factor is established as the entire slowdown's cause.

Instrumented observations only: GPU first 28.391713 s, warm 3.391150 s/it,
GPU phase 52.129781 s; host denoise 51.9252 s; Generate 65.1183 s; text phase
59.9383 s; VAE 3.0687 s; full run 128.6426 s. GPU baseline-subtracted peak
6.5195 GiB, process USS peak 29.7148 GiB (not RSS, not additional free VRAM).

All 55,066 CUDA event records have nonzero device-completion timestamps. Last
10 s ending at final transformer H2D: kernel union 2.788100 s, H2D union
2.633557 s, intersection 0.121081 s, neither recorded activity 4.699425 s,
largest gap 0.519209 s. Matched waits: 11,971 compute/compute, 130 compute/upload,
490 upload/compute; zero unmatched. For all 130 compute/upload edges, the
following copy begins after recorded event completion: median gap 12.698435 ms,
minimum 11.400410 ms, maximum 531.432495 ms, no negative gap. This excludes
that individual fence as a sufficient explanation in THIS instrumented run;
it does not establish normal-run engine eligibility, driver cause or throughput.

Accounting additionally for all recorded prior upload-stream waits and preceding
kernel/copy/memset activities: 4,160 tail copies, median residual gap 0.002080 ms,
maximum 418.857483 ms; no negative gaps or missing-completion exclusions.
This is a recorded-dependency bound, not proven eligibility: host API start is
not enqueue completion, hidden dependencies/other engines remain possible.
Observer warnings persist: driver 13.4 traced with Nsight libraries 13.2,
software instrumentation; bootstrap helpers have no CUDA, inference child
collected CUDA. Do not use this altered geometry to change normal scheduling.

Offline analyzer extended for actual completion timestamps, missing/ambiguous
records, prior upload activities and inconsistent gaps: **18 CPU tests passed
in 0.07 s**. No runtime/ABI change. Artifacts in sibling host:
`outputs/ltx_image_modular_vmm/native_event_completion_20261001.nsys-rep`,
same-name `.sqlite`, and `native_event_completion_20261001_run` metrics/PNG.
Approved sequence ended. Any further GPU execution requires fresh approval and
an explicit memory-monitor mode; do not repeat automatically.

### Offline cold-cost / engine-overlap review (2026-10-01)

Follow-up slot-dependency review: native acquire waits on the selected slot's
own last-use fence; release retains its lease on event-publication failure.
No normal-path global wait or dependency on the immediately preceding block's
other slot was found. This permits concurrency, not proof of engine overlap.
Read-only `scripts/analyze_cuda_dependencies.py` matches process/device/context
and eventSyncId, rejecting unknown/ambiguous records. Saved Driver tail:
18,638 compute/compute, 207 compute/upload and 687 upload/compute edges;
no unmatched records in this scope. For 207 compute/upload edges, the last
recorded predecessor activity to following-copy gap has median 12.364094 ms,
minimum 11.362514 ms, maximum 121.461979 ms and no negative gaps. These are
geometric lower-bound observations, NOT measured event wait durations or proof
of event completion. CPU analyzer contracts/regressions: 14 passed in 0.07 s.
No new GPU run, runtime change or demonstrated slot-dependency defect.

No new CUDA execution, DLL promotion or parameter change. Saved protected
control: native source-ready host wait 6.843185 s, registration 0.478961 s,
batch submission 7.574608 s. These cumulative/nested counters cannot be added
to phase timings or attributed entirely to the first GPU step. Source reads
25,846,148,360 B once, not once per denoising step. Prepare 0.9104 s excludes
some demand work; moving that work earlier alone is not an end-to-end gain.

Previously captured Driver trace's final 10 s: kernel union 5.089597 s,
H2D union 4.859931 s, intersection 0.082586 s. This is engine activity, not
the overlap of native forward/upload windows. Submitted host work is not
proof of simultaneous GPU eligibility. Investigate per-slot reuse fences
and upload/consumer dependencies before changing the single-upload-stream,
two-slot schedule; no stream-count or execution-granularity fix is yet proven.
Protected DLL and prior images/results remain unchanged. New GPU gates require
fresh approval; same-stream candidate remains unpromoted.

### Same-stream reuse full-model pair — no observed performance gain (2026-10-01)

Executed exactly the approved control then candidate, both without profiler,
with basic GPU timing enabled and native fine timeline/step waits disabled.
DDO `.venv-vmm`, real prompt/BF16/1280x704/seed 43/eight steps/no PAG;
4 GiB weight ceiling, effective 1.5 GiB staging, source and registration
ceilings 24.625 GiB unchanged. Control safety/image/memory inspected before
candidate. Both exit 0; selected native library paths verified in reports.

| Observation | Protected control (4AC3D8B...) | Candidate (4FC5B2F5...) |
|---|---:|---:|
| Transformer load (s) | 0.3464 | 0.3135 |
| DDO prepare/build plan (s) | 0.9104 | 1.0684 |
| First GPU step (s) | 7.676571 | 7.468226 |
| GPU steady-state (s/it) | 1.718794 | 1.790891 |
| GPU phase, dependencies included (s) | 19.708131 | 20.004463 |
| Host denoise submission (s) | 18.1559 | 18.3774 |
| Generate, full phase (s) | 27.7763 | 28.1763 |
| Text encoding phase (s) | 19.8476 | 19.5423 |
| VAE phase (s) | 1.7319 | 1.8913 |
| Full run (s) | 49.4374 | 49.6903 |
| GPU used minus baseline peak (GiB) | 6.7146 | 6.8285 |
| Process RSS peak (GiB) | 29.5661 | 29.5661 |

Candidate warm +4.19%, GPU phase +0.296332 s, Generate +0.4000 s, full run
+0.2529 s and observed GPU peak +0.1139 GiB. One sequential pair cannot
separate variance from a causal regression, but it **does not demonstrate a
performance improvement or close the non-regression gate**. Candidate not
promoted; protected active DLL remains unchanged. The original C++ candidate
source is still pending evaluation: do not rebuild/promote it over the protected
`build/Release` DLL on the strength of the passing correctness suite alone.
Current control's 1.718794 s/it is a saved single-run observation on existing
code, not a code-induced improvement or statistical baseline replacing prior
protected results. No additional runs or parameter changes were made.

Both PNGs match protected SHA256
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Fault/admission/source failures, temporary fallback, execution-plan mismatches,
pressure, sampler allocator release/eviction deltas and monitor errors all zero.
Final source tickets zero (control 1319 retired, candidate 1318 retired);
GPU timing complete with eight steps. Committed/registered source
26,038,239,232 B, source read 25,846,148,360 B, transformer H2D
187,588,136,576 B and weight backing 3,858,759,680 B identical. Standby purge
still skips missing privilege; existing text-only unused vision-layer and
deprecated Hub option warnings unchanged. GPU/host phase times are not additive.
Artifacts: sibling host `outputs/ltx_image_modular_vmm/`
`same_stream_control_20261001` and `same_stream_candidate_20261001` (metrics/PNG).
Next investigation should remain evidence-led on real transfer/compute overlap
and cold source/prepare cost, not assume redundant-command count is the dominant
bottleneck. New GPU execution requires fresh approval.

### Corrected same-stream candidate CUDA gate passed (2026-10-01)

Freshly approved repeat after aligning native library selection: **47 passed
in 6.10 s**, exit 0, including three direct actual-address cache reuse cases,
timeline/timing, execution buffers, transport, fast runtime forced-pressure
fallback and source-cache contracts. Candidate SHA256 unchanged:
`4FC5B2F5A017E2ECA9AE5D4126562EB8A2E22DC32D69B49BF53784760D017DEB`.
Protected active DLL unchanged:
`4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.
This closes the selected corrected CUDA correctness gate, not full-model
performance/non-regression. No benchmark executed and candidate not promoted.
Next proposed scope: one unprofiled eight-step real-prompt control and one
candidate with identical budgets, basic GPU timing enabled, fine native timeline
and step waits disabled. Inspect control image/safety/memory before candidate;
stop on failure/anomaly. Requires separate approval. Preserve previous artifacts
and compare Generate/first/warm/phase/total, not just one headline number.

### Candidate CUDA gate stopped — DLL selection mismatch corrected in CPU (2026-10-01)

Approved candidate suite stopped at first failure: **18 passed, 1 failed in
5.99 s**. The three new direct same/cross/legacy stream actual-address reuse
contracts passed on candidate `4FC5B2F5...`. The failure was the existing
forced-pressure fallback test: numerical result correct, expected fault absent.
Static inspection found `_load_native_allocator(None)` prioritized the local
default DLL over `DDO_VMM_LIBRARY_PATH`, whereas the test's guard loaded the
explicit candidate. Guard and arena could therefore operate on distinct native
global state. This mixed-library suite is **not a passing candidate gate**;
other runtime passes must not be attributed to the candidate DLL.

Corrected selection precedence to explicit argument, environment, then existing
local default, consistent with bootstrap and native binding. Default behavior
without overrides is unchanged; invalid explicit selection is not silently
replaced. Added five CPU selection contracts and an explicit library-identity
assertion before the CUDA fallback guard. CPU suite (selection/bootstrap/backend/
presets/release observations): **51 passed in 4.99 s**. Active DLL and candidate
hashes unchanged. No automatic CUDA rerun, benchmark or promotion. Full corrected
CUDA gate requires fresh confirmation; safety/performance validation remains open.

### Same-stream cache reuse candidate prepared — CPU only (2026-10-01)

Native allocator now stores the CUDA-reported unique stream ID/device with
each successful last-use fence. Reuse omits cuStreamWaitEvent only when both
identities are known and equal. Failed identity queries use the previous wait
path; cross-stream dependencies remain unchanged. Original fence survives
publication failure and is retired only on a renewed completion proof. Move,
reset and synchronous fallback preserve/clear identity with the fence lifetime.
No residency/cache ceilings, Python policy, public ABI or protected preset change.

Separate Release build with the existing CUDA 13.2 toolkit succeeded:
`ddo-vmm-allocator/build-same-stream-20261001/Release/ddo_vmm.dll`, SHA256
`4FC5B2F5A017E2ECA9AE5D4126562EB8A2E22DC32D69B49BF53784760D017DEB`.
Active protected DLL remains unchanged (`4AC3D8B...` full hash below).
Native CPU contracts PASS; selected Python CPU suite **41 passed in 5.79 s**.
Prepared, syntax-checked only: `tests/test_torch_cache_stream_cuda.py`, three
small actual-address cache reuse cases (same, cross and legacy stream), checking
last-writer order with an in-flight producer. No CUDA test or benchmark executed.
This reduces known redundant commands, but no speedup/non-regression is claimed.
CUDA ordering gate must pass before any candidate full-model case; confirmation
required. Do not replace the protected DLL based on compilation/CPU results.

### CUDA Driver dependency capture completed (2026-10-01)

One explicitly approved Nsight Systems 2025.6.3 capture; DDO `.venv-vmm`,
unchanged DLL `4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`,
real prompt/BF16/1280x704/seed 43/eight steps/no PAG. CUDA+NVTX tracing,
event-completion tracing disabled, CPU sampling/context-switch tracing disabled;
no Kineto wrapper or diagnostic step waits. GPU/source/staging ceilings unchanged.
Exit 0; protected PNG SHA256 unchanged. Zero fault/admission/source failures,
temporary fallback, pressure, sampler allocator release/eviction deltas and
monitor errors. Final source tickets zero; timeline 2444 samples complete,
zero dropped/failures/pending, event cleanup released. Source committed and
registered 26,038,239,232 B, source read 25,846,148,360 B, transformer H2D
187,588,136,576 B/11,168 copies, weight backing 3,858,759,680 B unchanged.

**Profiled observations, not benchmark:** GPU first 7.756089 s, warm
1.807403 s/it, GPU phase 20.407916 s, host denoise 18.906883 s; Generate
28.7847 s, text encoding 20.2207 s, VAE 1.6538 s, total 50.7405 s.
Baseline-subtracted GPU peak 6.6769 GiB; process RSS peak 29.8791 GiB.

Last ten seconds ending at the final transformer H2D (upload stream 17):
kernel union 5.089597 s, H2D union 4.859931 s, intersection 0.082586 s;
neither recorded activity 0.133058 s, largest gap 0.000409 s. CUDA Driver
calls are present in the export's RUNTIME table, alongside Runtime APIs.
Event-sync identities match records to stream waits: 18,638 compute-to-compute
waits, 207 compute-to-upload waits and 687 upload-to-compute waits in this
window. Their respective summed host call intervals are 0.029520, 0.000409
and 0.000635 s; these are **not GPU stall durations**. Native warm GPU windows
measure compute/upload dependency 0.002009 s and upload/slot 0.000437 s.
Host launch-call union remains 6.358786 s. No stream/context synchronization
or backing create/map/unmap/release was recorded in this tail; 6,464
cuEventSynchronize calls total only 0.003550 s host time there.

Correlation-ID/process matched H2D submissions show 7,624/7,656 copies had a
future kernel whose host launch had already returned before the copy began.
Median copy host-return-to-GPU-start lag 1371.113 ms. Submission does not prove
dependency eligibility; this rules against a simple absence of queued host
work, not against all scheduler/dependency causes. Redundant same-stream cache
waits are confirmed, but removing them is not yet a demonstrated performance
fix. Any correction must retain cross-stream dependencies and original fences
until publication/renewed completion, with special-stream identity handled safely.

Capture caveats: Nsight warns installed CUDA driver 13.4 is newer than its
supported 13.2 trace libraries; software-instrumented trace used. CUDA event
device completion timestamps are absent (event tracing intentionally disabled).
No NVTX events collected; bootstrap helpers have no CUDA events, while the
inference child reports CUDA injection and collection. Do not claim complete
causal proof, unprofiled non-regression or driver-managed residency from this run.
No allocator/preset change or automatic rerun. Artifacts in sibling host:
`outputs/ltx_image_modular_vmm/native_driver_dependencies_20261001.nsys-rep`,
`native_driver_dependencies_20261001.sqlite` (host root), and
`outputs/ltx_image_modular_vmm/native_driver_dependencies_20261001_run` JSON/PNG.

### Cadence causal control completed — insufficient to restore overlap (2026-10-01)

Approved consolidated CUDA suite: **44 passed in 6.17 s**. Executed exactly one
profiled case with `--ddo-diagnostic-step-wait`, current DLL unchanged:
`4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.
Exit 0, protected PNG identical, eight GPU steps, diagnostic waits exactly eight
(10.723414 s host wall), benchmark_eligible=false. Timeline complete/released,
2444/4096 samples, no lost/failed/pending/invalid samples. Zero fallback/fault/
admission/source failures/pressure, final tickets zero, monitors without errors;
source/backing/read/DMA capacities and bytes unchanged. Sampler allocator
release/eviction counter deltas remain zero.

Instrumented values only: first GPU step 7.571366 s, warm 1.795861 s/it,
phase 20.142836 s, host denoise 20.1484 s, Generate 28.4290 s, full run
50.3133 s; GPU baseline-subtracted peak 6.6288 GiB, RSS 29.7698 GiB.

| Last 10 s activity geometry | Current trace, no wait | Event-step-wait control |
|---|---:|---:|
| Kernel union (s) | 5.230698 | 5.166636 |
| HtoD union (s) | 4.727686 | 4.889390 |
| Actual kernel/HtoD intersection (s) | 0.100948 | 0.194569 |
| HtoD without recorded kernel (s) | 4.626738 | 4.694821 |
| Neither activity (s) | 0.142564 | 0.138543 |
| Host cudaLaunchKernel union (s) | 6.763309 | 0.183407 |

Upload stream confirmed by identical 187,588,136,576 B / 11,168 copies (ID 17).
Tail event-wait union is 8.685341 s, replacing launch-call backpressure rather
than removing GPU work. Host/runtime/driver and GPU timings are overlapping
observations, never additive exclusive costs. Equal wall tails contain unequal
work; kernel totals are not an equal-work throughput comparison.

Conclusion: the control changed host cadence as intended, but did **not**
restore the earlier 2.733414 s overlap or target throughput. GPU warm -3.09%,
phase -0.349338 s, Generate -0.0836 s, full run +0.4744 s against the previous
profiled case: no integral performance gain. A single diagnostic pair does not
establish unprofiled benefit or separate variance. Do not promote step waits or
implement a native cadence policy on this evidence. The hypothesis that simple
step-boundary throttling is sufficient is not supported. Next review must trace
native upload/slot/compute dependency ordering, including potential observation
effects, before proposing a correction; no further GPU tests or ceiling changes
were made. Diagnostic remains opt-in; stable and protected routes unchanged.
Artifacts: host `outputs/ltx_image_modular_vmm/native_cadence_trace_20261001`
(`cuda_trace.json`, `run/` JSON/PNG).

### Submission-cadence causal control prepared — CPU only (2026-10-01)

Added explicit DDO timer `diagnostic_step_wait` (default false) and thin host
`--ddo-diagnostic-step-wait`. Requires enabled native timing on VMM-fast through
the public result; protected/no-op routes reject it. Waits only on each compute
boundary event, with no scalar conversion or device-wide synchronization;
reports attempted calls/host time and `benchmark_eligible=false`, with console
warning. No preset, native DLL, cache, residency, transfer or budget changes.

Focused CPU suite **31 passed, 1 skipped in 5.75 s** (optional built-DLL test
not selected by environment); host callback/validation suite **4 passed in
0.06 s**. Mock events prove exactly one wait per step and zero start/end waits;
default timing still does not wait. Small CUDA contract prepared, not executed.
Proposed consolidated gate: existing selected CUDA suite plus new boundary
contract, then one profiled case identical to current activity diagnostic except
the explicit cadence control. No CUDA/performance validation yet; approval
required and no automatic repetitions.

### Current-DLL activity diagnostic — serialization/backpressure hypothesis (2026-10-01)

Executed the approved single Kineto+native-timeline diagnostic with unchanged
workload/capacities and DDO `.venv-vmm`, DLL
`4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.
Exit 0, protected PNG hash unchanged, eight GPU steps complete; no faults,
admission/source failures, fallback or pressure; final tickets zero; monitors
without errors. Backing/source bytes/read/DMA remain those of the paired gate.
Timeline complete, 2,444 samples, zero dropped/failures/pending, cleanup released.

**Profiled, not benchmark:** GPU warm 1.853196 s/it, first 7.519797 s,
phase 20.492174 s; host denoise 18.9417 s; Generate 28.5126 s; full run
49.8389 s. Peak GPU used minus baseline 6.4547 GiB; process RSS 29.7110 GiB.
No performance improvement or non-regression claim from instrumented timings.

Upload stream identified by payload, not assumed ID: actual ID 17, whole-run
187,588,136,576 B and 11,168 HtoD copies. Latest versus earlier trace, both
clipped to last 10 s of that stream's HtoD span (not exact step boundaries):

| Recorded activity | Earlier profiled run | Current profiled run |
|---|---:|---:|
| Kernel union (s) | 6.416869 | 5.230698 |
| HtoD union (s) | 6.127331 | 4.727686 |
| Kernel/HtoD intersection (s) | 2.733414 | 0.100948 |
| HtoD without recorded kernel (s) | 3.393918 | 4.626738 |
| Neither recorded activity (s) | 0.189213 | 0.142564 |
| Host cudaLaunchKernel union (s) | 0.179496 | 6.763309 |
| Largest clipped cudaLaunchKernel call (s) | 0.0001213 | 0.0423411 |

Current addmm-associated kernels sum to 4.070735 s and attention-forward kernels
0.427728 s. Counts differ because equal wall-clock tails contain different
amounts of work; duration totals cannot be compared as equal-work kernel speed.
Native warm forward/upload-window intersection is still 6.023273/6.116104 s,
illustrating why window overlap alone cannot establish engine overlap.

Earlier tail has eight cudaStreamSynchronize calls (clipped union 8.062375 s),
all associated by profiler External id with `aten::_local_scalar_dense`.
Current tail has no such explicit synchronization. Current callback resolves
display timesteps on CPU before sampler rather than reading CUDA scalars per
step. Long host launches alongside near-serialized GPU activities make submission
backpressure/cadence a concrete hypothesis, **not proven causal attribution**.
Runs also differ in DLL and detailed recorder; profiler itself can perturb them.
Do not restore scalar reads or blanket synchronization in the official route,
switch attention, or change cache/residency on this evidence alone.

Next causal control should change only an explicitly diagnostic submission
boundary while retaining the current DLL/recorder/workload, before implementing
any bounded native queue policy. A new CUDA scope requires approval. No second
GPU run was executed. Offline analyzer now includes per-API clipped unions,
without adding nested driver/runtime rows as exclusive CPU time; **6 CPU tests
passed in 0.04 s**. Artifacts:
host `outputs/ltx_image_modular_vmm/native_activity_trace_20261001/cuda_trace.json`
and `run/` metrics/PNG.

### Offline activity review — no new CUDA run (2026-10-01)

Added generic `scripts/analyze_cuda_trace.py`: JSON/standard-library only,
interval unions/intersections clipped to the selected HtoD stream's activity
span, optional last-N-second window, explicit device isolation, operator
association by profiler External id. **5 CPU tests passed in 0.04 s**.
No allocator/runtime/runner or benchmark parameter changes.

Analyzed the existing `completion_timing_trace_20261001/cuda_trace.json` with
`--upload-stream 17 --last-seconds 10`. That stream has the expected whole-run
187,588,136,576 B and 11,168 HtoD copies. In the selected 10 s:

| Recorded activity geometry | Seconds |
|---|---:|
| Kernel union on device 0 | 6.416869 |
| Selected-stream HtoD union | 6.127331 |
| Kernel/HtoD intersection | 2.733414 |
| HtoD without recorded kernel activity | 3.393918 |
| Neither selected HtoD nor kernel activity | 0.189213 |
| Largest such gap | 0.023591 |

Kernels associated with `aten::addmm` sum to 4.956772 s; attention-forward
kernels sum to 0.521386 s. Summed operator durations are not exclusive critical
path attribution. This is the **older profiled DLL/run**, not the current
uninstrumented benchmark, and this tail is not an exact steps-2–8 interval.
No extrapolation of its timings into the latest run or performance claim.

Important refinement: native *upload/execution-window* intersection is not
*copy/kernel* intersection. Even while a forward window remains open, kernels
may be absent or affected by transport. Tiny measured explicit dependency
windows therefore do not exclude HtoD as a critical-path contribution. Current
evidence does not justify adjusting cache/slots again, nor switching attention.
Next approved diagnostic should obtain current engine activities once with the
same workload/ceilings and check GEMMs, HtoD and their overlap before a corrective
runtime patch. No new GPU run executed during this review.

### Native timeline contracts and approved paired benchmark completed (2026-10-01)

Corrected consolidated CUDA suite: **43 passed in 5.96 s** with DDO
`.venv-vmm`. Executed exactly one control and one detailed case, without Kineto:
BF16, real prompt, 1280x704, seed 43, eight steps, GPU weight budget 4 GiB,
effective staging 1.5 GiB, source/registration ceilings 24.625 GiB. DLL SHA256
`4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.

| Observation | Control (timeline off) | Detailed timeline |
|---|---:|---:|
| GPU first step (s) | 6.550243 | 7.094968 |
| GPU steps 2–8 mean (s/it) | 1.819584 | 1.813431 |
| GPU sampler phase (s) | 19.287336 | 19.788992 |
| Host denoise call (s; not GPU completion) | 17.6115 | 18.1969 |
| Generate including setup/readback (s) | 27.0564 | 27.6606 |
| Text encoding phase (s) | 19.8633 | 20.9070 |
| VAE phase (s) | 1.8312 | 2.4330 |
| Full run (s) | 48.8310 | 51.0807 |
| Peak GPU used minus baseline (GiB) | 6.6817 | 6.6460 |
| Peak process RSS (GiB; not USS) | 29.5052 | 29.5748 |

Both outputs have protected PNG SHA256
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Both report eight complete GPU steps, final tickets zero, no faults/admission/
source failures, temporary fallback or pressure evictions, and null monitor
errors. Native weight backing remains 3,858,759,680 B; committed/registered
source remains 26,038,239,232 B; source reads 25,846,148,360 B; transformer DMA
187,588,136,576 B. No ceilings, residency or transport policy changed.

Detailed recorder gate PASS: complete, 2,444 samples of capacity 4,096,
dropped/failures/pending zero, cleanup released, all signed offsets finite and
ordered. Types: 404 upload-batch windows, 1,304 compute/upload dependencies,
342 upload/slot dependencies, 384 execution-unit windows and 10 step markers.

Warm timeline covers 12.694026 s. Union of execution-unit windows: 12.572607 s
(336 units, mean 37.418 ms); only 0.121419 s lies outside those windows.
Upload-batch window union: 6.105212 s, intersection with execution windows:
6.001095 s (98.295%). Marked compute/upload dependency windows total 0.001710 s;
upload/slot dependency windows 0.000386 s. Sampler deltas of allocator reset
sync calls/time/failures and cache-eviction attempts/bytes/time are all zero.

Interpretation: recorded dependency stalls and temporary-cache release are not
large observed contributions in this case; upload/forward windows already
overlap. The remaining time is predominantly inside execution-unit windows.
These are stream windows, **not kernel/DMA engine activity**: they do not rule
out host launch gaps within a forward, memory-bandwidth contention, or transport
effects on kernel duration. Do not label the 12.572607 s as pure computation.
Next investigation must distinguish those interior costs before changing cache,
slot scheduling or residency again.

Detailed versus control: warm mean -0.34%, GPU phase +0.501656 s, Generate
+0.6042 s, full run +2.2497 s. One pair cannot separate recorder overhead from
cold/environment variance. No performance gain or integral non-regression claim;
timeline remains opt-in. Previous uninstrumented 1.763602 s/it is not surpassed.
Artifacts in host `outputs/ltx_image_modular_vmm/native_timeline_control_20261001`
and `native_timeline_candidate_20261001`, each with PNG and metrics JSON.

### Corrected timeline retry stopped at integration fixture (2026-10-01)

Approved CUDA retry: **2 passed, 1 failed in 5.02 s**. Distinct-stream pending
ownership/cleanup and bounded capacity/stale-token contracts passed. Public
integration produced a complete/released timeline but lacked upload-batch and
compute/upload-dependency windows. Sequence stopped; no benchmarks executed.

Code review identified a fixture route mismatch: legacy source-cache slots do
not enable `native_source_cache_model_windows`, which selects persistent native
execution buffers/batches. The fixture now uses the existing execution-buffer
gate geometry (eight generic 512x512 linear units, 6 MiB aggregate GPU ceiling,
2 MiB staging, 64 MiB model-source capacity). It explicitly checks native batch
submissions and execution-buffer uploads before timeline kinds. No production
runtime/DLL or benchmark parameters changed. Syntax check PASS; corrected CUDA
fixture not yet executed. Full remaining gate requires renewed approval.

### Native timeline CUDA gate stopped; wrapper/test correction (2026-10-01)

Approved sequence executed with DDO `.venv-vmm` and Release DLL
`4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.
Allocator probe PASS: async lifetime, nonblocking reuse and renewed consumer
fence verified. Selected CUDA suite stopped at its first test: **1 failed in
4.70 s**. No control or detailed benchmark executed.

The test assumed that a pending snapshot guaranteed a pending subsequent close;
GPU completion between calls makes that assertion invalid. Review also found
that direct wrapper close could release events without caching the final report.
Wrapper close now uses the nonblocking snapshot/cleanup path, preserving its
report. The CUDA contract leaves a scope unended for deterministic pending
ownership, then ends it before the explicit test synchronization boundary;
the artificial GPU sleep was removed. No native DLL, scheduling, residency,
budget or benchmark parameter changed.

Corrected CPU timeline suite **13 passed in 4.43 s**, including pending retention
and completed explicit-close report preservation. CUDA corrections remain
unvalidated; fresh approval required before rerun. Latest measured steady state
remains **1.763602 s/it**; no performance or non-regression claim for this patch.

### Native stream-window recorder prepared — CPU only (2026-10-01)

Implemented opt-in native timeline-v1, independent of existing lifetime fences:
bounded preallocated timing events, generation-scoped tokens, pending/failure/
saturation reporting, explicit seal and nonblocking close. Context-scoped one
session/owner at a time; retained owners do not silently disappear or permit
unbounded replacement sessions. Off mode performs no CUDA timeline queries.
Native markers cover upload batch windows, compute/upload waits and upload/
last-consumer waits for execution slots. Generic adapter markers cover unit
forward windows before cleanup/unpin. Public DDO timer adds native step markers
and returns normalized JSON through the existing metrics report. New host flag
`--ddo-native-timeline` defaults off and requires public native/GPU timing.
Budgets, prefetch, slot count and lifetime events remain unchanged.

Signed offsets and union/intersection reduction are CPU-tested; summaries
explicitly describe window geometry, not DMA/kernel activity. Cold source work
and host launch gaps remain inside relevant windows. Detailed instrumentation
overhead is not measured yet; a control is required. No performance gain claimed.
Definitions, missing-byte semantics and context/concurrency limits in
`docs/metrics.md`. Capacity defaults to 4096 samples, no automatic increase.

Release build PASS; C++ stream-dependency/token/capacity CPU contracts PASS.
DDO CPU suite **79 passed in 4.57 s**, including real DLL disabled/invalid-call
contracts without CUDA initialization; host CPU **5 passed in 4.68 s**.
Prepared three new CUDA contracts for distinct streams/bidirectional waits/
pending retention, bounded capacity/stale tokens, and public file-backed source
integration/output parity. Initially not executed; see the stopped gate above.
No CUDA benchmark or profiler run
with this recorder; the last measured steady state remains **1.763602 s/it**.

Release DLL SHA256
`4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.
Previous validated DLL retained as
`build/Release/ddo_vmm.pre-native-timeline-20261001.dll`, SHA256
`79EF18CBE27DE535C950658D36AF369E86ECF204511223A7CABC7F6DB3888F16`.
Do not promote this compiled candidate before its CUDA/performance gates.

### Approved release-observation sequence — no denoise cache eviction (2026-10-01)

Executed the approved sequence once with DDO `.venv-vmm` and Release DLL
SHA256 `79EF18CBE27DE535C950658D36AF369E86ECF204511223A7CABC7F6DB3888F16`:
16 MiB/128 MiB cache allocator probe PASS (asynchronous lifetime, nonblocking
reuse and renewed consumer fence); selected CUDA contracts **40 passed in
5.94 s**; one eight-step real-prompt benchmark without profiler, exit 0.
BF16/1280x704/seed43/no-PAG, GPU budget 4 GiB, source ceilings 24.625 GiB,
registered staging 1.5 GiB unchanged. No repeat, ceiling change or promotion.

| Measurement | Previous event-timed unprofiled | Release-observation unprofiled |
| --- | ---: | ---: |
| First complete GPU step (s) | 7.500963 | 7.617333 |
| Complete GPU steps 2–8 mean (s/it) | 1.796179 | **1.763602** |
| Complete compute-stream phase (s) | 20.074572 | **19.962551** |
| Host denoise call, excludes pending completion (s) | 20.0747 | 18.3214 |
| Generate with readback/cleanup (s) | 28.2962 | 28.4103 |
| Text-encoding phase (s) | 19.7489 | 19.9230 |
| VAE decode phase (s) | 1.7633 | 2.0092 |
| Total full pipeline (s) | 49.8867 | 50.4205 |
| Sampled GPU peak, baseline-subtracted (GiB) | 6.6675 | 6.6884 |
| Sampled process RSS peak (GiB) | 29.5667 | 29.5002 |

GPU interval is complete, eight steps, range for steps 2–8
1.730525–1.804066 s, tail 0.000002 s. After removing per-step display scalar
readback, host call can return with GPU work pending. Its 18.3214 s and the
0.178573 s host callback interval are **not completed sampler timings**. The
existing CPU latent readback establishes completion before final metrics.
Steady mean improved 1.81%, GPU phase 0.112021 s, while first step is slower
by 0.116370 s. One observation does not establish statistical significance or
attribute the small improvement exclusively to the logging change. Generate
is 0.1141 s slower and full pipeline 0.5338 s slower (1.07%); text/VAE account
for 0.1741/0.2459 s increases. End-to-end non-regression gate **not closed**;
retain prior candidates/DLLs, no defaults promoted. Target ~1.3 s/it unmet.

**Decisive native observation:** public pre-denoise and post-readback counters
are identical for all six release observations: delta reset sync calls=0,
reset sync seconds=0, reset failures=0, cache eviction attempts=0, attempted
eviction bytes=0, eviction sync seconds=0. Process totals (367 calls /
0.223327 s, 363 cache eviction attempts / 7,277,117,440 attempted bytes /
0.175393 s eviction sync) occurred before denoise. Do not attribute those
cumulative totals to sampler time. Temporary-cache release/eviction waits
are not an observed bottleneck in this measured phase. Unmeasured event-record
fallback, mapping costs and GPU dependency service are separate questions;
do not remove necessary lifetime barriers based on these counters.

PNG SHA256 matches the protected image:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Phase collection **1312 -> 0 tickets**. Same 187,588,136,576 H2D bytes,
25,846,148,360 source-read bytes, 388 registrations / zero unregistrations,
24.25 GiB committed/registered source and 3.59375 GiB native weight backing.
376/376 prefetch hits, 8 execution-plan passes. Reported fault/read/reset failures,
admission denials, order mismatches, pressure evictions, registration waits and
temporary fallbacks are all zero;
monitors report no errors. Standby purge privilege unavailable, no OS policy
change. No full-model post-close ledger snapshot or protected-route benchmark
was added to this sequence.

Source event creation/reuse cumulative counts: **1344/9824**, versus 7558/3610
in previous unprofiled event-timed run; same 11,168 transfers. This confirms
more reuse in this observation, not a proportionate performance gain or live
event count. Native batch-submit host scope 7.535698 s and source-ready
6.816491 s remain nested, cold-inclusive and non-additive.

Artifacts: sibling
`outputs/ltx_image_modular_vmm/release_observation_candidate_20261001/` PNG/metrics.
Commands and approved limits in `docs/native-source-cache-plan.md`.
Next investigation should isolate actual H2D/compute service and overlap
without the profiler perturbation; do not pursue temporary-cache eviction
as an established warm bottleneck or infer the unprofiled overlap from an
earlier profiled trace. No further run is approved by this completed sequence.

### Progress logging / native release observation patch — CPU only (2026-10-01)

Existing-artifact review found the profiled/unprofiled event-phase difference
concentrated in steps 2–8: first steps 7.499001/7.500963 s, phase difference
2.732176 s, identical H2D/source bytes. Aggregate host batch submission differs
by only 0.051257 s; source-ready differs by 0.166689 s. Those nested scopes
do not explain or establish the cause of the warm difference. No unprofiled
GPU activity timeline exists for attributing it to overlap, GPU service or clocks.

The host progress callback converted a device timestep to a Python scalar.
The current trace shows seven long `aten::item` operations after scheduler
arithmetic, with nested stream synchronization totaling 8.064509 s. The source
path identifies the per-step display conversion as the repeated scalar read;
the scheduler also has an initial index scalar conversion. These waits include
GPU work and are **not** 8 s of recoverable idle overhead or proof of the
profile/unprofiled difference. Display scalars are now resolved once before
the measured denoise scope; scheduler/model computation remains unchanged.
That preparatory CPU readback is not free and must remain part of phase/total
accounting. Callback CUDA event markers and existing memory observation remain.

Native allocation release now observes existing completion synchronization
host time and cache-eviction attempts/bytes; no added CUDA query, synchronization,
cache policy or ceiling change. Public DDO reports receive optional cumulative
process-lifetime counters; old DLLs omit unavailable counters. Definitions and
scope limits are in `docs/metrics.md`. Necessary completion barriers remain intact.
The event-timing harness also saves a public pre-denoise report to allow phase
deltas without interpreting native metric names in the runner.

Release build succeeded; C++ CPU stream-dependency contracts PASS.
DDO CPU contracts **64 passed in 4.83 s**, including real export output-capacity
checks without CUDA initialization. Host CPU callback/config contracts
**4 passed in 5.10 s** after correcting a mock signature in the initial failed
host test. These are not CUDA/performance validation.
New DLL SHA256
`79EF18CBE27DE535C950658D36AF369E86ECF204511223A7CABC7F6DB3888F16`.
Previous DLL preserved as `build/Release/ddo_vmm.pre-release-observation-20261001.dll`
(SHA256 `32531DBE284E8FC62F522130104AD2AD06EE7AD949D8601232E3DA3E03FFD917`).
No CUDA run, benchmark, budget change or default promotion in this patch.
The last unprofiled milestone remains **1.796179 s/it**.

### Approved unprofiled event-timed candidate — target not reached (2026-10-01)

Executed exactly one approved eight-step run without profiler, with DDO
`.venv-vmm`, unchanged DLL SHA256
`32531DBE284E8FC62F522130104AD2AD06EE7AD949D8601232E3DA3E03FFD917`,
and the same event-timing/phase-collection option validated by the trace.
Real prompt/BF16/1280x704/seed43/no-PAG, GPU budget 4 GiB, source ceilings
24.625 GiB, registered staging 1.5 GiB. Exit 0; no repeat or parameter change.

Completed CUDA event timings, eight steps:
**7.500963 s first**, **1.796179 s/it steps 2–8** (range
1.773498–1.815399 s), **20.074572 s phase**, **0.000359 s tail**.
Host denoise call **20.0747 s** agrees with the complete event interval.
The earlier profiled 1.406142 s/it does not carry over to this run; its
cause is not established. Do not call that profiled number the benchmark
or extrapolate the measured overlap fraction to this untraced run.

| Measurement | Previous unprofiled stream-ordering run | Event-timed unprofiled run |
| --- | ---: | ---: |
| Transformer load / DDO setup (s) | 0.3807 / 0.9184 | 0.3535 / 0.9595 |
| Host denoise call (s) | 20.1530 | 20.0747 |
| Generate phase with readback/cleanup (s) | 28.1420 | 28.2962 |
| Text-encoding phase (s) | 19.2101 | 19.7489 |
| VAE decode phase (s) | 1.7165 | 1.7633 |
| Total, full pipeline (s) | 49.1456 | 49.8867 |
| Sampled GPU peak, baseline-subtracted (GiB) | 6.7392 | 6.6675 |
| Sampled process RSS peak (GiB) | 29.5668 | 29.5667 |

Denoise host scope is 0.0783 s faster (~0.39%), but Generate is 0.1542 s
slower and total 0.7411 s slower (~1.51%). Text phase contributes 0.5388 s
of that increase. One run neither establishes significance nor identifies
the event-timing option as its cause. Under the non-regression criterion,
do not promote defaults or claim end-to-end improvement; retain both
measured candidates and all previous DLLs/artifacts. No automatic repetition.
The first complete unprofiled GPU steady-state observation is 1.796179 s/it;
older callback-derived means are not identical measurement definitions.
The approximately 1.3 s/it target remains unmet; cold first-step source work
and setup are still material, rather than a solved preparation target.

PNG SHA256 unchanged:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Phase collection **1312 -> 0 tickets**, source retention/registration intact.
Same 187,588,136,576 H2D bytes, 25,846,148,360 source-read bytes, 388
registrations / zero unregistrations, 24.25 GiB committed/registered source,
3.59375 GiB native weight backing, 376/376 prefetch hits. All fault/admission/
read/order/registration-wait/pressure/fallback counters zero; monitors without
errors. Cumulative events_created=7558/events_reused=3610; batch-submit host
scope 7.339612 s, source-ready host scope 6.705243 s (nested/non-additive).
These scopes do not independently establish a warm-run event bottleneck.
Standby purge privilege remains unavailable; no driver/OS policy changed.
Full-model post-close ledger snapshot and protected runtime full benchmark
validation remain separate from successful phase retirement/small contracts.

Artifacts: sibling
`outputs/ltx_image_modular_vmm/completion_timing_candidate_20261001/` PNG/metrics.
Next work must explain the profile/unprofiled scheduling difference using
existing evidence and address a demonstrated mechanism; do not enlarge
ceilings or tune by guessing. Any further CUDA run needs new approval.

### Approved completion/timing trace — overlap and retirement verified (2026-10-01)

Approved selected CUDA contracts: **40 passed in 6.38s**, exit 0, no skips.
Includes pending DMA protection, preservation of external ticket ownership,
completed native batch collection and asynchronous compute-stream event timing.
DDO `.venv-vmm`, Release DLL SHA256
`32531DBE284E8FC62F522130104AD2AD06EE7AD949D8601232E3DA3E03FFD917`.
One approved eight-step trace followed with `--ddo-gpu-timing`, unchanged
real prompt/BF16/1280x704/seed43/no-PAG, source ceilings 24.625 GiB,
GPU budget 4 GiB, staging 1.5 GiB. Exit 0; no automatic repeat.

Trace contains **527,095 events / 160,829,654 bytes**, including 36,751 GPU
kernels and 12,568 GPU memcpy activities. Native transformer H2D on Kineto
stream 17: 11,168 transfers / **187,588,136,576 bytes** / 7.227756 s summed
GPU duration. Kernels on Kineto stream 7. CUDA timing reports stream handle
0 (default stream); this is a handle, not Kineto's stream-ID numbering.

Merged interval intersection, using the same method as the earlier trace:

| Window | H2D union (s) | Kernel union (s) | Overlap (s) | Neither measured class (s) |
| --- | ---: | ---: | ---: | ---: |
| First-to-last transformer upload, 17.237181 s | 7.227756 | 7.406435 | **2.733414** | 5.336404 |
| Final 10 s ending at last transformer upload | 6.127331 | 6.416869 | **2.733414** | 0.189213 |

Previous trace measured zero overlap. This trace demonstrates actual H2D/
compute overlap after stream-ordered temporary reuse, not just queued APIs.
The fixed 10-second windows cover different work quantities; they are not
equal-step performance comparisons. "Neither" excludes only these measured
classes, not every GPU engine. Full upload window includes cold host work and
omits kernels after its final copy, so it is not the complete sampler timer.

DDO CUDA event report: **complete**, eight steps, first **7.499001 s**,
steps 2–8 mean **1.406142 s/it** (range 1.378479–1.452468 s), phase
**17.342396 s**, tail **0.000404 s**. Host denoise call 17.3431 s. These
intervals include compute-stream waits/host launch gaps; not kernel-only
duration. The former 0.3-second host callback step is not a completed GPU step.
All timings are from a profiled diagnostic and do NOT replace unprofiled
milestones or establish the final 1.3 s/it target. No unprofiled run followed.

After existing blocking latent readback, native collection retired **1216
tickets to zero**, without dropping source bytes/registration. This confirms
the end-of-phase ticket snapshot backlog was collectible completed work in
this run, not persistent live batch ownership. It is not a full-model
post-close ledger snapshot or proof against every possible lifetime failure.
Events_created=6772/events_reused=4396 are cumulative; do not present them as
live resource counts or a reduction caused by end-of-phase collection.

PNG SHA256 matches protected output:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Same 25,846,148,360 source-read bytes, 388 registrations / zero unregistrations,
24.25 GiB committed/registered source and 3.59375 GiB native weight backing.
All fault/admission/read/order/registration-wait/pressure/fallback counters
zero; 376/376 prefetch hits. GPU peak baseline-subtracted **6.6859 GiB**,
below previous unprofiled 6.7392 but above pre-stream-ordering 6.4832; RSS
**29.7626 GiB** (profiler present). All memory monitors report no error.
DDO setup 0.9489 s; profiled Generate phase 25.5055 s / total 46.8494 s are
diagnostic only. Standby purge privilege remains unavailable/unchanged.

Artifacts in sibling host:
`outputs/ltx_image_modular_vmm/completion_timing_trace_20261001/cuda_trace.json`
and `completion_timing_trace_20261001/run/` PNG/metrics. Prior artifacts and
DLLs retained. Correctness/timing/overlap gates passed; performance/default
promotion and protected linear/NVMe full benchmarks remain separate. Next
proposed validation is one unprofiled eight-step run with the same event
timing/collection option, not another fragmented parameter exploration.

### Completion collection and compute-stream timing — prepared, CUDA pending (2026-10-01)

Inspection confirms batch submission reaps ready tickets across all batches;
the final ordinary snapshot itself does not reap. Each ticket keeps its DMA
event/registration protection until completion is proven. Idle event pool is
capped at 256, but pending events are not capped by that pool; events_created
is cumulative, not a simultaneous live count. This explains why the earlier
snapshot cannot independently establish a leak or peak event resources.

New optional native phase collection uses the existing nonblocking reaper;
DDO reports tickets before/after/retired via its public metrics object. External
owners and pending DMA remain protected. No residency/registration/host limit,
pool size, normal scheduling or protected runtime behavior was changed.
New opt-in DDO CUDA timer records compute-stream start/step/end markers;
no per-step waits, pending durations omitted, phase tail accounted separately.
Sibling harness consumes these public operations behind `--ddo-gpu-timing`,
off by default. Marker intervals include dependencies/launch gaps, not just
kernels. Collection after existing latent readback is not proof of post-close
full-model ledgers or an optimization of in-flight event creation.

Release build and CPU C++ contracts passed. CPU suite **94 passed in 5.10s**;
syntax checks passed. DLL SHA256
`32531DBE284E8FC62F522130104AD2AD06EE7AD949D8601232E3DA3E03FFD917`;
prior stream-ordering DLL saved as
`build/Release/ddo_vmm.pre-completion-collection-20261001.dll`.
Prepared small CUDA ownership/query/retirement and event-timing contracts,
followed by one proposed eight-step diagnostic trace with unchanged budgets.
Commands/hypothesis/approval criteria in `docs/native-source-cache-plan.md`.
No new CUDA/trace/performance run yet; measured milestones below unchanged.

### Stream-ordering sequence executed — correctness passed, promotion held (2026-10-01)

Approved sequence ran with DDO `.venv-vmm` and Release DLL SHA256
`FBABF22A4E3581E410E653B437F477FD728AA441188749624C8CC23E326AB3ED`.
Fresh-process allocator probe: **PASS**, 16 MiB mapping / 128 MiB cache,
stable address, full payload, nonblocking three-stream reuse, renewed
consumer fence and zero live allocations after free. Selected execution
buffer/native transport/public runtime/source-cache CUDA contracts:
**38 passed in 5.64s**, exit 0, no skips. No error-retention message occurred.
Then the approved unchanged real-prompt/BF16/1280x704/seed43/no-PAG/eight-step
candidate ran once, exit 0; no additional trace or parameter change.

| Measurement | Previous registered-source candidate | Stream-ordered candidate |
| --- | ---: | ---: |
| Transformer load / DDO setup (s) | 0.2849 / 0.9582 | 0.3807 / 0.9184 |
| Host-scoped denoise call (s) | 21.6363 | 20.1530 |
| Generate phase, includes latent readback/cleanup (s) | 29.1332 | 28.1420 |
| Total, including text encoder/VAE (s) | 50.0931 | 49.1456 |
| Sampled GPU peak, baseline-subtracted (GiB) | 6.4832 | 6.7392 |
| Sampled process RSS peak (GiB) | 29.4990 | 29.5668 |
| Snapshot live source tickets | 32 | 1312 |
| Source events created / reused | 32 / 11136 | 7616 / 3552 |
| Native source batch-submit scope (s) | 6.893076 | 7.259783 |

PNG SHA256 unchanged:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Same 187,588,136,576 H2D bytes, 25,846,148,360 source-read bytes, 388
registrations / zero unregistrations, 24.25 GiB committed/registered source,
1.5 GiB registered staging and 3.59375 GiB native weight backing. GPU
budget stays 4 GiB and host ceilings 24.625 GiB. All admission/fault/source
fallback/pressure counters remain zero; 376/376 prefetch hits. Memory
monitors have no errors; standby purge remains skipped for missing privilege.

Observed host-scoped denoise reduction 6.86%; total reduction 1.89% in one
run. Do not equate these timers with separately verified GPU completion.
The callback records perf_counter before any completion fence, and ordinary
profile_run is disabled/no-op. New step 2 is 0.188734 s, steps 3–8 range
1.796110–1.833908 s; the nominal steps 2–8 mean is NOT a valid comparable
steady-state result after asynchronous reuse. Do not advertise it as reaching
the target or insert per-step synchronizations that defeat overlap. Generate
phase includes the blocking latent CPU readback and is a stronger completion
boundary than callback timestamps. GPU overlap itself remains unmeasured.

The increased GPU peak (+0.2560 GiB) and pending ticket/event counts are
observational regressions requiring investigation. Snapshot is before close;
1312 tickets does not prove a leak, nor does successful process exit prove
post-close ledgers were zero for this full model. Small CUDA tests do verify
their own cleanup. No automatic repeat, budget increase or promotion: keep
the previous measured milestone and its saved DLL protected. Next work must
verify native completion/retirement and GPU-event timing without adding a
Python residency policy; further CUDA/trace needs separate confirmation.

Artifacts: sibling host
`outputs/ltx_image_modular_vmm/stream_reuse_candidate_20261001/` PNG and metrics.
The patch-preparation status below is historical, superseded by this section.

### Temporary allocator stream-ordering patch — compiled, CUDA pending (2026-10-01)

Native malloc now enqueues its cached record's last-use event on the consumer
stream instead of synchronizing the host. It retains the old fence until
free successfully records the next consumer fence, including failed allocation
publication. Eviction/close still require completion before unmap. Failed
completion proof retains backing until process teardown and emits an error;
such records are never cached or reused. This is exceptional safety retention,
not a performance path or permission to continue after a CUDA error.

Release build passed. Pure CPU C++ dependency/failure/drain contracts passed;
Python launch/config/probe option suite: **50 passed in 5.61s**.
New DLL SHA256 `FBABF22A4E3581E410E653B437F477FD728AA441188749624C8CC23E326AB3ED`.
Previous DLL preserved at `build/Release/ddo_vmm.pre-stream-reuse-20261001.dll`,
SHA256 `F189D035D10089AE77DD9A7F21EED2FF82B103D2714118DE573D227B346D8258`.
No post-patch CUDA, performance, overlap or linear/NVMe regression gate has run.
Budgets/defaults/source transport/runner scheduler unchanged.

Prepared fresh-process CUDA probe checks nonblocking reuse through three
streams, refreshed last-consumer fencing, identical pointers and full payload.
Proposed consolidated sequence: small probe, existing CUDA contracts, then
one unchanged eight-step candidate without profiler. Commands/hypothesis/
approval criteria in `docs/native-source-cache-plan.md`; user confirmation
required. The 21.6363 s / 1.922283 s/it candidate below remains the measured
milestone. Compilation and mocked contracts do not constitute hardware proof.

### Approved CUDA timeline — host cache-reuse barrier identified (2026-10-01)

Generic `scripts/trace_vmm_runner.py` ran the approved eight-step candidate,
same native DLL, parameters, registration ceiling and RSS monitor. Exit 0.
Trace contains 526,379 events / 160,181,632 bytes, including 36,751 real GPU
kernels and 12,568 GPU memcpy activities. Native transformer H2D on stream
17 has 11,168 transfers totaling exactly **187,588,136,576 bytes**, matching
the public runtime report; GPU kernels run on stream 7. This is not merely
a CPU API trace. Full artifact is local to the sibling host, not this repo.

GPU interval analysis uses merged timestamp/duration intervals (microseconds),
not summed nested host scopes. Bounds first-to-last transformer upload:
22.826491 s, H2D union 7.284463 s, kernel union 7.197513 s, overlap **0 s**,
neither class active 8.344515 s. These bounds include cold filling and omit
kernels after the last upload; they are not the full sampler timer.
Late 10 s ending at the last upload excludes cold filling: H2D 4.387028 s,
kernels 4.314711 s, overlap **0 s**, neither class active 1.298262 s.
"Neither" is idle relative to these two measured classes, not proof that
every GPU engine is idle. Late captured CUDA runtime/driver Synchronize calls
total only 0.000166 s, but profiler does not list native cuEventSynchronize:
absence of that API activity is not absence of the host wait.

Code inspection found a separate normal-path barrier in
`native/src/ddo_vmm_torch_allocator.cpp`: malloc ignores its stream argument;
every cached record runs wait_for_stream_completion, which calls
cuEventSynchronize before returning the allocation. This can stop host launch
of the current unit before the post-hook can enqueue next-unit H2D. The
weight-pool fences already use stream waits; their asynchronous API is not
sufficient if ordinary activation/temporary allocations block the host.
This is a concrete candidate mechanism consistent with the zero-overlap
timeline, not proof of the milliseconds attributable to each allocation.

Same PNG hash as both A/B cases and protected milestone; 388 registrations,
zero unregistrations, same H2D/3.59375 GiB weight backing, no fault/temporary/
pressure/host-copy fallback and no monitor errors. Profiled denoise 22.9409 s,
setup 0.9381 s and total 52.5954 s are diagnostic only. They do not replace
the unprofiled candidate's 21.6363 s / 1.922283 s/it or promote defaults.

Next correction: use malloc's consumer stream for an ordered reuse dependency,
keeping the last-use event and physical mapping alive safely; preserve a safe
failure path and event-protected eviction/close. Do not simply delete waits,
increase cache/residency budgets, or modify linear/NVMe. Prepare ownership,
cross-stream reuse, cancellation/failure and memory-ceiling contracts before
an explicitly approved CUDA/benchmark/trace sequence. No native change or
additional run occurred in this diagnostic turn.

Artifacts: sibling host
`outputs/ltx_image_modular_vmm/registration_trace_20261001/cuda_trace.json`
and `registration_trace_20261001/run/` PNG/metrics. Existing artifacts preserved.

### Full persistent registration — approved sequence executed (2026-10-01)

This execution supersedes the preparation status immediately below. DDO
`.venv-vmm`, unchanged Release DLL SHA256
`F189D035D10089AE77DD9A7F21EED2FF82B103D2714118DE573D227B346D8258`.
Consolidated CUDA contracts: **38 passed in 5.80s**, exit 0, no skips.
Full-source registration capacity: **1 passed in 14.36s**, exit 0.
All 394 segments retained/registered at 24.625 GiB, staging 1.5 GiB;
second pass had 394 hits, unchanged 26,248,801,544 source bytes read,
394 registrations total, zero unregistrations/waits/pressure/bypass/fallback
before trim. Sample DMA data matched; trim/clear/close refunded ledgers.
This capacity gate sampled 17 bytes per segment, not full-model throughput.

Approved eight-step A/B followed, control then candidate, fresh processes,
BF16/real prompt/1280x704/seed43/no PAG, same local model and RSS monitor.
GPU ceiling stayed 4 GiB, source storage 24.625 GiB and staging 1.5 GiB.
Selected model/sampling/configuration manifest fields match; only requested
source registration differs, apart from observational RAM/timing fields.
No automated repeat or additional budget change was performed.

| Measurement | Control: 0.5 GiB registration | Candidate: 24.625 GiB ceiling |
| --- | ---: | ---: |
| Transformer load / DDO setup (s) | 0.3207 / 0.8961 | 0.2849 / 0.9582 |
| First denoise step (s) | 7.997031 | 8.166015 |
| Steps 2–8 mean (s/it) | 2.997579 | 1.922283 |
| Denoise (s) | 28.9837 | 21.6363 |
| Total (s) | 56.9413 | 50.0931 |
| Sampled GPU peak, baseline-subtracted (GiB) | 6.6353 | 6.4832 |
| Sampled process RAM peak, RSS (GiB) | 29.5001 | 29.4990 |
| Source registrations / unregistrations | 2805 / 2797 | 388 / 0 |
| Register / unregister scopes (s) | 1.295382 / 13.685070 | 0.424389 / 0 |
| Native batch-submit scope (s) | 19.774054 | 6.893076 |

Identical PNG SHA256 in both cases and protected milestone:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Same 187,588,136,576 H2D bytes (174.7051 GiB), 25,846,148,360 source-read
bytes, 388 demanded windows, 840 signature hits, 376/376 look-ahead hits,
3.59375 GiB total native weight backing. Candidate registers the demanded
24.25 GiB progressively; unused capacity is not eagerly registered.
No source/fault/admission failures, arena evictions, temporary tensors,
registration waits, pressure evictions or host/ring fallback in either case.
Both RSS/GPU monitors reported no errors. Snapshot has 32 pooled live source
tickets before cleanup in both cases; not a post-close leak measurement.
Separate gates verify closure. Standby purge remained skipped due to missing
process privilege. No process/driver policy was changed.

Observed denoise reduction **25.35%**, steady-state reduction **35.87%**.
Same transfer volume/backing and zero candidate unregistration support
registration churn as a material bottleneck. Nested scopes are not additive
and this single A/B does not establish statistical repeatability or exact
causal attribution of every saved millisecond. Setup increased 0.0621 s
and first step 0.1690 s; do not claim no regression across every scope.
Candidate is retained for follow-up, not automatically promoted as final
milestone/default. Setup target, ~1.3 s/it goal, demonstrated overlap,
linear/NVMe performance gates and text-encoder expansion remain pending.

Artifacts: sibling host
`outputs/ltx_image_modular_vmm/registration_control_20261001/` and
`outputs/ltx_image_modular_vmm/registration_candidate_20261001/`, including
their `metrics/` directories. Exact approved commands/limits are in
`docs/native-source-cache-plan.md`; no older artifacts were overwritten.

### Full persistent registration — prepared, not executed (2026-10-01)

Prepared native-cache contracts for full-payload repeated DMA without
registration churn, plus an independently opt-in existing-shard capacity
gate with an exact approved registration ceiling. Existing recycling,
fallback and runtime contracts remain. No allocator/DLL/default/budget change.
CPU configuration/backend/profile/preset gate: **75 passed in 7.64s**;
hardware-test syntax validated. CUDA, large registration and A/B pending
explicit approval. No new performance claim or promotion.
Commands, limits and stop criteria: `docs/native-source-cache-plan.md`,
section dated 2026-10-01. Proposed comparison changes only source registration
from 0.5 to 24.625 GiB; GPU ceiling remains 4 GiB. Protected milestone below
remains setup 0.8870 s / denoise 29.7999 s / steps 2–8 mean 3.002195 s.

### Fast setup accounting — functional gates passed, performance not accepted

Inspection found previously untimed copies of auxiliary dense linears outside
the arena. The approximately 0.364 s remaining outside existing setup scopes
must not be labelled planner overhead. Fast VMM now reports distinct metadata,
native-resource preparation and auxiliary-linear copy scopes, preserving the
same copy methods/order/placement, without new streams, buffers or deferred
first-step work. No linear/NVMe, native DLL or capacity changes.

CPU consolidated contracts: **119 passed in 5.29s**, including mocked-copy
sequence equivalence, already-placed modules, transfer-byte accounting and
fast-only metrics. A CPU/meta fixture initially failed because PyTorch rejects
cross-type Parameter.data assignment; the test simulation was corrected, not
the transfer implementation. Approved CUDA gate: **37 passed in 6.66s**, exit 0.
The subsequent approved real-prompt eight-step run also exited 0, using DDO
`.venv-vmm`, unchanged parameters and native DLL. Its performance was worse;
it is **not promoted** over the protected setup 0.8870 s / denoise 29.7999 s
milestone. The instrumentation remains applied; this is not a rollback.

| Measurement | Protected coalesced-view milestone | Setup accounting run |
| --- | ---: | ---: |
| Transformer load / DDO setup | 0.4191 / 0.8870 s | 0.5237 / 1.2644 s |
| Arena attach | 0.3931 s | 0.6487 s |
| Denoise / steps 2–8 mean | 29.7999 / 3.002195 s | 36.3482 / 3.629138 s |
| Prompt phase / generation phase | 19.5278 / 38.3389 s | 22.3117 / 46.6039 s |
| Total / total minus prompt phase | 59.6115 / 40.0837 s | 70.9472 / 48.6355 s |
| Sampled GPU peak, baseline-subtracted | 6.9727 GiB | 6.9727 GiB |
| Sampled RAM peak, RSS | 29.6242 GiB | 29.5067 GiB |

New disjoint setup scopes: metadata **0.0231 s**, native resources **0.0071 s**,
auxiliary-linear movement **0.3882 s / 0.4122 GiB**; existing small-parameter
movement **0.1768 s / 0.0055 GiB**, root movement **0.0003 s**, arena attach
**0.6487 s / 24.0282 GiB virtual**. Sum 1.2442 s leaves about 0.0202 s
of wrappers/scans/rounding outside these scopes. Auxiliary copies are now a
measured setup cost; metadata was not the dominant cost in this run. Do not
attribute the earlier run's residual using timings from this different run.

Image SHA256 is unchanged:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Backing remains 3.59375 GiB under the 4 GiB ceiling; H2D 187,588,136,576
bytes, source reads 25,846,148,360 bytes, hits/misses 10780/388, signatures
840 and look-ahead 376/376 remain unchanged. No evictions, fault/admission
failures, temporary fallback or host-copy fallback. Native batch submission
25.625996 s versus bridge 25.633184 s leaves only 0.007188 s of bridge residual.

Unmodified encoder and connector phases were also slower; sampled encoder
GPU peak rose from 4.57 to 5.69 GiB. Monitor reported no error, and a point
GPU check showed free memory, but neither proves the cause of the slowdown.
GPU peak excludes the approximately 1.03 GiB launch baseline; no budget
increase is justified. Functional correctness passed; performance acceptance
remains open. Linear/NVMe performance gates were not included. No automatic
repeat, parameter tuning or environment/process changes are authorized by
this result. Total-minus-prompt is not the reference sampler timer.

Artifacts: sibling host `outputs/ltx_image_modular_vmm/native_setup_accounting_8steps/`
and `metrics/`. The approved CUDA command used the same four test files as
the coalesced-view gate below, with `--basetemp .pytest-fast-setup-accounting-cuda`;
the unchanged eight-step invocation used only the new output directory.

### Registration-cost decomposition — existing reports, no new execution

The protected and accounting runs respectively report registration 0.956338 /
2.393552 s, unregistration 14.196268 / 16.401211 s, and explicit registration
waits **0 / 0**. Registration scheduling totals 15.198933 / 18.870604 s;
the residual after register/unregister is about 0.046328 / 0.075841 s.
These scopes are nested, not additive to batch submission or denoise.

Source-cache `ensure_registration` recycles ready entries without active DMA
when the 0.5 GiB registration ceiling is full. It unregisters the old native
storage and registers the requested storage before enqueuing DMA. The storage
bytes remain committed, so hits avoid disk rereads but do not eliminate this
registration churn: 2805 registrations and 2797 unregistrations in both runs.
`unregister_storage` includes the ring mutex and `cuMemHostUnregister`; the
existing timer does not isolate driver time from mutex acquisition. Zero
explicit registration waits does not prove CUDA driver calls never block.

The measured dominant scheduling cost is therefore unregistration, not the
explicit event-wait branch or Python planner. This identifies a target, not a
validated replacement: reducing churn must retain DMA ownership and the same
registration ceiling. Neither registering the full host cache nor releasing
live transfers is an acceptable shortcut. No allocator change or new run was
made for this decomposition; the cause of run-to-run slowdown is still open.

### Coalesced setup-view patch — approved CUDA and eight-step validation

Temporary view mappings now span one contiguous structural execution unit,
reusing the existing native single-page bootstrap allocation instead of
mapping/unmapping shared pages per tensor. Release precedes the next unit and
all inference; errors release via scope cleanup, and a release failure prevents
successful attach. Refused spans preserve individual-tensor prepare/fallback.
No native DLL, memory ceiling, source/DMA/event or linear/NVMe code change.

Final consolidated CPU gate: **113 passed in 5.03s**. CUDA contracts prepared
for original/renamed containers and one/two tensors per unit, including setup
operation counts, numerical parity, cancellation and fixed capacity.
Subsequent approved gate: **37 passed in 5.43s**, exit 0. Only after success,
one approved real-prompt eight-step run completed with exit 0, unchanged
parameters, DDO `.venv-vmm`, local cache and the same ABI v11 DLL/hash.

| Measurement | Structural milestone | Coalesced views |
| --- | ---: | ---: |
| Transformer load / DDO setup | 0.4458 / 1.1137 s | 0.4191 / 0.8870 s |
| Native arena attach | 0.5874 s | 0.3931 s |
| Small parameter movement | 0.1471 s | 0.1299 s |
| Denoise | 30.8719 s | 29.7999 s |
| Steps 1–8 | 8.9167 / 3.1128 / 3.0802 / 3.1599 / 3.2420 / 3.1553 / 3.1175 / 3.0875 s | 8.7851 / 3.0198 / 3.0366 / 2.9950 / 2.9886 / 2.9854 / 2.9962 / 2.9938 s |
| Steps 2–8 mean | 3.136468 s | 3.002195 s |
| Prompt phase / generation phase | 20.1047 / 39.4750 s | 19.5278 / 38.3389 s |
| Total / total minus prompt phase | 61.4271 / 41.3224 s | 59.6115 / 40.0837 s |
| Sampled GPU peak, baseline-subtracted | 6.9727 GiB | 6.9727 GiB |
| Sampled RAM peak, RSS | 29.5917 GiB | 29.6242 GiB |
| View prepare / create / release attempts | Not collected | 48 / 1152 / 48 |
| View prepare / create / release host scopes | Not collected | 0.017449 / 0.099560 / 0.053906 s |

Image SHA256 matches the prior real-prompt milestones:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Geometry and numerical output are preserved: 1152 views now use 48 unit
prepare/release pairs, rather than the previous per-tensor mechanism (previous
call counts were not instrumented). Aggregate weight backing remains 3.59375
GiB under 4 GiB. H2D remains 187,588,136,576 bytes, source reads
25,846,148,360 bytes, hits/misses 10780/388, registrations/unregistrations
2805/2797, signature hits 840 and look-ahead enqueued/consumed 376/376.
No evictions, fault/admission failures, temporary tensors or host-copy fallback.
GPU observation remains enabled; generation recorded 348 samples, RAM-query
cost 0.032134 s, GPU-query cost 0.086505 s, no monitor error. FFI/native batch
submit residual 0.005107 s; the USS-related delay did not return.

Observed setup decreased 20.36%, attach 33.08%; the single-run steady mean is
inside the initial 2.80–3.13 s target band. This is not repeated statistical
proof, nor an isolated inference optimization: denoise code did not change.
The 0.16 s setup and 1.3 s/it aspirations remain unmet. Total-minus-prompt is
not the reference sampler timer. GPU peak still excludes the approximately
1.03 GiB launch baseline; do not raise ceilings. Linear/NVMe performance
regression benchmarks and full UNet/text-encoder validation were not run.
Same unused-vision warning and unavailable standby purge as earlier cases.

Artifacts: sibling host `outputs/ltx_image_modular_vmm/native_coalesced_views_8steps/`
and its `metrics/` directory. Approved gate used the existing launch-local
SSL workaround and DDO/native environment:

```powershell
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-coalesced-view-setup-cuda
```

Approved host invocation is the unchanged real-prompt eight-step command below
with output directory `outputs/ltx_image_modular_vmm/native_coalesced_views_8steps`.
No additional run, tuning, DLL or compatibility change.

### Structural setup patch — approved CUDA and real-prompt validation

The experimental native adapter now shares a single selected-module/layout
snapshot for sizing and attachment. ModuleList items define execution units
without architecture names; nested lists use the enclosing item, and standalone
linears retain individual units. Fixed address/page/virtual geometry is cached;
live residency/signatures/pressure are not. No new convolution support, default
filter, CUDA DLL, budget, DMA/event policy or linear/NVMe route change.

CPU layout/ownership/configuration: **58 passed in 5.40s**. CPU presets,
named profiles and backend routing: **45 passed in 4.81s**. These are not CUDA
or performance regression gates. Subsequent approved CUDA sequence: **35 passed
in 5.89s**, exit 0, including original and renamed list containers. Only after
success, the approved real-prompt eight-step case ran once with exit 0, same
DLL hash and all parameters of the previous milestone. Linear/NVMe performance
regression benchmarks were not included; their code/routes remain unchanged.

| Measurement | Pre-patch milestone | Structural setup |
| --- | ---: | ---: |
| Transformer load / DDO setup | 0.3720 / 1.3067 s | 0.4458 / 1.1137 s |
| Native arena attach scope | 0.7862 s | 0.5874 s |
| Small parameter movement | 0.1532 s | 0.1471 s |
| Denoise | 31.4093 s | 30.8719 s |
| Steps 2–8 mean | 3.150964 s | 3.136468 s |
| Steps 1–8 | 9.3527 / 3.1685 / 3.1116 / 3.1062 / 3.1106 / 3.1168 / 3.1558 / 3.2872 s | 8.9167 / 3.1128 / 3.0802 / 3.1599 / 3.2420 / 3.1553 / 3.1175 / 3.0875 s |
| Complete prompt / generation phase | 19.7553 / 40.0205 s | 20.1047 / 39.4750 s |
| Total / total minus prompt phase | 61.7671 / 42.0118 s | 61.4271 / 41.3224 s |
| GPU sampled peak, baseline-subtracted | 6.9727 GiB | 6.9727 GiB |
| RAM sampled peak, RSS | 29.5723 GiB | 29.5917 GiB |

PNG SHA256 is identical to the real-prompt milestone:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Weight backing remains 3.59375 GiB within 4 GiB; source reads
25,846,148,360 bytes, H2D 187,588,136,576 bytes, 376 look-ahead hits, 840
signature hits, 10780 source hits/388 misses and 2805 registrations/2797
unregistrations are unchanged. Zero evictions, admission/fault failures,
temporary tensors or host-copy fallback. Monitor errors: none. Generation
recorded 366 samples, RAM query cost 0.044864 s and GPU query cost 0.115691 s.
FFI/native submit residual remains only 0.005588 s (404 calls).

The single comparison preserves output and measured capacity while setup is
14.77% lower and native attach scope 25.29% lower. Small denoise differences
are inconclusive as an isolated optimization benefit; no repeated estimates.
The 0.16 s setup target remains unmet. GPU peak excludes the approximately
1.03 GiB launch baseline: do not infer spare capacity or raise ceilings.
The same unused-vision-branch warning and unavailable standby purge occurred.
Artifacts: sibling host `outputs/ltx_image_modular_vmm/native_structural_setup_8steps/`
and its `metrics/` directory. No additional run, tuning or runtime change.

Approved gate, from DDO (same launch-local SSL workaround and environment as
the benchmark commands below):

```powershell
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-structural-layout-cuda
```

Approved host command: the real-prompt eight-step command recorded below,
unchanged except output directory `outputs/ltx_image_modular_vmm/native_structural_setup_8steps`.

### Two-slot native pipeline — selected CUDA gate passed — 2026-09-30

The persistent-source route now uses two reusable native execution slots,
native immutable source batches and pooled ticket events. The C++ pool owns
rotation, backing and event-protected reuse; its entire capacity is carved
from the existing GPU ceiling. Python caches views and submits one execution
unit instead of request/copy/reap calls per file window. Hooks still orchestrate
units. No production capacity, linear/NVMe route or compatibility was changed.

Source ABI v11 / stats v8 adds complete host request/copy/close scopes, event
acquisition/retirement, registration scheduling and batch submit/reap timing.
These scopes nest and are not GPU DMA durations. Old snapshot prefixes are
preserved. Partial returns and host exceptions record completion; an unproven
event retains ownership rather than permitting unsafe recycling/closure.

- Release compiled successfully using the existing CMake configuration.
- DLL SHA256: `F189D035D10089AE77DD9A7F21EED2FF82B103D2714118DE573D227B346D8258`.
- CPU ownership, rotation, cached-view/batch dispatch, failed-close retention,
  existing look-ahead and source configuration: **42 passed in 4.50s**, exit 0.
- Changed Python syntax checks and root/submodule whitespace checks passed.
- Approved selected CUDA gate: **34 passed in 5.32s**, exit 0, same DLL.
- Subsequent approved four-step LTX diagnostic completed; results below.
  The selected CUDA gate alone does not establish throughput.

CPU command, DDO `.venv-vmm`:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\ddo-vmm-allocator\src"
& .\.venv-vmm\Scripts\python.exe -m pytest ddo-vmm-allocator/tests/test_execution_buffer.py ddo-vmm-allocator/tests/test_gpu_prefetch.py tests/test_vmm_source_cache_config.py -q -x -p no:cacheprovider --basetemp .pytest-native-pipeline-cpu-complete
```

Passed consolidated CUDA gate: public two-slot numerical parity over three
passes including a skipped unit/cancellation; aggregate residency within a
6 MiB **small-test-only** ceiling (two driver-rounded slots plus an admitted
page), zero tail evictions/temporary fallback, source reuse, native batch
ownership/event reuse and stats v7 prefix compatibility. Existing selected
public fast-runtime/transport and source-cache controls accompany it. Production
GPU ceiling remains 4 GiB. The approved sequence used `-x`; no LTX execution
or parameter extension was included. Launch omitted inherited `SSLKEYLOGFILE`
only inside the launch process, using the previously approved workaround.
This closes selected functional contracts, not throughput, actual full-model
overlap, full-model capacity or linear/NVMe performance regression.

```powershell
Remove-Item Env:\SSLKEYLOGFILE -ErrorAction SilentlyContinue
$env:PYTHONPATH="$PWD\src;$PWD\ddo-vmm-allocator\src"
$env:DDO_VMM_LIBRARY_PATH="$PWD\ddo-vmm-allocator\build\Release\ddo_vmm.dll"
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_execution_buffer.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py ddo-vmm-allocator/tests/test_source_cache.py -q -x -p no:cacheprovider --basetemp .pytest-native-pipeline-cuda
```

### Two-slot native pipeline: approved LTX diagnostic — 2026-09-30

Subsequent upload-boundary instrumentation is prepared in the Python native
adapter: transport preparation/creation/submit, ctypes create/submit scopes,
argument preparation, lock wait, result handling and existing stream-context
entry/body/exit. Native DLL hash/ABI, transfers, lifetimes and budgets remain
unchanged. **47 CPU tests passed in 4.74s**, including deterministic clock
partitions and partial-failure accounting. No CUDA/LTX run has used these new
scopes at preparation time; the later approved validation is recorded below.
The measurements in this subsection belong to the adapter before instrumentation.
The next proposed validation is a single approved sequence: selected CUDA
contracts, then the same four-step case only on success. No parameter sweep
or additional case is authorized. Metric names/scopes are in `docs/metrics.md`.

Same ABI v11 DLL hash above, DDO `.venv-vmm`, local module/model cache,
1280x704, seed 43, fake prompt, no PAG, four steps, one generation. GPU
weight ceiling 4 GiB, persistent source 24.625 GiB, source registration
512 MiB; host staging argument 0.5 GiB resolves to 1.5 GiB total native
staging. No budget increase or additional case. Exit 0; standby purge
remained unavailable because the launch token lacked its required privilege.

| Measurement | Result |
| --- | ---: |
| Transformer load / DDO setup | 0.4134 / 1.5776 s |
| Denoise | 87.9825 s |
| Steps 1–4 | 26.0203 / 26.7594 / 17.6311 / 17.5623 s |
| Steps 2–4 mean | 20.6509 s |
| Total | 101.9847 s |
| Peak VRAM / RAM | 6.3592 / 27.0508 GiB |
| Model arena / execution pool backing | 2.53125 / 1.0625 GiB |
| Aggregate weight backing / configured ceiling | 3.59375 / 4 GiB |
| Slots / capacity per slot | 2 / 0.53125 GiB |
| Arena evictions / tail / pressure | 0 / 0 / 0 |
| Fault failures / temporary tensors / host-copy fallback | 0 / 0 / 0 |
| GPU look-ahead units / consumed / capacity skips | 188 / 188 / 0 |
| Signature hits | 360 |
| Execution-buffer uploads / bytes | 172 / 92,450,313,728 |
| Total H2D bytes | 95,137,822,848 |
| Source read bytes / hits / misses / bypasses | 25,846,148,360 / 5276 / 388 / 0 |
| Source committed bytes / peak registered bytes | 26,038,239,232 / 536,870,912 |
| Registrations / unregistrations | 1425 / 1417 |
| Broad host upload scope | 66.7107 s |
| Native batch submit / reap | 13.5557 / 0.0062 s |
| Native request / copy / close total scopes | 0.0484 / 13.4964 / 0.0023 s |
| Source readiness / registration scheduling | 4.9803 / 8.4706 s |
| Registration / unregistration / capacity wait | 0.6872 / 7.7481 / 0 s |
| Ticket events created / reused | 32 / 5632 |
| Event acquisition / retirement | 0.000528 / 0.000121 s |
| Native H2D submit / event record / ticket-close wait | 0.0373 / 0.0054 / 0.0012 s |
| Native successful batch submissions | 232 |

Output SHA256 matches all preceding controls:
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.
Artifacts are in sibling host `outputs/ltx_image_modular_vmm/native_two_slot_pipeline/`
and its `metrics/` directory. Runtime snapshots precede cleanup; 32 live
source tickets at snapshot are not a post-close leak measurement.

Denoise decreased 8.03% from the preceding single-slot 95.6604 s diagnostic;
steps 2–4 decreased from 23.0855 to 20.6509 s. This passes the narrowly agreed
improvement/parity/capacity criterion, not the final performance goal: denoise
is still about 2.42 times the earlier 36.3550 s cache-off control. These are
single sequential runs, not repeated statistical estimates or an isolated
measurement of overlap. The two late steps are not the agreed steady-state
metric by themselves. Larger execution-pool carving also changes useful
model-arena coverage within the same ceiling; H2D increased about 1.72%.

The new scopes account for only 13.5557 s inside native batch submission,
versus 66.7107 s in broad host upload. The 53.1550 s difference is still
unattributed: batch creation/metadata and Python/stream-boundary work are
outside the native submit timer. Do not label this residual as pure DMA,
Python overhead or WDDM traffic without targeted evidence. Event pooling
worked (5632 reuses); measured ticket event management is small and cannot
explain that difference. Do not start budget tuning or another LTX case before
closing these remaining scope boundaries and agreeing the next validation.

Approved command, launched from the sibling host directory:

```powershell
Remove-Item Env:\SSLKEYLOGFILE -ErrorAction SilentlyContinue
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --local-files-only --fake-prompt --no-pag --width 1280 --height 704 --seed 43 --steps 4 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 0.5 --metrics-level 2 --save-metrics --output-dir outputs/ltx_image_modular_vmm/native_two_slot_pipeline
```

### Upload-boundary instrumentation: approved consolidated validation — 2026-09-30

Subsequent observer correction is implemented and validated below: the sibling
host offers `--memory-monitor rss` to use working-set RSS instead of repeated
USS scans. Existing default `full`, GPU queries and 0.1 s poll delay remain;
RAM kind is explicit in console/JSON and query costs/counts are recorded.
**62 consolidated CPU tests passed in 4.66s**; syntax checks passed. No native
DLL/runtime/budget change was made in this correction round.
The results below are the earlier full/USS observer control. RSS peaks must
not be silently compared/relabelled as USS.

Selected CUDA gate: **34 passed in 6.07s**, exit 0, basetemp
`.pytest-upload-boundary-cuda`, same ABI v11 DLL hash. Only after success,
the approved four-step LTX case ran with unchanged capacities, model/seed,
fake prompt, local files, staging and no PAG. DDO `.venv-vmm` for both;
inherited `SSLKEYLOGFILE` omitted only in each launch process. Standby purge
remained unavailable. No extra run or parameter sweep was performed.

LTX exit 0; PNG SHA256 again matches
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.
Artifacts: sibling host `outputs/ltx_image_modular_vmm/native_upload_boundaries/`
and its `metrics/` directory.

| Measurement | Result |
| --- | ---: |
| Transformer load / DDO setup | 0.4365 / 1.4625 s |
| Denoise / total | 80.4517 / 94.1583 s |
| Steps 1–4 | 25.7974 / 18.2962 / 18.2360 / 18.1218 s |
| Steps 2–4 mean | 18.2180 s |
| Peak VRAM / RAM | 6.2658 / 27.1112 GiB |
| Aggregate weight backing / ceiling | 3.59375 / 4 GiB |
| Arena evictions / fault failures / temporary tensors | 0 / 0 / 0 |
| GPU look-ahead units / consumed / skips | 188 / 188 / 0 |
| Total H2D / retained source read bytes | 95,137,822,848 / 25,846,148,360 |
| Broad host upload | 67.273721 s |
| Transport dispatch total / calls | 67.266990 s / 232 |
| Transport prepare / create | 0.015415 / 0.013443 s |
| Transport submit | 67.248095 s |
| Bridge create total / FFI / calls | 0.011522 / 0.006710 s / 103 |
| Bridge submit total / FFI / calls | 67.246808 / 67.239150 s / 232 |
| Bridge lock wait / argument preparation / result | 0.000157 / 0.000804 / 0.000261 s |
| Existing stream enter / body / exit / contexts | 0.001914 / 1.427560 / 0.000694 s / 60 |
| Native batch submit / calls | 13.488301 s / 232 |
| Native copy total / source ready / registration schedule | 13.436775 / 5.408864 / 7.990266 s |
| Source register / unregister / capacity wait | 0.615133 / 7.344132 / 0 s |
| Ticket events created / reused | 32 / 5632 |

The partitions reconcile across the Python layers: dispatch and submit account
for practically all broad upload. One-time creation, descriptors, lock waiting,
status handling and stream entry/exit are small. Stream body is a subset of
transport/native submissions; do not add nested rows. FFI and native submit
counts both equal 232, but FFI wall time exceeds the C++ body scope by
**53.750849 s**. This localizes the dominant unexplained boundary to the DLL
call interval, not immutable batch preparation or stream switching. It does
not by itself prove device DMA cost, Python CPU work or a specific GIL owner.

Read-only inspection identifies a concrete candidate in host telemetry:
`PeakMemoryMonitor` polls every 0.1 s and Windows `get_ram_gb()` invokes
`psutil.Process().memory_full_info().uss`, not a cheap RSS counter. The installed
psutil wrapper calls `proc_memory_uss`; GPU polling also invokes PyTorch
`mem_get_info`. ctypes returns must re-enter Python while these background
polls run. GIL/telemetry interference is therefore a targeted hypothesis,
**not confirmed causal attribution**. Do not disable metrics silently, relabel
RSS as USS, or claim allocator speedup from altered monitoring. Any observer
comparison or monitoring change needs an explicitly approved validation with
memory definitions/cadence recorded.

Denoise is lower than the preceding 87.9825 s run, but no transfer/scheduling
optimization was made: counters, source coverage and H2D volume are unchanged.
Treat timing differences as sequential-run variability/possible observation
effects, not an instrumentation performance win. The earlier 36.3550 s control
and the final comparable target remain unmet. This sequence passed numerical,
capacity and scope-reconciliation criteria; it localized the residual without
establishing its final cause. No additional benchmark or CUDA mutation ran.

### Explicit RSS observer: approved four-step diagnostic — 2026-09-30

The user-approved observer comparison completed with exit 0, using DDO
`.venv-vmm` and the same ABI v11 DLL/hash, source/budget/staging settings,
model, seed and four-step fake-prompt case. Only RAM observation changed from
Windows USS scanning to explicit RSS. GPU observation and the 0.1 s poll
delay remain enabled. The default `full` mode has not been changed.

| Measurement | USS control | Explicit RSS |
| --- | ---: | ---: |
| Transformer load / DDO setup | 0.4365 / 1.4625 s | 0.3368 / 1.0862 s |
| Denoise | 80.4517 s | 18.8714 s |
| Steps 1–4 | 25.7974 / 18.2962 / 18.2360 / 18.1218 s | 9.0350 / 3.3166 / 3.2565 / 3.2632 s |
| Steps 2–4 mean | 18.2180 s | 3.2788 s |
| Total | 94.1583 s | 29.0629 s |
| Sampled peak VRAM | 6.2658 GiB | 6.5172 GiB |
| Peak RAM (different definitions; not comparable) | 27.1112 GiB USS | 27.2251 GiB RSS |
| Bridge submit FFI / calls | 67.239150 s / 232 | 13.615473 s / 232 |
| Native batch submit / calls | 13.488301 s / 232 | 13.612305 s / 232 |
| FFI minus native body scope | 53.750849 s | 0.003168 s |
| Broad host upload | 67.273721 s | 13.652648 s |
| Source readiness / registration schedule | 5.408864 / 7.990266 s | 5.249163 / 8.263075 s |
| Source register / unregister / capacity wait | 0.615133 / 7.344132 / 0 s | 0.643687 / 7.583179 / 0 s |

RSS denoise decreased 76.54%. Output PNG SHA256 remains
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.
Aggregate weight backing remains 3.59375 GiB within the 4 GiB ceiling;
two slots remain 0.53125 GiB each. H2D remains 95,137,822,848 bytes;
source reads remain 25,846,148,360 bytes, with 5276 hits and 388 misses.
All 188 look-ahead units were consumed without skips. There were no arena,
tail or pressure evictions, fault failures, temporary tensors or host-copy
fallbacks. Native host committed bytes remain 26,038,239,232 and peak source
registration 536,870,912 bytes. No budget tuning or native code change.

Generate observation recorded 244 completed samples, 0.026642 s RAM queries,
0.067471 s GPU queries and no monitor error. Sampled peaks are not exact
timeline maxima. Snapshots precede cleanup; 32 live source tickets there are
not a post-close leak measurement. Standby purge remained unavailable.

This controlled single-run comparison strongly supports interference from
the USS observer as the dominant prior boundary delay, not a new allocator
algorithm speedup. It does not establish the precise GIL/thread mechanism
without a trace. Native submit time and transfer volume stayed essentially
unchanged while the FFI residual disappeared. Earlier USS timings must not
be used as unbiased allocator throughput comparisons. The 3.2788 s steady
diagnostic is near, but still above, the 2.80–3.13 s target range; the final
real-prompt eight-step comparison and text-encoder expansion remain pending.
No additional CUDA test, benchmark, default-mode or compatibility change ran.

Approved command, from the sibling host directory; artifacts are under
`outputs/ltx_image_modular_vmm/native_rss_monitor/` and `metrics/`:

```powershell
Remove-Item Env:\SSLKEYLOGFILE -ErrorAction SilentlyContinue
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --local-files-only --fake-prompt --no-pag --width 1280 --height 704 --seed 43 --steps 4 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 0.5 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/native_rss_monitor
```

### Explicit RSS, real prompt, eight steps: approved benchmark — 2026-09-30

One approved run completed with exit 0 using the same ABI v11 DLL/hash and
DDO `.venv-vmm`. Local model/module cache, BF16, 1280x704, seed 43, no PAG;
the default real prompt describes a flower-crowned cat in a sunny park.
Text encoding remains on existing Diffusers leaf group offload, not native
VMM. No code, residency/source ceilings or monitoring defaults were changed.

| Measurement | Result |
| --- | ---: |
| Text-encoder call / complete prompt phase | 16.0043 / 19.7553 s |
| Transformer load / DDO setup | 0.3720 / 1.3067 s |
| Prepare latents (not transformer setup) | 0.0160 s |
| Denoise | 31.4093 s |
| Steps 1–8 | 9.3527 / 3.1685 / 3.1116 / 3.1062 / 3.1106 / 3.1168 / 3.1558 / 3.2872 s |
| Steps 2–8 mean | 3.150964 s |
| Complete generation phase / VAE phase | 40.0205 / 1.9113 s |
| Total / total minus complete prompt phase | 61.7671 / 42.0118 s |
| Sampled peak VRAM / RAM | 6.9727 GiB baseline-subtracted / 29.5723 GiB RSS |
| Aggregate weight backing / configured ceiling | 3.59375 / 4 GiB |
| Materializations / signature hits | 960 / 840 |
| GPU look-ahead enqueued / consumed / skips | 376 / 376 / 0 |
| Evictions / fault and admission failures / temporary tensors | 0 / 0 / 0 |
| Total H2D / source read bytes | 187,588,136,576 / 25,846,148,360 |
| Source hits / misses / bypasses | 10780 / 388 / 0 |
| Source committed / peak registered bytes | 26,038,239,232 / 536,870,912 |
| Source registrations / unregistrations / capacity waits | 2805 / 2797 / 0 |
| Broad host upload | 21.624998 s |
| Bridge submit FFI / native submit / calls | 21.573191 / 21.567679 s / 404 |
| FFI minus native body scope | 0.005513 s |
| Source readiness / registration schedule | 5.481160 / 15.903724 s |
| Source register / unregister | 1.116618 / 14.729850 s |

Generate monitoring: 362 samples, RAM queries 0.040005 s, GPU queries
0.135249 s, no error. GPU metric is used-minus-launch-baseline, not absolute
device occupancy: the approximately 1.03 GiB launch baseline must not be
treated as free headroom. The real-prompt case comes close to the device's
8 GiB total; no capacity increase is justified by the reported 6.97 GiB.
No CUDA failure, temporary or host-copy fallback was observed. Standby purge
remained unavailable. A Diffusers lazy-prefetch warning listed unused vision
branches of the text encoder; encoding and subsequent inference completed.
Source snapshots precede cleanup; live-ticket count is not a close/leak test.

The PNG was visually checked: a coherent flower-crowned cat in a park, at the
requested resolution. SHA256:
`877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
This establishes an output artifact, not numerical parity with a real-prompt
control (none ran in this sequence). Different prompt/step count means the
four-step fake-prompt PNG hash is not expected to match.

The observer correction holds for the full case: the previous multi-second
FFI residual did not return. The 3.150964 s mean is about 0.67% above the
upper 3.13 s steady target; transformer setup is still 1.3067 s rather than
the aspirational 0.16 s. Neither the 1.3 s/it aspiration nor full roadmap is
closed. Total minus prompt phase is 42.0118 s, but that includes connectors,
setup, decode and save: it must not be called the reference's sampler scope
or used to assert a win over its 49.66 s without matching timing boundaries.
Single-run results are not repeated statistical estimates. Next work must
preserve this real-prompt milestone; text-encoder native expansion and the
remaining regression/delivery gates still require separate validation.

Approved command, executed from the sibling host directory:

```powershell
Remove-Item Env:\SSLKEYLOGFILE -ErrorAction SilentlyContinue
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build\Release\ddo_vmm.dll'
& E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --local-files-only --no-pag --width 1280 --height 704 --seed 43 --steps 8 --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 0.5 --metrics-level 2 --save-metrics --memory-monitor rss --output-dir outputs/ltx_image_modular_vmm/native_rss_real_prompt_8steps
```

### Persistent-storage correction: selected functional gate passed — 2026-09-30

Native config v5 / source-cache ABI v8 now separates committed host contents
from CUDA registration. Unregister/re-register preserves bytes without a new
file read. Registration recycling waits only for protected transfer events;
actual host pressure and explicit cleanup retain separate storage lifetimes.
Registration refusal uses retained bytes through the existing native staging
ring, including chunking for requests larger than one staging slot. Host copy,
registration churn, retained commit and fallback bytes are reported separately.

Public settings expose `vmm_source_registered_gb` alongside retained-storage
capacity; the sibling host passes `--ddo-vmm-source-registered-gb`. Zero uses
the same ceiling as retained storage. No host/GPU budget, benchmark parameters
or compatibility route was increased/removed. A partial buffer still does not
cover the full model and this change is not evidence of improved denoise time.

- Release DLL compiled successfully with the existing DDO CMake configuration.
- DLL SHA256: `F345CA76B8AB5BDD052AA38DADE264DCF1D835C569A73B7E62C5AF80C48BCCC8`.
- Changed Python files passed static syntax/error checks using DDO `.venv-vmm`.
- Contracts prepared for live-ticket registration recycling, preserved bytes
  after trim, re-upload without reread, native fallback below/above ring-slot
  size, refusal dispatch and independent capacity validation.
- **No CPU/CUDA tests or benchmarks executed in this correction round**, at
  the user's request to complete implementation first. Previous measurements
  refer to earlier code/DLLs; consolidated functional, capacity and baseline
  regression gates remain required before any performance conclusion.

Subsequent user-approved consolidated functional gate, same DLL hash:
**63 passed in 5.67s**, exit code 0, no failures. This supersedes the pending
functional-gate status above, not the pending capacity/performance gates.
Executed from DDO with its own `.venv-vmm`, without changing memory ceilings
or running LTX. The source-cache suite includes persistent-byte preservation,
live-ticket registration recycling and retained-byte staging fallback for
requests smaller/larger than one ring slot. CPU refusal-dispatch/configuration
contracts and selected public transport/fast-runtime CUDA controls also passed.
No full-model capacity, linear/NVMe performance baseline or throughput claim.

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\ddo-vmm-allocator\src"
$env:DDO_VMM_LIBRARY_PATH="$PWD\ddo-vmm-allocator\build\Release\ddo_vmm.dll"
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_source_cache_config.py ddo-vmm-allocator/tests/test_gpu_prefetch.py ddo-vmm-allocator/tests/test_source_cache.py tests/test_vmm_native_transport.py tests/test_vmm_fast_runtime.py -q -x -p no:cacheprovider --basetemp .pytest-persistent-source-v8
```

Capacity gate attempt, same ABI v8 DLL and DDO environment: **1 passed,
1 failed in 5.96s**, exit code 1; stopped at the first failure as approved.
The legacy-v4 1536 MiB ring / 512 MiB cache control passed. Persistent-v5
completed sixteen numerical DMA comparisons with only 512 MiB of source reads
and peak/current registration within 128 MiB, then failed a test assertion
requiring sixteen registrations: the actual count was fifteen. A still
registered segment legitimately served another DMA without a new registration.
The corrected contract permits 14–16 registrations (two reusable segments),
checks registration/unregistration balance and exact DMA/hit/miss accounting.
Runtime, DLL, memory ceilings and workload are unchanged. The later trim,
fallback-volume and post-close assertions were not reached; this is **not a
passing persistent capacity gate**. Repeat requires renewed user approval.

User-approved repeat of the exact same command, DLL and memory ceilings,
with only the corrected test expectation: **2 passed in 6.14s**, exit code 0.
Both legacy-v4 and persistent-v5 capacity cases passed. Persistent storage
retained 512 MiB alongside 1536 MiB staging, kept peak/current source
registration within 128 MiB, verified all sixteen numerical transfers with
only 512 MiB source reads, and reported no retained-source staging fallback.
Registration trim preserved all retained bytes; clear/close refunded cache
commit, registrations and staging ledgers. The aggregate admission guard
also rejected an extra staging slot while the retained storage filled its cap.
This closes the selected 2 GiB host-capacity contract, not full-model coverage
or LTX throughput. The failed first attempt remains recorded above.

### Persistent ABI v8 LTX diagnostic A/B — 2026-09-30

Follow-up full-source capacity gate, explicitly approved, same ABI v8 DLL:
**1 passed in 15.08s**, exit code 0. Used four existing immutable transformer
shards without loading the model, DDO `.venv-vmm`, 8 GiB source-admission
RAM/commit headroom, 1536 MiB staging and a 2 MiB GPU destination.
File payload totaled 26,248,801,544 bytes; 394 aligned 64 MiB segments committed
26,440,892,416 bytes (24.625 GiB). Aggregate host ceiling was 28,051,505,152
bytes (26.125 GiB). Registration current/peak remained 536,870,912 bytes.
First pass: 394 misses, zero hits. Second pass: 394 hits, no further source
reads; source-read bytes remained exactly 26,248,801,544. Native registrations
rose from 394 to 781, unregistrations from 386 to 773, with zero registration
waits, admission denials, bypasses, pressure eviction and retained-source
ring fallback/host-copy bytes. Trim preserved storage and removed registrations;
clear/close returned cache commit, registration and staging ledgers to zero.
Only 17-byte samples per segment were DMA-verified (13,396 total bytes across
two passes); this is source coverage/lifetime validation, **not full-weight DMA
throughput or LTX performance**. Pre-gate read-only RAM query observed 45.450
GiB available physical memory and 69.974 GiB available commit capacity.
No LTX benchmark at this larger capacity has yet run. The earlier 512 MiB
A/B below remains a separate measured configuration.

User-approved pair, DDO `.venv-vmm`, Release DLL SHA256
`F345CA76B8AB5BDD052AA38DADE264DCF1D835C569A73B7E62C5AF80C48BCCC8`.
Public native runtime; BF16, 1280x704, seed 43, fake prompt, no PAG, local
files, four steps, one generation/prepare, 4 GiB arena, 1536 MiB total
staging, retention off/512 MiB. Source registration default equals retention
capacity; no budget tuning or further runs. Both exited 0.

| Retained GiB | Load s | DDO setup s | Denoise s | Steps 2–4 mean s | Total s | Peak VRAM/RAM GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.3770 | 1.3749 | 36.3550 | 8.8628 | 43.9194 | 6.4940 / 2.8011 |
| 0.5 | 0.3728 | 1.3622 | 40.1332 | 9.9731 | 47.7777 | 6.5307 / 3.3020 |

Step times: off `[9.7660, 9.0082, 8.6352, 8.9450]`, on
`[10.2145, 9.9289, 9.9543, 10.0359]`. Both PNGs match previous numerical
output SHA256 `87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.
Both reports show 188 GPU-prefetched units/consumers, zero prefetch capacity
skips/cancellations, fault failures, temporary tensors or execution mismatch.

H2D remained 93,927,995,136 bytes in both cases. Cache-off ring reads were
93,927,995,136 bytes; cache-on ring reads 93,430,144,752 bytes plus
536,870,912 bytes of persistent-source fill. Retention delivered only
497,850,384 DMA bytes, with 23 hits and 5549 bypasses. It retained/committed
512 MiB, registered eight segments once, and reported zero unregisters,
registration waits, pressure eviction, retained-source ring fallback or host
copy bytes. Therefore this workload exercised persistent retention but not
registration recycling: its registration ceiling equaled retained capacity.
Most model traffic still used ring reads, and the new lifetime split did not
reduce overall H2D volume. Native enqueue/read wall counters were
20.0758/19.9611 s off and 20.4679/19.6101 s on; these overlapping host-side
counters are not isolated device DMA durations and must not be added together.

No speedup demonstrated: cache-on denoise was approximately 10.4% slower in
this single pair. Standby purge was skipped due to missing privilege; OS-cache
and environment variation prevent attributing the timing difference to one
mechanism. The fixed 512 MiB buffer is partial coverage, not a full-model
persistent source. Do not repeat this configuration expecting full-model
traffic elimination or treat passing functional gates as performance success.
Artifacts are in the sibling host's `outputs/ltx_image_modular_vmm/persistent_v8`
(JSON under `metrics/`), preserving earlier outputs. No full-model buffer,
real-prompt/text-encoder run or final eight-step benchmark was executed.

### Full-source LTX, registration capped at 512 MiB — 2026-09-30

Explicitly approved single run using DDO `.venv-vmm` and the same ABI v8 DLL.
Preflight required >=34.125 GiB available physical RAM and commit; observed
46.390/72.046 GiB respectively. Retention ceiling 24.625 GiB, source
registration ceiling 512 MiB, staging 1536 MiB, arena 4 GiB. Otherwise same
BF16 fake-prompt/no-PAG/local-files/1280x704/seed43/four-step diagnostic.
Exit code 0; PNG SHA256 matches prior outputs:
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.

| Load s | DDO setup s | Denoise s | Steps 2–4 mean s | Total s | Peak VRAM/RAM GiB |
| --- | --- | --- | --- | --- | --- |
| 0.4358 | 1.5625 | 122.5626 | 28.0650 | 134.6286 | 6.3586 / 27.0430 |

Steps `[38.3676, 30.4869, 27.1274, 26.5805]`. A during-run host memory sample
observed 18.756 GiB physical RAM and 38.369 GiB commit available. No pressure
evictions, admission denial, fault failure, temporary fallback or execution
mismatch was reported. GPU look-ahead enqueued/consumed 188 units, with no
capacity skips or cancellations.

Native retention supplied all 93,927,995,136 H2D bytes. Ring reads and ring
read/slot-wait seconds were **zero**, cache bypasses zero. Demand admitted
388 file windows: retained payload/source reads 25,846,148,360 bytes,
committed storage 26,038,239,232 bytes, 5192 hits. Demand need not fill every
shard window; the earlier capacity gate deliberately covered all 394 windows.
Registered current/peak stayed 536,870,912 bytes. Registration churn was
**1413 registrations / 1405 unregistrations**; zero registration-capacity
event waits, retained-source staging fallback or host-to-host copy bytes.
Snapshot contained 15 live source tickets before cleanup; it is not a
post-close ledger snapshot and must not be presented as a leak or zero-ticket
cleanup measurement. Normal harness cleanup completed and the process exited.

Host DMA-enqueue scope was 57.3920 s, versus 20.0758 s in the cache-off control;
that scope includes native registration work and is not pure device DMA time.
Source registration/unregistration durations are not separately instrumented.
Thus persistent coverage eliminated ring/file rereads but **did not improve
performance**: denoise 122.5626 s versus 36.3550 s cache-off, steady-state
28.0650 versus 8.8628 s. Repeated registration of a large retained buffer under
a 512 MiB cap is a concrete suspect, not a measured causal attribution.
No registration-cap increase, retest or tuning was performed. Standby purge
again lacked privilege; the slow connector load also indicates environment
variation, without explaining away the steady-state regression.
Artifacts: sibling `outputs/ltx_image_modular_vmm/persistent_v8_full`, JSON in
`metrics/`, suffix `cache24.625gb_registered0.5gb`.

### Native registration timing instrumentation — prepared

Follow-up instrumentation prepared, not tested: source-cache ABI v9/stats v6
adds persistent registration/unregistration and capacity-wait host seconds,
with old stats prefixes preserved. No capacities, policy or synchronization
changed. Release compiled; DLL SHA256
`16E872447364F7BFE155CA4739145715A6337047EF5FA6B1FDA2A64C25BA009B`.
New/extended contracts and an instrumented workload require fresh approval;
no CPU/CUDA test or inference run executed with this DLL yet. Earlier timings
remain associated with the ABI v8 DLL, not this instrumented build.

### Instrumented ABI v9 validation attempt

Approved ABI v9 consolidated sequence: selected contracts **65 passed in
7.37s**, exit code 0, including timing fields and older snapshot-prefix
sentinels. The following instrumented full-source LTX attempt passed memory
preflight (45.694 GiB RAM / 71.230 GiB commit available, minimum 34.125 GiB),
then exited 1 during initialization with
`OPENSSL_Uplink(...,08): no OPENSSL_Applink`, after the Hugging Face deprecated
symlink-argument warning and before any prompt/transformer/denoise event was
reported. No inference JSON/PNG or registration-time measurement was produced.
The approved sequence stopped there; no retry, dependency modification,
capacity increase or additional benchmark was executed. The failure text
alone does not establish its root cause or implicate/exonerate the allocator.
Output directory selected: sibling
`outputs/ltx_image_modular_vmm/registration_timing_v9`.
Thus ABI v9 selected functional contracts passed, but instrumented performance
measurement remains pending. The earlier ABI v8 full-source result remains
122.5626 s and must not be represented as a measurement of these new timers.

### OpenSSL startup diagnosis

Read-only startup diagnosis after the failed ABI v9 attempt, no CUDA or model
run: DDO Python reproduced the same fatal OpenSSL message with only
`ssl.create_default_context()`, without importing Torch/DDO. Python's SSL
context constructor and certificate loading succeeded separately. The launch
environment had `SSLKEYLOGFILE` set; Python's default-context helper activates
its keylog filename when this variable is present. Removing this variable
inside the isolated diagnostic child made default-context creation succeed,
with hostname checking enabled and `CERT_REQUIRED` retained. No keylog path or
contents were read/disclosed. Loaded `_ssl`, libssl and libcrypto came from
the same uv-managed CPython 3.12.11 runtime, not a confirmed cross-package DLL
mix. Both DDO and host venv configurations point to the same base Python.
Thus the inherited TLS-keylog activation is a reproduced trigger; OpenSSL's
underlying application/runtime bridge defect was not repaired or independently
characterized. Proposed workaround is launch-local omission of the variable,
not TLS verification bypass, dependency changes or global environment edits.
No new benchmark was run; approval is required for the instrumented repeat.

### ABI v9 instrumented full-source repeat — 2026-09-30

User-approved repeat after launch-local removal of `SSLKEYLOGFILE` (no global
environment or dependency change). Same ABI v9 Release DLL hash
`16E872447364F7BFE155CA4739145715A6337047EF5FA6B1FDA2A64C25BA009B`.
Selected functional gate had passed 65 tests in 7.37 s. Preflight observed
45.407 GiB physical RAM and 70.982 GiB commit available (minimum 34.125 GiB).
Same four-step diagnostic, 24.625 GiB retention ceiling, 512 MiB source
registration, 1536 MiB staging and 4 GiB arena. Exit 0; no OpenSSL abort.

| Load s | DDO setup s | Denoise s | Steps 2–4 mean s | Total s | Peak VRAM/RAM GiB |
| --- | --- | --- | --- | --- | --- |
| 0.4309 | 1.5802 | 132.1528 | 29.9801 | 145.8549 | 6.5367 / 27.1082 |

Steps `[42.2129, 28.6858, 30.6100, 30.6445]`. PNG SHA256 remained
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.
During-run RAM/commit sample: 18.162/36.533 GiB available. No fault failures,
temporary tensors, admission denials, pressure eviction, execution mismatches
or prefetch cancellation/capacity skips; 188 GPU-prefetched units/consumers.

Source accounting matched the preceding full-source case: 388 admitted
windows, 5192 hits, zero bypass/ring-read bytes, 25,846,148,360 source-fill
bytes, 93,927,995,136 cache DMA bytes, 26,038,239,232 committed bytes,
512 MiB current/peak registration, 1413 registrations and 1405 unregistrations.
No retained-source ring fallback/host-copy bytes. Pre-cleanup snapshot had
15 live tickets; this is not a post-close snapshot.

Measured native registration helper time **0.5851 s**, unregistration
**6.3593 s**, explicit registration-capacity wait **0 s**. Their direct wall
cost sums to 6.9444 s, approximately 5.25% of denoise, and does **not** explain
the full regression. This weakens the earlier main-bottleneck hypothesis:
registration churn exists, but its measured call duration is not the dominant
132.1528 s cost. It does not rule out indirect registration effects on driver
transfer behavior. The host upload scope was 80.2949 s and includes source
read readiness, cache scheduling/retirement, register/unregister and enqueue;
it is not an isolated CUDA device DMA duration. Do not subtract nested scopes
and label the remainder pure DMA, or add them as independent measurements.
No speedup, capacity tuning or further run followed this result. Standby purge
remained unavailable. Next diagnosis should isolate the native transfer/ready
path and compare ring versus retained-source behavior, rather than assume
raising registration capacity necessarily solves performance.
Artifacts: sibling `outputs/ltx_image_modular_vmm/registration_timing_v9`,
JSON in `metrics/`, suffix `cache24.625gb_registered0.5gb`.

### Native transfer/retirement timing split — prepared, not executed

After the measured ABI v9 result ruled out registration call wall time as a
complete explanation, ABI v10 / stats v7 adds direct-source readiness, native
H2D submission, event-record and ticket-close synchronization wall seconds.
No ceilings, admission policy, event ownership or synchronization behavior
changed; old stats v1–v6 prefixes remain supported. Broader host upload scopes
still include Python scheduling/query work and must not be called pure DMA.
Release compiled; DLL SHA256
`BF2FBFE702A19166DFB13BBD3566D40D8E129E702DC3E2E532E8FFFD9917D5FF`.
Static checks passed, contracts prepared. **No CPU/CUDA test or benchmark
executed with ABI v10**; the 65-test gate and measured 132.1528 s diagnostic
belong to ABI v9. A new explicitly approved consolidated validation is required.

### Native ABI v10 approved validation — 2026-09-30

The selected consolidated CPU/CUDA gate passed: **66 passed in 6.13s**,
exit 0, using DDO `.venv-vmm`, `-x`, and basetemp
`.pytest-transfer-timing-v10`. Same five suites as the selected ABI v8 gate
above; launch omitted inherited `SSLKEYLOGFILE` only in the launch process.
DLL SHA256 remains
`BF2FBFE702A19166DFB13BBD3566D40D8E129E702DC3E2E532E8FFFD9917D5FF`.

The subsequent approved LTX diagnostic used the existing public native route,
24.625 GiB source storage, 0.5 GiB source registration, 1.5 GiB native staging,
4 GiB GPU residency, 1280×704, seed 43, four steps, fake prompt, no PAG and
local files only. Available RAM/commit before launch: 46.555/71.844 GiB;
the required minimum was 34.125 GiB. No parameter or policy changes.

| Measurement | Result |
| --- | ---: |
| Denoise | 136.9374 s |
| Steps 2–4 mean | 31.6803 s |
| Total | 150.4132 s |
| Peak VRAM / RAM | 6.3388 / 27.1080 GiB |
| Broad host upload scope | 66.6854 s |
| Native source readiness | 5.2250 s |
| Native H2D submit call | 0.0377 s |
| Native event record call | 0.0049 s |
| Native ticket-close synchronization | 0.0030 s |
| Native registration / unregistration | 0.5468 / 6.4005 s |
| Registration-capacity wait | 0 s |
| H2D bytes | 93,927,995,136 |
| Source file-read bytes | 25,846,148,360 |
| Retained committed bytes | 26,038,239,232 |

Exit 0. PNG SHA256 matches the prior controls:
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.
Artifacts: sibling-host `outputs/ltx_image_modular_vmm/transfer_timing_v10/`,
with the public report JSON under its `metrics/` directory.

Functional validation passed; performance did not improve. These host-wall
timers exclude mutex acquisition and parts of scheduling/query work, overlap
broader scopes, and do not measure device DMA duration. Do not assign the
unaccounted broad-scope time to any one cause without evidence. Full retained
host storage is not full CUDA backing or full GPU residency: the virtual
model arena still has a 4 GiB physical residency ceiling. Retaining source
bytes eliminated repeat source reads, not repeated H2D traffic. No additional
benchmark, capacity increase or baseline performance run was authorized or
performed in this sequence.

### Execution-buffer connection — CUDA pending

Execution-buffer connection prepared on 2026-09-30: native admission boundary,
reusable pinned backing, retained-source transport and native compute fencing
are connected in the persistent-source route. Capacity is carved out of the
existing GPU ceiling, not added to it. Release compiled; DLL SHA256
`86DFD402FCAA0E9D4853697D920674CB3457EE7E8C2BA3E5A58C0EC1191AE505`.
CPU ownership/dispatch and existing look-ahead contracts: **11 passed in
4.67s**. A numerical/capacity CUDA contract is prepared but **not executed**.
No inference timing or performance gain is claimed for this DLL. The earlier
ABI v10 benchmark belongs to the previous DLL hash and streaming architecture.

First approved CUDA sequence, same DLL hash, DDO environment, `-x` and
basetemp `.pytest-execution-buffer-cuda`: **1 failed in 5.03s**, exit 1.
Stopped at the first test. The first forward passed numerical parity and
reported execution-buffer uploads, then the fixed 1 MiB residency assertion
failed with 2 MiB observed. Native creation rounds the requested page to CUDA
minimum allocation granularity. The test now requires exact unit capacity
rounded to the reported native page size, with unchanged aggregate ceiling.
Runtime/DLL/parameters are unchanged. Later assertions, subsequent passes and
all remaining suites were not reached; this is not a passing capacity or
performance gate. No LTX benchmark or additional CUDA test was executed.

User-approved repeat, same command/DLL/ceilings/basetemp and only corrected
alignment expectation: **32 passed in 5.33s**, exit 0. The new public native
buffer case completed all three numerical comparisons, checked exact
page-aligned capacity and total weight residency <=4 MiB, reused signatures,
performed no additional source reads after the first pass, and reported zero
tail evictions and temporary tensors. Existing public native transport,
fast-runtime and source-cache contracts in the approved sequence also passed.
This closes the selected functional gate. It does not validate LTX throughput,
full-model overlap or linear/NVMe performance regression. No new benchmark was
performed and previous inference timings are not measurements of this DLL.

### Native execution-buffer LTX diagnostic — 2026-09-30

Explicitly approved four-step case, DLL
`86DFD402FCAA0E9D4853697D920674CB3457EE7E8C2BA3E5A58C0EC1191AE505`.
DDO environment, 1280×704, seed 43, fake prompt, no PAG, local files only;
4 GiB total weight-residency ceiling, 24.625 GiB persistent source, 512 MiB
source registration and 1.5 GiB native staging. No extra case or budget change.
Preflight available RAM/commit: 46.713/72.575 GiB, required 34.125 GiB.
During denoise: 19.672/38.709 GiB available. Standby purge remained unavailable
because the launch token lacked the required privilege.

| Measurement | Result |
| --- | ---: |
| Transformer load / DDO setup | 0.4701 / 1.5600 s |
| Denoise | 95.6604 s |
| Steps 1–4 | 26.4036 / 23.0436 / 22.4033 / 23.8094 s |
| Steps 2–4 mean | 23.0855 s |
| Total | 109.2922 s |
| Peak VRAM / RAM | 6.3416 / 27.1073 GiB |
| Model-arena / execution-buffer residency | 3.03125 / 0.53125 GiB |
| Aggregate weight residency | 3.5625 GiB |
| Arena evictions / tail / pressure | 0 / 0 / 0 |
| Signature hits | 432 |
| Execution-buffer uploads / payload bytes | 168 / 90,300,306,432 |
| Total H2D bytes | 93,525,317,376 |
| Broad host upload scope | 65.5584 s |
| Source readiness | 2.1207 s |
| Registration / unregistration | 0.5368 / 7.0079 s |
| Native H2D-submit / event-record calls | 0.0314 / 0.0037 s |
| Ticket-close synchronization / registration-capacity wait | 0.0014 / 0 s |
| Source file-read bytes | 25,846,148,360 |
| Source hits / misses / bypasses | 5180 / 388 / 0 |
| Native source registrations / unregistrations | 1401 / 1393 |
| Temporary tensors / registration-refusal host copies | 0 / 0 |

Exit 0. Output PNG SHA256 matches prior controls:
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.
Artifacts are in sibling-host
`outputs/ltx_image_modular_vmm/native_execution_buffer/`, JSON under `metrics/`.

The intended structural change is observed: zero arena evictions versus 2663
in the preceding full-source diagnostic, with backing reused inside the same
ceiling. Denoise decreased from 136.9374 to 95.6604 s (about 30.1%), but remains
about 2.63× the earlier 36.3550 s cache-off control. Therefore the agreed
performance criterion did not pass. These are sequential diagnostic runs,
not repeated statistical estimates, and the comparison cannot isolate all
causes. H2D volume remained almost unchanged and the broad host upload scope
remained high (65.5584 versus 66.6854 s). The measured narrow native scopes
do not explain that whole scope or measure device DMA duration. In particular,
event creation/destruction, queries and other boundary work are not all covered.
The connected buffer is a single reusable slot: it protects reuse correctly,
but does not establish full transfer/compute overlap. No extra CUDA run,
real-prompt/eight-step benchmark or parameter increase was performed.

### Archived infrastructure validations

This is an infrastructure contract, not a performance benchmark. On Windows,
the DDO `vmm-dev` environment (Python 3.12, Torch `2.12.1+cu132`) validated
the pinned `ddo-vmm-allocator` submodule at commit `3f0a916` with **62 passed**
tests in **4.82 s** after the persistent-source closing contract was added.
The follow-up route-separation contracts passed: `vmm_fast_runtime` completed
the file-descriptor/no-CPU-source test in **3.21 s**, and the protected
`vmm_runtime` baseline plus `vmm_fast_runtime` passed together in **4.06 s**.
The transport-slot event contract passed in **3.67 s**, followed by the
stream-dependency contract in **3.76 s** and a deterministic native-admission
temporary-fallback contract in **3.64 s**. These remain functional contracts,
not latency measurements. The consolidated gate after those additions passed
with **67 tests in 4.25 s**.
The VMM capacity-profile contract then passed with **3 tests in 5.73 s**: it
records native capacity without changing the configured VMM budget. This is a
generic component contract; it is not yet evidence of text-encoder or LTX
benchmark performance. The updated consolidated gate passed with **69 tests in
4.66 s**.
The command set `DDO_VMM_LIBRARY_PATH` to the local Release DLL and ran both
the allocator's CUDA contracts and DDO's test suite.

The next performance gates remain separate: the fixed diagnostic workload
(fake prompt, `1280x704`, seed `43`, 2/4 steps, 512 MiB cache and 4 GiB active
budget) and then the comparable real-prompt 8-step workload. Do not treat this
test gate as evidence of latency or memory improvement.

### Current `vmm_fast` diagnostic result

The first complete Windows native-arena diagnostic used the fixed fake-prompt
workload (`1280x704`, seed `43`, 2 steps), a 4 GiB active VMM budget and the
0.625 GiB staging buffer required by the host harness. It ran in a detached
child process so the test host could not terminate the CUDA process early.

| Route | Denoise | Total | Peak VRAM | Peak RAM | Resident arena | Materializations | Temporary fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Windows `vmm_fast_runtime` diagnostic | `29.3821 s` (`14.6910 s/it`) | `45.3 s` | `6.03 GiB` | `5.50 GiB` | `4.00 GiB` | `2,336` | `0` |
| Same gate, batched weight/bias staging | `28.7276 s` (`14.3635 s/it`) | `44.7 s` | `6.04 GiB` | `5.44 GiB` | `4.00 GiB` | `2,336` | `0` |
| Block execution units with 512 MiB transport subbatches | `31.8319 s` (`15.9156 s/it`) | `51.5 s` | `6.18 GiB` | `5.50 GiB` | `4.00 GiB` | `2,304` | `0` |
| Block units, nonblocking prefetch eligibility | `29.9425 s` (`14.9709 s/it`) | `48.9 s` | `6.97 GiB` | `5.57 GiB` | `4.00 GiB` | `2,304` | `0` |
| Block units, native registered-host shard transport | `28.7847 s` (`14.3924 s/it`) | `45.5 s` | `6.11 GiB` | `5.57 GiB` | `4.00 GiB` | `2,304` | `0` |
| Native shard transport with numeric execution queue and bounded look-ahead | `25.4624 s` (`12.7311 s/it`) | `42.6 s` | `6.07 GiB` | `5.40 GiB` | `4.00 GiB` | `2,304` | `0` |
| Native segmented-window shard transport | `24.0938 s` (`12.0469 s/it`) | `41.7 s` | `6.08 GiB` | `5.51 GiB` | `4.00 GiB` | `2,304` | `0` |
| Native C++ asynchronous source tickets | `24.7358 s` (`12.3679 s/it`) | `42.0 s` | `6.09 GiB` | `5.57 GiB` | `4.00 GiB` | `2,304` | `0` |
| Registered Windows windows with persistent native handles | `24.4081 s` (`12.2041 s/it`) | `42.1 s` | `6.06 GiB` | `5.51 GiB` | `4.00 GiB` | `2,304` | `0` |
| Bounded persistent native source-worker queue | `23.4671 s` (`11.7336 s/it`) | `40.5 s` | `6.08 GiB` | `5.50 GiB` | `4.00 GiB` | `2,304` | `0` |
| Native cyclic-pass eviction policy | `21.5207 s` (`10.7604 s/it`) | `38.2325 s` | `6.07 GiB` | `5.50 GiB` | `4.00 GiB` | `22,558` pages | `0` |
| Cyclic-pass policy with native eviction priority queue | `21.1658 s` (`10.5829 s/it`) | `38.0285 s` | `6.07 GiB` | `5.50 GiB` | `4.00 GiB` | `22,558` pages | `0` |
| 32 MiB arena pages, tail eviction, positioned 2 MiB native source and global allocator-pressure contract | `11.5994 s` (`5.7997 s/it`) | `28.6917 s` | `6.16 GiB` | `5.56 GiB` | `4.00 GiB` | `1,443` pages | `0` |
| Same native route, four-step steady-state gate | `24.5130 s` (`6.1283 s/it`) | `39.6830 s` | `6.18 GiB` | `5.49 GiB` | `4.00 GiB` | `2,791` pages | `0` |
| Same four-step gate, progressive per-slot source completion → DMA | `22.5448 s` (`5.6362 s/it`) | `35.6823 s` | `6.10 GiB` | `5.49 GiB` | `4.00 GiB` | `2,791` pages | `0` |
| Same two-step gate, progressive per-slot source completion → DMA | `11.9267 s` (`5.9630 s/it`) | `25.5583 s` | `6.18 GiB` | `5.49 GiB` | `4.00 GiB` | `1,443` pages | `0` |
| Per-source-chunk DMA diagnostic (rejected) | `12.2607 s` (`6.1299 s/it`) | `26.3 s` | `6.28 GiB` | `5.43 GiB` | `4.00 GiB` | `1,443` pages | `0` |

This is a functional 2-step gate, not the 8-step comparable benchmark and not
a performance claim. Its purpose is to validate native mapping, asynchronous
uploads, signatures and cleanup under the configured VMM capacity. The final
comparison remains the real-prompt 8-step workload with its separately
approved command and environment record.

The native transport gate replaced the runner-owned PyTorch staging buffers
with DDO-owned registered-host slots. It read `51.60 GiB` directly from shard
files in `9.9164 s` and enqueued `51.60 GiB` of DMA in `10.0480 s`, with no
slot-reuse wait and no temporary fallback. Its `14.3924 s/it` is close to the
earlier best `14.3635 s/it`, so it proves the route and its accounting but is
not a performance improvement claim. The remaining bottleneck is the volume
of source bytes re-read per denoise step, not runner-side staging ownership.

The first native C++ asynchronous-source ticket gate kept the same fixed
two-step workload and safety envelope. It completed with zero fallback or VMM
admission/fault errors, but measured `24.7358 s` (`12.3679 s/it`), slower than
the prior `12.0469 s/it` segmented-window result. Its cumulative native source
read time was `24.3038 s` for `51.60 GiB`, DMA enqueue time was `4.5179 s`, and
the ring reported `284` reuse waits (`0.8491 s`). This establishes allocator
ownership of the read-ticket → event → DMA lifecycle, not a throughput gain:
the then-current provisional implementation still opened a file for each
window and used per-window task creation. The allocator now keeps a persistent
native handle per registered slot and shard, with a `VirtualAlloc` window
registered once with CUDA so Windows workers can fill it directly. Its
contract tests pass. The ring now also uses a bounded persistent native worker
queue (at most one request per slot) rather than creating a thread for each
window. This revised source path has not yet received a LTX latency
measurement; the VMM capacity budget remains unchanged.

The registered-Windows-window diagnostic completed at `24.4081 s`
(`12.2041 s/it`) with `6.06 GiB` peak VRAM, `5.51 GiB` peak RAM, and no
fallback, fault, or admission failure. It improves on the earlier C++ ticket
probe (`12.3679 s/it`) and reduced DMA enqueue accounting to `3.4407 s`, but
does not beat the `12.0469 s/it` segmented-window result. Native source reads
still accumulated `24.0091 s` for the same `51.60 GiB`; `281` slot reuse waits
totalled `0.9689 s`. Treat it as a validated native ownership/safety result and
a partial transport improvement, not as the target throughput result.

The bounded native source-worker queue improved the same gate to `23.4671 s`
(`11.7336 s/it`), the best native result so far, with `6.08 GiB` peak VRAM,
`5.50 GiB` peak RAM, and zero temporary fallback or VMM error. Source-read
accounting was `23.0600 s`; `281` reuse waits accounted for `1.2388 s`. The
result validates persistent workers instead of per-window thread creation, but
not the throughput target. The next native block batches a ticket's source
completion and H2D enqueue inside the allocator to remove per-window Python to
C++ scheduling overhead.

The batched transport reduced staging-slot reuse waits from `2,334` to `1,166`
and their measured wait time from `0.0793 s` to `0.0082 s`; DMA enqueue time
fell from `0.2215 s` to `0.1316 s`. Shard reads remained approximately
`11.72 s` for `52.49 GiB`, so the next optimization target is source grouping
at the transformer-block execution boundary rather than a larger VMM budget.

The block-unit gate is the first successful run of the new execution boundary:
`48` block hooks manage `1,152` attached tensors. Its `512 MiB` staging slots
required transport subbatches within each block; it completed with no fault
failure, capacity admission failure, or temporary tensor. It measured `9.80 s`
reading `48.06 GiB` of shards, `189` slot-reuse waits (`0.0026 s`) and `0.0546 s`
of DMA enqueue time. It is deliberately recorded as a functional result, not a
speed improvement: its `15.9156 s/it` is slower than the earlier leaf-level
diagnostics, so source reuse/prefetch remains the next block-4 investigation.

The first speculative look-ahead saturated the transport ring and was not
retained as an active optimization. The replacement only prefetches a next
block when it fits in one immediately available slot. In the same gate all `94`
attempts were correctly skipped because the block payloads require subbatches;
there was no added materialization or slot wait (`0.0024 s`), no fallback, and
the observed `14.9709 s/it` is a functional nonblocking baseline rather than
evidence that prefetch improved the result. A useful prefetch needs native or
transport-level subbatch scheduling, not a Python hook that reserves a whole
next block.

The next native-transport diagnostic fixed the execution queue to follow the
numeric module suffix rather than lexical name order. The two-step gate kept
the same 4 GiB arena and 512 MiB source-slot limit, completed with no
temporary fallback or slot wait, and reduced denoise to `12.7311 s/it`.
It recorded `94` bounded look-ahead attempts, `2,068` consumed prefetched
bindings and zero execution-plan mismatches. Source reads still accounted for
`25.0075 s` across `51.60 GiB`, so this is a correctness and partial-overlap
improvement only; it is not the comparable 8-step benchmark or evidence that
the final throughput objective has been reached.

The following gate retained the same logical 512 MiB transport capacity but
split it internally into 64 MiB native registered-host windows, each with its
own completion event. It reduced denoise further to `12.0469 s/it`, with no
slot wait, fallback, fault failure or execution-plan mismatch. Native source
read accounting fell from `25.0075 s` to `23.6611 s` and DMA enqueue time was
`2.7412 s`, while the route still moved `51.60 GiB` in two steps. This is
therefore a verified transfer-overlap improvement, not evidence that the
final throughput objective has been reached.

The native arena now records materialized, reused, and evicted physical pages.
On the same fixed two-step gate, a cyclic-pass eviction policy retained the
oldest unpinned virtual pages rather than the most recently consumed pages. It
completed in `21.5207 s` (`10.7604 s/it`), with `6.0745 GiB` peak VRAM and
`5.4998 GiB` peak RAM. It had zero fault/admission errors and zero temporary
fallbacks. The report recorded `22,558` materialized pages, `4,348` reused
pages, and `20,510` evicted pages; compared with the preceding instrumented
run (`24,606` / `2,300` / `22,558`), reuse increased and transfer volume fell
to `44.18 GiB`. Native shard reads accounted for `20.8966 s`, DMA enqueue for
`4.1561 s`, and slot waits for `0.6063 s` across `264` waits. This validates
the cycle-reuse hypothesis under the unchanged safety envelope, but remains a
two-step fake-prompt diagnostic and does not establish the final benchmark
target.

Replacing the arena-wide victim scan with a native priority queue preserved the
same residency accounting (`22,558` materialized, `4,348` reused, `20,510`
evicted pages) and safety result. It reduced denoise by `0.3549 s` to
`21.1658 s` (`10.5829 s/it`), with `6.0720 GiB` peak VRAM, `5.5032 GiB` peak
RAM, and zero fallback or error. The small improvement confirms that victim
selection was not the dominant latency source: the report still attributes
`20.5470 s` of accumulated native shard-read time to `44.18 GiB` of source
data. Further work must improve the native source-to-DMA overlap rather than
change the VMM budget or add runner-side residency logic.

The subsequent native-contract gate changed the arena to 32 MiB logical pages,
preserved its contiguous low-address prefix by evicting the eligible tail,
split persistent positioned shard reads into 2 MiB requests across up to eight
native workers, and connected configured allocation headroom to native arena
reclamation. Under the unchanged fake-prompt two-step workload and 4 GiB
active budget, it completed in `11.5994 s` (`5.7997 s/it`) with `6.1551 GiB`
peak VRAM, `5.5606 GiB` peak RAM, zero fallback or admission failures, and
zero source-read failures. The report recorded `1,443` materialized pages,
`1,315` tail evictions, a 95-page resident prefix, 25,390 source chunks, and
zero pressure evictions. The last counter means this particular diagnostic did
not need to reclaim model pages for concurrent ordinary CUDA allocations; the
allocator-wide pressure contract is covered separately by its CUDA test. This
is a substantial diagnostic improvement, but still not the real-prompt,
8-step comparable benchmark or the final throughput target.

The four-step steady-state gate kept every route parameter unchanged. Before
progressive submission it completed in `24.5130 s` (`6.1283 s/it`); steps two
through four were `5.9718`, `5.7576`, and `5.9218 s`. It remained fallback-free
with `6.1757 GiB` peak VRAM and `5.4892 GiB` peak RAM. Source reads accumulated
`23.2606 s` for `93.11 GiB`, while DMA enqueue accounted for `6.0216 s`.

The progressive native ticket change waits for each individual source slot and
immediately queues its own DMA, while later slots continue reading. Under the
same four-step gate it reduced denoise to `22.5448 s` (`5.6362 s/it`), an
`8.0%` improvement, with `6.1027 GiB` peak VRAM, `5.4944 GiB` peak RAM, zero
temporary fallback and zero fault/admission/source-read failures. It recorded
`93.11 GiB` of source reads, `87.48 GiB` of file-backed DMA uploads,
`21.3066 s` cumulative source-read time, `5.8344 s` DMA-enqueue time and
`1.8176 s` across `527` source-slot waits.
The gain validates slot-level source/DMA overlap without changing the 4 GiB
capacity budget. Source transport remains the dominant constraint; this is
still a fake-prompt diagnostic rather than the comparable real-prompt 8-step
benchmark.

The matching two-step gate stayed stable but did not reproduce a latency gain:
`11.9267 s` (`5.9630 s/it`) versus `11.5994 s` (`5.7997 s/it`) before the
progressive change. It kept zero temporary fallback and zero source-read,
fault, or admission failures, with `6.1789 GiB` peak VRAM and `5.4927 GiB`
peak RAM. Its first step was `6.4283 s` and its second `5.4978 s`; it recorded
`11.3858 s` cumulative source-read time, `3.7879 s` DMA enqueue time and
`1.2336 s` across `271` slot waits. Thus the four-step result is promising but
the change is not promoted as a universal improvement: the short gate's
performance criterion is not met and requires a focused first-step diagnostic
before treating this as a closed performance block.

A follow-up two-repeat same-process probe did not attribute the short-gate
result solely to process initialization: its first two-step repeat took
`13.5415 s` (`6.7707 s/it`) and the already-warm second repeat took `12.0391 s`
(`6.0196 s/it`). The probe was fallback-free, but it is diagnostic-only and
does not replace the single-repeat gate. A controlled native A/B comparison of
the ticket submission order is required before retaining the progressive path
as the default.

The direct per-source-chunk DMA diagnostic was rejected and removed after its
contract passed. It issued `24,365` individually completed chunk DMAs for
`25,390` source chunks (the remainder belonged to already reusable ranges),
but denoise regressed to `12.2607 s` (`6.1299 s/it`) and peak VRAM reached
`6.2784 GiB`. It remained fallback- and error-free, yet the added small-copy
submission cost outweighs the extra overlap on this workload. The transport
therefore retains the prior per-window DMA granularity.

The Windows memory-observability gate added report samples without changing the
4 GiB capacity budget. On the two-step diagnostic, the arena reserved `24.03
GiB` of virtual address space but began with `0 GiB` resident. Immediately
after the first denoise pass it had `4.00 GiB` resident; DXGI reported `5.89
GiB` local WDDM usage against a `6.90 GiB` local budget and only `1.56 GiB`
non-local usage against a `31.21 GiB` shared budget. The second pass remained
at `5.89 GiB` local and `1.57 GiB` non-local. Thus this route is not silently
materializing the full model through WDDM shared memory. A much larger value
shown by an external system monitor must be investigated as process-wide
commit/accounting or a separate host allocation path, not reproduced by
raising this arena's physical residency budget.

The native immutable-source foundation was implemented on 2026-09-21. Before
the change, the restored window transport passed its three selected CUDA
contracts. The added source ABI then compiled in Release and passed 14 CPU-only
file/ownership contracts plus the platform test (`15 passed`). These cover
positioned reads, Unicode paths, overflow/EOF bounds, retained ownership,
concurrent readers, write/delete sharing, metadata changes, ABI validation,
old-DLL diagnostics, repeated handle cleanup and no CUDA initialization.
The source shares the ring's existing file-reader primitives; it does not yet
enable persistent weight caching or change the runtime's transport selection.
The post-build CUDA regression subsequently passed all three selected window
transport contracts (`3 passed, 11 deselected in 5.33s`), without warnings or
CUDA failures, using the same DLL hash below. The approved invocation was
`pytest tests/test_native_ranges.py -m vmm -k
'ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks'
-p no:cacheprovider --basetemp ../.pytest-vmm-source-foundation`, with the DDO
`.venv-vmm` interpreter from the allocator directory. Memory parameters were
unchanged. Registered-cache implementation and performance gates remain pending.

Validation used the DDO `.venv-vmm` environment, with
`pytest tests/test_native_source.py tests/test_platform.py -q -p no:cacheprovider
--basetemp ../.pytest-native-source-integrity` from the allocator directory.
The resulting DLL SHA-256 is
`3F51F7D30648D1E194D733456F72E4F602F36802C409C4E416AD89DBD9867248`.
Parent revision was `c8846e2b002b34e1461aee2e91166622440dbd33` and allocator
revision `3f0a91659580a5b9d8b47b81403417d1c0335f87`, both with local changes;
those revisions alone do not reproduce the tested source tree. See
`docs/native-source-cache-plan.md` for implementation status and remaining gates.

The subsequent retained-slot cache increment compiled in Release and retained
`15 passed` in CPU-only source/platform tests (`4.90s`). Its six new CUDA
contracts and three existing transport regressions passed on 2026-09-22 after
user confirmation: `9 passed, 11 deselected in 7.79s`, without warnings or CUDA
failures. The gate verified byte-correct reuse after GPU eviction with one source
read for two uploads, retained partial-cache hits, alias reuse, ticket ownership,
two-stream transfers and rejection of insufficient headroom/changed sources.
Command from the allocator directory, using the DDO environment:
`../.venv-vmm/Scripts/python.exe -m pytest tests/test_source_cache.py tests/test_native_ranges.py -m vmm -k
'source_cache or ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks'
-p no:cacheprovider --basetemp ../.pytest-vmm-source-cache`.
The implementation retains native registered ring slots, reads through the same
worker pool, and protects independent H2D tickets with separate events. It is
limited to micro-contract use (hard capacity ceiling 128 MiB, eager slots);
progressive allocation, aggregate pressure and public-runtime integration are
not yet delivered. No inference benchmark or performance claim accompanies it.
DLL SHA-256: `C00F12D369E4C29DD22F854AD4A9D701807B1BF924C195A76186059F5DD63372`.

The next cache revision (2026-09-22) implements on-admission registration and
process-wide accounting for cache slots only. Creation registers zero bytes;
clear drains reads and refunds registrations. The strictest live cache cap
controls admission, with RAM/commit checks before each growth. Staging accounting,
automatic pressure reduction and inference integration remain pending. Release
compiled and CPU source/platform contracts passed (`15 passed in 4.59s`).
The approved CUDA gate passed on 2026-09-22: **12 passed, 11 deselected in 4.64s**,
with no failures or warnings. This includes growth/release, shared capacity,
strictest live cap and the v1 snapshot boundary check, plus previous regressions.
The command used the DDO environment from the submodule:

```powershell
& ..\.venv-vmm\Scripts\python.exe -m pytest tests/test_source_cache.py tests/test_native_ranges.py -m vmm -k 'source_cache or ticket_copies_each_completed_source_slot_to_arena or native_file_read_ticket_feeds_async_dma or uses_positioned_2mib_chunks' -p no:cacheprovider --basetemp ../.pytest-vmm-source-progressive
```

No inference benchmark or NVMe baseline rerun accompanied this micro gate;
no new performance result is claimed. DLL SHA-256:
`69DF3D95EB722959C71CC0F8E3E4D6ABECC63F1642EF0D82F20A65D24A049588`.

The next native micro revision (2026-09-29) includes eager staging rings in
process host registration accounting and cache admission. It preserves legacy
ring creation and adds an ABI v3 staging counter. Release compiled; its new
CUDA contracts passed the approved gate (`14 passed, 11 deselected in 6.50s`).
This validates staging capacity admission/refund and selected regressions;
no inference benchmark or performance claim is attached to this revision.
CPU source/platform contracts: `15 passed in 16.18s`. DLL SHA-256:
`FE2FD87BFA12033DDC82DF8B58619688968A7E1FF1F26CCB4EC7FF186CC447D0`.

The following native pressure micro revision (2026-09-29) polls host RAM and
commit on cache requests, suspends admission below a low margin, and releases
ready unleased tail entries until recovery or an active entry blocks it. An
explicit trim contract covers the same release and ticket protection without
consuming machine memory. Release compiled and CPU source/platform tests passed
(`15 passed in 5.04s`); the approved CUDA gate passed (`15 passed,
11 deselected in 4.86s`).
No inference benchmark accompanies it. DLL SHA-256:
`74D09DCDF8E23FFDE844C7FF7D812F3AC6D23B5CE15A067BAF524A017A025A17`.

The next integration prerequisite (2026-09-29) adds a nonblocking per-ticket
DMA completion query (cache ABI v5). It preserves the v4 metrics layout.
Release compiled and CPU source/platform contracts passed (`15 passed in
4.67s`); the approved CUDA gate passed (`15 passed, 11 deselected in 4.82s`). No model
integration or inference performance result is claimed. DLL SHA-256:
`45257BDDBFDF360308B53C5126807531C3DF0A111463E0C9051048EE1F26369D`.

An opt-in model-adapter micro integration now combines native source-cache
tickets with ring bypass and reaps completed ticket DMAs without a per-tensor
host wait. Its approved small-model numerical/cache-hit/bypass CUDA gate passed
(`2 passed, 9 deselected in 4.79s`); adapter CPU contracts passed (`4 passed in
4.52s`). The second upload uses a test-only signature reset, not physical GPU
eviction. No DDO public-runtime or inference benchmark result is claimed.

The DDO public `vmm_fast_runtime` now has an opt-in connection to this native
cache. Its default PyTorch staging route is unchanged. A 128 MiB native
ring-plus-cache cap is enforced before attach, so the usual 640 MiB model
staging is deliberately rejected for this micro-contract route. Configuration
and related CPU regressions passed (`46 passed in 4.73s`). A small public-route
CUDA numerical/accounting gate first ran with `1 passed, 1 failed, 2 deselected
in 4.70s`: the new cache route passed; the prior staging-route test required
an event in every slot although one short forward used only one slot. That
test expectation was corrected, and the approved CUDA rerun passed (`2 passed,
2 deselected in 4.67s`). No full-model
performance result follows from this integration.

Before increasing the host cap, the native cache/ring creation order was
reversed so the active cache cap and RAM/commit headroom guard the eager ring
registration. The 128 MiB micro cap is unchanged. Release compiled and
CPU-only contracts passed (`19 passed, 7 deselected in 4.59s`); the approved CUDA
cap/rollback and adapter regression gate passed (`2 passed, 22 deselected in
4.76s`). This is a safety gate, not a latency
measurement.

The next revision (cache ABI v6) keeps local retained-cache capacity at 128 MiB
and adds config v3 for an aggregate ring-plus-cache cap up to 1 GiB. Legacy
config v1/v2 keep their 128 MiB process ceiling. DDO computes 768 MiB for a
640 MiB ring plus a 128 MiB cache. Release compiled; CPU configuration/route
regressions passed (`48 passed in 4.83s`) and native CPU contracts passed
(`19 passed, 7 deselected in 4.58s`). The approved CUDA compatibility gate
passed (`17 passed, 11 deselected in 4.85s`). The approved 768 MiB host capacity
gate passed (`1 passed in 4.62s`), with 640 MiB of ring and 128 MiB of retained
cache at peak; both ledger values returned to zero on close. No new inference
result is claimed. DLL SHA-256:
`5F676304490E92AEB41E9CEED59EF0FF054B6E8B0587FFE86A4E0838185C936E`.

Public native transport can now be requested independently of retained-source
caching, and the standard CLI helper passes both settings to DDO. This keeps
cache-off/on controls on the same ring capacity. Defaults and protected routes
are unchanged. Configuration/argument/preset CPU regression passed (`54 passed
in 4.75s`). A three-layer public CUDA gate with a source alias, repeated passes,
JSON and cleanup accounting ran after approval: `2 failed, 4 passed in 4.87s`.
Both new native-ring variants failed numerical comparison; the four previous
public contracts passed. The new gate uses only a
2 MiB ring and optional 2 MiB cache, not the LTX model. Harness migration is
pending capacity/page/execution-unit parity; no new inference result is claimed.
Expanded CPU regression including profiles and bootstrap passed (`70 passed in
4.81s`), with lint passing for the updated CLI and tests. The native DLL hash
is unchanged from the preceding approved capacity gate.

Inspection identified self-reuse of a read-ticket slot before DMA submission:
each tensor occupies at least one window, so a bytes-only batch limit does not
bound the required slot count. Demand uploads now submit bounded sub-tickets,
waiting only for a recycled slot's previous DMA, and speculative oversized
tickets are rejected before I/O. No memory limit, residency policy or DLL was
changed. Post-fix CUDA validation is pending; LTX execution remains gated.
CPU window-ownership and adapter regression passed (`8 passed, 7 deselected in
4.46s`), including cache bypass and a range larger than the whole ring. New
test lint passed. The approved post-fix CUDA rerun had `1 failed, 5 passed in
4.80s`: both new variants passed numerical comparisons across three passes,
signature reuse and JSON. The cache-on variant also passed cleanup accounting.
The remaining cache-off failure was creation of the fresh ledger probe without
a current primary CUDA context after arena close. A test-only PyTorch allocation
did not reactivate it: the approved rerun remained `1 failed, 5 passed in 4.81s`
at the same probe creation. The test now explicitly retains/pushes the primary
context, then restores/releases it, preserving the zero-registration assertions.
Five CPU simulated-driver ownership/error tests passed (`5 passed in 0.03s`),
and lint passed. The approved CUDA rerun with this explicit probe setup passed:
`6 passed in 4.77s`, covering both native-ring variants and the four previous
public staging/cache/fallback/profile contracts. Registered cache/ring ledger
values were zero after close. The DLL and memory limits were unchanged. This
closes the small public-route gate, not model capacity or performance; harness
parity and LTX measurement remain separate work.

Parity audit: the recent two-step host JSON reports staging argument `0.5 GiB`,
32 MiB pages, 48 execution units and 4 GiB active residency. The current host
code uses that staging argument per slot with three slots, implying 1.5 GiB
of ring for that configuration, whereas the new public argument means total
capacity. The historical JSON omits effective ring bytes; its original actual
capacity must not be inferred as measured fact. Preserving the current host
configuration exceeds the new experimental 1 GiB aggregate ceiling once cache
is added. A cap extension or a changed benchmark manifest requires a separate
user decision and approved gates; neither was performed in this audit.

Historical source-route probes were rejected and removed. Submitting DMA as each
individual source window completed regressed to `21.3751 s` (`10.6876 s/it`),
so the protected route retains the batched submission. Mapping each complete
shard persistently was substantially worse: the two-step gate took `93.1694 s`
(`46.5847 s/it`), reached `27.004 GiB` peak RAM and `6.2522 GiB` peak VRAM,
and accumulated `92.0940 s` of source-read time. Windows paged the mapped
views into the process working set; complete-shard mappings are therefore not
a safe source strategy on this environment. Neither negative probe changes the
protected `10.5829 s/it` route.

## Executive Summary

All headline comparisons below use the same LTX 2.3 distilled image workload unless noted: BF16, real prompt encoding, `1280x704`, `8` steps, seed `43`, native attention.

| Environment / comparison | Baseline | DDO / staged result | Total latency change | Denoise change | Peak RAM change | Management readout |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Windows modular, DDO vs official Diffusers block-level group offload | `315.1s` total, `222.88s` denoise, `28.98 GB` RAM | `133.2s` total, `14.83s` denoise, `27.38 GB` RAM | `57.7%` lower total time (`2.37x` faster) | `93.3%` lower denoise time (`15.03x` faster) | `5.5%` lower peak RAM | DDO is a major win on this Windows low-VRAM setup when enough system RAM is available for pinned CPU weights. |
| Windows modular, DDO vs `diffusers-mm` auto | `375.1s` total, `211.59s` denoise, `30.21 GB` RAM | `133.2s` total, `14.83s` denoise, `27.38 GB` RAM | `64.5%` lower total time (`2.82x` faster) | `93.0%` lower denoise time (`14.27x` faster) | `9.4%` lower peak RAM | `diffusers-mm auto` selected `block_pin`, but calculated `0 pinned` and streamed all blocks; CUDA reserved grew to `9.66 GiB`, spilling past dedicated VRAM on Windows. |
| Windows old full pipeline, DDO vs official Diffusers block-level group offload | `668.0s` total, `658.44s` pipeline call, `49.62 GB` RAM, `6.97 GB` VRAM | `314.1s` total, `279.27s` pipeline call, `48.07 GB` RAM, `6.97 GB` VRAM | `53.0%` lower total time (`2.13x` faster) | `57.6%` lower pipeline-call time (`2.36x` faster) | `3.1%` lower peak RAM; equal peak VRAM | Same full `LTX2ImagePipeline` flow. DDO pays more setup time but substantially reduces the run time inside the pipeline call. |
| Windows old full pipeline vs staged traditional pipeline | `314.1s` total, `48.07 GB` RAM | `132.5s` total, `28.88 GB` RAM | `57.8%` lower total time (`2.37x` faster) | Not directly comparable because the full pipeline hides internal phases | `39.9%` lower peak RAM | Both flows use the isolated transformer loader. Staging gives explicit cleanup boundaries between prompt encoding, denoise, and VAE decode. |
| Windows staged traditional pipeline, DDO vs official Diffusers block-level group offload | `295.5s` total, `228.84s` denoise, `33.83 GB` RAM, `4.45 GB` VRAM | `132.5s` total, `32.43s` denoise, `28.88 GB` RAM, `6.97 GB` VRAM | `55.2%` lower total time (`2.23x` faster) | `85.8%` lower denoise time (`7.06x` faster) | `14.6%` lower peak RAM; VRAM `56.6%` higher | Same old-style staged flow. DDO trades more VRAM for much lower latency and lower host RAM. |
| WSL modular, DDO `wsl_compat` vs official Diffusers block-level group offload | `218.1s` total | `144.4s` total | `33.8%` lower total time (`1.51x` faster) | DDO denoise `97.80s` vs Diffusers `172.00s` | DDO peak RAM `23.47 GB` vs Diffusers `38.39 GB` | WSL benefits from DDO's dynamic path with pinning disabled, but remains slower and less stable than native Linux/Windows. |
| Native Ubuntu modular, DDO `one_shot_fast` vs best official Diffusers leaf group offload | `35.6s` total, `24.26s` denoise, `1.42 GB` RAM | `38.0s` total, `13.00s` denoise, `1.45 GB` RAM | DDO is `6.7%` slower than leaf total; it is `1.3%` faster than the block-level total (`38.5s`) | `46.4%` lower denoise time (`1.87x` faster) vs leaf; `52.1%` lower (`2.09x` faster) vs block | Similar recorded RAM | DDO edges past block-level Diffusers overall, but leaf-level Diffusers remains the fastest one-shot total because setup cost is tiny. |
| Windows modular SDNQ int4, compact zero-resident baseline vs DDO `auto` | `105.8s` total, `66.67s` denoise | `79.2s` total, `37.15s` denoise | `25.1%` lower total time (`1.34x` faster) | `44.3%` lower denoise time (`1.79x` faster) | Same recorded peak RAM (`26.19 GB`) | Experimental `sdnq_runtime` preserves SDNQ's forward and keeps `4.80 GiB` of its modules resident; it is a measured Windows win for this workload. |
| Windows modular SDNQ int8, compact zero-resident baseline vs DDO `auto` probe | `130.7s` total, `92.89s` denoise | `117.4s` total, `69.70s` denoise | `10.2%` lower total time (`1.11x` faster) | `25.0%` lower denoise time (`1.33x` faster) | Same recorded peak RAM (`26.19 GB`) | The `6 GiB` auto probe improved denoise, then sampled driver peak and saved a more conservative `4.84 GiB` steady-state recommendation. |

High-level conclusion: DDO's dense-linear streaming path is most valuable on Windows low-VRAM BF16 workloads where official Diffusers group offload is dominated by repeated transfers. The experimental SDNQ runtime adapter is a separate compatible path: it preserves SDNQ's backend-specific forward and is a measured win for the smaller LTX image workload, but it is not a general SDNQ recommendation. The LTX video probe found official Diffusers leaf offload substantially faster; retaining SDNQ blocks on CUDA made it slower. WSL also benefits from the dynamic path when pinning is disabled. On native Linux, DDO is slightly ahead of Diffusers block-level total latency while keeping a much faster denoise loop; Diffusers leaf-level still has the lowest one-shot total because its setup cost is tiny. The routing/preset layer should therefore remain platform- and workload-aware, using DDO when its denoise gain justifies setup cost and retaining official Diffusers compatibility routes for quantized backends that do not have a dedicated adapter.

### Evidence Coverage

Closed, apples-to-apples comparisons:
- Windows modular BF16: DDO `auto -> one_shot_fast` vs official Diffusers group-offload compatibility presets.
- WSL modular BF16: DDO `wsl_compat` vs official Diffusers block-level group offload.
- Native Ubuntu modular BF16: DDO `one_shot_fast` vs official Diffusers block/leaf group offload.
- Windows SDNQ int4/int8: compact zero-resident SDNQ-runtime baseline vs the experimental DDO `sdnq_runtime` auto-resident route.
- Windows old full-pipeline BF16: DDO `one_shot_fast` vs official Diffusers block-level group offload in the same full-pipeline runner.
- Windows staged traditional pipeline BF16: DDO `one_shot_fast` vs official Diffusers block-level group offload in the same staged runner.
- Windows modular BF16: DDO `one_shot_fast` vs `diffusers-mm` `auto`/`block_pin` in the same modular staged runner.

Exploratory comparisons, useful but not headline product claims:
- Old full pipeline vs old staged pipeline: shows the value of staging/cleanup boundaries, not a pure DDO-vs-Diffusers comparison.
- Simulated 32 GB RAM: useful planner behavior probe, but not a substitute for a physical 32 GB machine.
- Warm-process/server mode: useful for cache behavior, but totals include multiple generation repeats and should be reported separately from one-image latency.
- FLUX.2 Klein and Z-Image staged runners: useful multi-model probes for auto-budget behavior, but not part of the LTX benchmark claims.

Numbers still worth adding before sharing broadly:
- Optional WSL/native Ubuntu runs for `run_dynamic_old_staged_distilled.py`, only if we want old-pipeline portability claims.
- Quantized SDNQ on WSL/native Ubuntu, if quantized portability becomes part of the Diffusers discussion.
- A direct official Diffusers leaf-group-offload rerun in the new SDNQ runner, labelled independently from the DDO zero-resident baseline.

Baseline unless noted:
- Model: LTX 2.3 distilled image modular runner
- Resolution: 1280x704
- Steps: 8
- Seed: 43
- Dtype: bfloat16
- Prompt mode: fake prompt for transformer-only comparisons
- Attention backend: native
- Windows standby purge before transformer enabled for Windows tests

## Windows Transformer Comparisons

| Scenario | Preset / route | RAM policy | Setup seconds | Denoise seconds | Copy seconds | Peak VRAM | Peak RAM | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dynamic offload, enough RAM | `one_shot_fast` | detected available RAM 46.84 GB, full pin selected | 34.67 | 14.38 | 0.56 | 6.97 GB | 25.15 GB | Correct fast path: `snap_to_full_pin`, 444 patched modules, 11 resident blocks. |
| Dynamic offload, simulated 32 GB RAM | `one_shot_fast` | pin skipped, insufficient RAM | 7.72 | 198.2 total pass, denoise dominated by copies | 174.12 | 6.97 GB | 25.87 GB | Correct safe path: avoids pinning when usable RAM is below required RAM. |
| Low RAM fallback | `low_ram_safe` | no pin, smaller resident budget | 3.78 | 194.10 | 187.81 | 3.28 GB reserved during denoise | not captured in pasted slice | Lower VRAM, but slower than `one_shot_fast` simulated 32 GB because fewer resident modules and more runtime copies. |
| Diffusers group offload | `off` + transformer `leaf_level`, stream, record stream, low CPU mem usage off | official Diffusers group offload | 0.07 group setup | 315.29 | not tracked by dynamic offload | 6.49 GB | 26.60 GB | Very compatible/official path, but much slower in this low-VRAM 8-step transformer scenario. |
| Diffusers group offload | `off` + transformer `leaf_level`, stream, record stream, low CPU mem usage on | official Diffusers group offload | 0.05 group setup | 320.10 | not tracked by dynamic offload | 6.56 GB | 26.60 GB | Same shape as low CPU off; setup is tiny, but denoise remains copy/offload bound. |
| Diffusers group offload | `off` + transformer `leaf_level`, no stream, no record stream, low CPU mem usage on | official Diffusers group offload | 0.03 group setup | 507.83 | not tracked by dynamic offload | 6.58 GB | 39.37 GB | Worst official offload probe so far; disabling stream made denoise much slower and increased peak RAM. |
| Diffusers group offload | `off` + transformer `block_level`, `num_blocks_per_group=1`, stream, record stream, low CPU mem usage on | official Diffusers group offload | 0.02 group setup | 265.47 | not tracked by dynamic offload | 6.49 GB | 26.79 GB | Better than leaf, but still far from dynamic offload fast path. `torch_alloc` stayed near 0.03 GiB while reserved grew to 2.11 GiB. |
| Diffusers group offload | `off` + transformer `block_level`, `num_blocks_per_group=1`, stream, no record stream, low CPU mem usage on | official Diffusers group offload | 0.01 group setup | 312.03 | not tracked by dynamic offload | 6.51 GB | 26.78 GB | Worse than record stream on; reserved VRAM was lower, but denoise returned to leaf-level timing. |

## diffusers-mm Comparison

These runs used the same modular staged LTX 2.3 distilled workload as the Windows headline benchmark: BF16, real prompt encoding, `1280x704`, `8` steps, seed `43`, native attention. The runner applied `diffusers-mm` through `ModelManager` on the staged components so it had the same cleanup boundaries as the DDO modular runner.

| Platform | Runner / strategy | Applied strategy | Setup | Denoise | Total | Peak VRAM | Peak RAM | Readout |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Windows | `run_diffusers_mm_modular_distilled.py`, transformer `auto`, text encoder `group_offload` | `auto -> block_pin` | transformer setup `81.8440s` | `211.5898s` | `375.1s` | `6.97 GB` | `30.21 GB` | `diffusers-mm` reported no VRAM budget for pinning (`0 pinned, all blocks stream`). `torch_reserved` rose from `5.66 GiB` on step 1 to `9.66 GiB` from step 2 onward, which matches the Windows shared-memory slow path. |
| Windows | `run_diffusers_mm_modular_distilled.py`, transformer `block_pin`, text encoder `group_offload` | `block_pin` with `0 pinned` | transformer setup `0.5608s` | `401.3015s` | `475.9s` | `6.97 GB` | `6.49 GB` | Forcing `block_pin` did not help because the planner still found no usable pin budget. It streamed all blocks and kept the same `9.66 GiB` reserved pattern, with worse denoise latency. |

Interpretation: `diffusers-mm` has budget/headroom logic, but on this Windows 8 GB dedicated VRAM workload it concludes that the denoise working set does not fit and falls back to streaming all blocks. That fallback remains much slower than DDO's dense-linear dynamic path for this model. The optional Windows standby purge probe repeated the same reserved-memory pattern, so it is not reported as a benchmark row; it only ruled out standby cache as the cause.
## Current Interpretation

`one_shot_fast` is the default general preset. It pins CPU weights only when the measured or supplied available RAM can cover the model weight copy, resident GPU modules, and configured system headroom. On machines with enough RAM, it pays setup time once and keeps denoise fast. On constrained RAM, it chooses safety over speed.

The resident-module budget must be VRAM-headroom aware, not just model-size aware. FLUX.2 Klein showed that using more resident VRAM is not always faster: a manually larger budget could stay under nominal VRAM capacity while still pushing the CUDA allocator/driver into a slow path. On the tested 8 GB Windows setup, about `5 GB` resident budget gave fast denoise, while larger budgets increased `torch_reserved` near the card limit and slowed the loop. The planner should therefore leave accelerator headroom and treat `max_resident_module_budget_gb` as an optional cap, not as the default budget.

Z-Image validated the same generic policy on a different, locally patched Diffusers-style pipeline. DDO `one_shot_fast` used more VRAM than Diffusers leaf offload but reduced denoise time from `101.83s` to `18.31s` at `768x768`, `9` steps. Its total was dominated by the custom pipeline build stage, so this is evidence for the offload policy, not a report-ready pipeline benchmark.

`low_ram_safe` is not the recommended 32 GB preset for this model. It is a fallback for tighter VRAM cases where the user accepts slow denoise to reduce accelerator memory pressure.

The Diffusers group offload path is useful as an official compatibility baseline, but in this test it is not close to
the dynamic offload fast path. `block_level` with one block per group and `record_stream=1` was the best official
offload probe, but it still remained dominated by repeated offload movement. `leaf_level` without stream is not viable
for this model/shape.

## Preset Direction

Recommended defaults:
- `auto` -> `one_shot_fast`
- `one_shot_fast` -> balanced planner, RAM-aware pinning, VRAM-headroom-aware resident budget
- `low_ram_safe` -> explicit fallback only
- `wsl_compat` -> explicit WSL fallback when stream/pin behavior is unstable
- `warm_process` -> process-lifetime cache comparisons and server-like usage
- `diffusers_offload_compat` -> official Diffusers group-offload compatibility fallback
- `diffusers_leaf_offload_compat` -> official Diffusers leaf-level fallback for models where finer granularity may fit better

Inspect the current preset contract without loading model weights:

```powershell
$env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS="1"
python run_dynamic_modular_distilled.py
Remove-Item Env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS -ErrorAction SilentlyContinue
```

Closed report sections:
- one-shot setup cost vs denoise throughput
- RAM-aware pinning behavior
- Windows standby cache impact
- Diffusers official group offload baseline
- Windows, WSL, and native Ubuntu validation matrix
- Quantized model compatibility risks and follow-up results


## Platform Validation Runs

These runs use real prompt encoding unless noted. Keep width `1280`, height `704`, steps `8`, seed `43`, BF16, native attention, and generation repeats `1` fixed across platforms.

| Platform | Preset / route | Prompt | Platform cleanup | Encode pass | Transformer setup | Denoise | Pass 1 | VAE | Total | Peak VRAM | Peak RAM | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Windows | `auto -> one_shot_fast` | real | standby purge before run, after text encoder, before transformer | `72.0s` | `29.76s` DDO setup, `21.47s` pin, `7.80s` resident modules | `14.83s` | `52.6s` | `4.9s` | `133.2s` | `6.45 GB` | `27.38 GB` | Run 1 after package split. Planner detected `43.53 GB` available RAM and selected `snap_to_full_pin` for `18.5181 GB`; runtime copy `0.3153s / 148.1445 GB`. |
| Windows | `auto -> one_shot_fast` | real, simulated 32 GB RAM | standby purge before run, after text encoder, before transformer | `79.6s` | `8.43s` DDO setup, pin skipped, `7.93s` resident modules | `114.70s` | `132.2s` | `2.6s` | `217.2s` | `6.41 GB` | `28.22 GB` | RAM-aware safety path. Planner used `available_system_ram_gb=32.0`, usable `24.0 GB`, required `30.0234 GB`, selected `skip_insufficient_ram`; runtime copy rose to `110.1362s / 148.1445 GB`. |
| Windows | `diffusers_offload_compat` | real | standby purge before run, after text encoder, before transformer | `76.5s` | `0.066s` Diffusers group-offload setup | `222.88s` | `232.9s` | `2.5s` | `315.1s` | `6.09 GB` | `28.98 GB` | Official Diffusers `block_level`, `num_blocks_per_group=1`, stream and record stream. First denoise step `83.68s`, later steps around `19-20s`; much slower than DDO pinned fast path but compatible baseline. |
| Windows | `diffusers_leaf_offload_compat` | real | standby purge before run, after text encoder, before transformer | `77.9s` | `0.030s` Diffusers group-offload setup | `233.39s` | `242.7s` | `2.7s` | `326.4s` | `6.21 GB` | `28.79 GB` | Official Diffusers `leaf_level`, stream and record stream. First denoise step `86.75s`, later steps around `20-21s`; slightly slower than block-level fallback in this Windows run. |
| Windows | `low_ram_safe` | real | standby purge before run, after text encoder, before transformer | `81.4s` | `4.00s` DDO setup, no pin, `3.36s` resident modules | `136.49s` | `149.6s` | `2.5s` | `236.5s` | `6.11 GB` | `28.22 GB` | Explicit VRAM-pressure fallback. Denoise torch allocation stayed near `2.96 GiB`, but more modules were patched (`516`) and runtime copy rose to `132.5071s / 172.1680 GB`; slower than simulated-32GB `auto` for this model. |
| WSL Ubuntu | `low_ram_safe` | real | no Windows standby purge | `36.2s` | `1.77s` DDO setup, no pin, `3 GB` resident budget | `122.88s` | `131.6s` | `2.0s` | `170.4s` | `5.99 GB` | `23.56 GB` | Corrected WSL low-RAM route: text encoder now uses Diffusers `leaf_level` group offload with stream disabled, avoiding the earlier `device not ready` connector failure. Transformer stays low VRAM around `2.96 GiB` allocated, but denoise is copy-bound without pinned CPU weights. |
| WSL Ubuntu | `wsl_compat` | real | no Windows standby purge | `36.2s` | `3.66s` DDO setup, no pin, `6 GB` resident budget | `97.80s` | `105.8s` | `1.8s` | `144.4s` | `6.30 GB` | `23.47 GB` | Stable WSL compatibility run after text-encoder routing fix. Text encoder uses non-streaming Diffusers group offload; transformer uses DDO dynamic offload with pin disabled. Denoise steps were consistent around `11.8-12.5s`, slower than Windows full-pin but faster than `low_ram_safe` and Diffusers group offload on WSL. |
| WSL Ubuntu | `diffusers_offload_compat` | real | no Windows standby purge | `35.8s` | `0.012s` Diffusers group-offload setup | `172.00s` | `180.1s` | `1.8s` | `218.1s` | `6.10 GB` | `38.39 GB` | Official Diffusers `block_level` group offload baseline on WSL. First denoise step `39.36s`, later steps around `18-19s`; faster than the Windows Diffusers block-level run but still slower than DDO WSL dynamic-offload fallbacks for this model/shape. |
| WSL Ubuntu | `diffusers_leaf_offload_compat` | real | no Windows standby purge | not completed in the WSL run | `0.011s` Diffusers group-offload setup | not completed | not completed | not completed | not completed | not captured | not captured | WSL-only result: official Diffusers `leaf_level` group offload loaded and prepared, then the WSL process exited/crashed immediately after `Starting denoise loop` before the first denoise callback. This is separate from the completed Native Ubuntu leaf run below. |
| Native Ubuntu | `auto -> one_shot_fast` | real | none | `10.6s` | `11.92s` DDO setup, full-pin fast path | `13.00s` | `26.5s` | `0.6s` | `38.0s` | `6.27 GB` | `1.45 GB` | Native Linux DDO execution. Denoise steps stayed around `1.57s`; transformer setup is still the main remaining cost, but total slightly beats the block-level Diffusers execution. |
| Native Ubuntu | `low_ram_safe` | real | none | `32.3s` | `2.61s` DDO setup, no pin, `3 GB` resident budget | `37.47s` | `48.4s` | `1.6s` | `82.7s` | `5.98 GB` | `23.45 GB` | Low-RAM fallback completed. First step paid `20.16s`; later steps were around `2.45s`, giving a much better Linux low-RAM result than WSL. |
| Native Ubuntu | `diffusers_offload_compat` | real | none | `8.4s` | `0.015s` Diffusers group-offload setup | `27.14s` | `29.3s` | `0.6s` | `38.5s` | `5.96 GB` | `1.53 GB` | Native Linux official Diffusers `block_level` group-offload execution. Its setup remains almost free, but the denoise loop is about `2.1x` slower than DDO and its total is slightly slower than DDO. |
| Native Ubuntu | `diffusers_leaf_offload_compat` | real | none | `8.4s` | `0.025s` Diffusers group-offload setup | `24.26s` | `26.3s` | `0.6s` | `35.6s` | `5.96 GB` | `1.42 GB` | Fastest native Ubuntu run so far. Official Diffusers `leaf_level` completed cleanly and still outperformed block-level and DDO for one-shot total latency on this platform run. |
| Native Ubuntu | `auto -> one_shot_fast`, simulated `32 GB` RAM | real | none | `7.8s` | `0.73s` DDO setup | `18.38s` | `21.4s` | `0.6s` | `30.0s` | `6.24 GB` | `1.48 GB` | Simulated 32 GB RAM decision on native Linux. This run was unusually fast and likely benefited from OS/file/cache warmth after earlier tests; keep as warm/native Linux data, not as cold one-shot baseline. |
| Native Ubuntu | `warm_process` | real | none | `19.7s` | prepare 1 `20.04s`, prepare 2 `0.90s` after cache | repeat 1 `12.98s`, repeat 2 `12.89s` | `52.4s` | `1.5s` | `73.9s` | `6.27 GB` | `1.43 GB` | Corrected warm-process route completed. Text encoder uses Diffusers group offload; transformer dynamic offload benefits from process-lifetime cache on the second prepare. Total includes two generation repeats, so this is a server/warm comparison rather than one-image latency. |


## Quantized SDNQ Validation

These Windows runs use the LTX 2.3 image SDNQ transformer path, `1280x704`, `8` steps, real prompt encoding, and Diffusers leaf group offload for the text encoder. `sdnq_runtime` is deliberately not DDO's dense-linear path: it retains each SDNQ layer's native forward (including its dequantization and `F.linear` behavior) and only stages whole SDNQ modules around that forward.

### Historical compatibility baseline

| Platform | Preset / route | Transformer behavior | Encode pass | Transformer setup/load | Denoise | Pass 1 | VAE | Total | Peak VRAM | Peak RAM | Denoise allocation | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Windows | `off` | No transformer offload; transformer moved to CUDA/shared GPU memory | `32.3s` | `load_transformer_to_cuda 17.86s` | `189.00s` | `214.9s` | `1.9s` | `250.8s` | `6.97 GB` | `21.46 GB` | `torch_alloc 12.63 GiB`, `reserved 13.10 GiB` | Lower RAM than BF16, but denoise spills beyond 8 GB dedicated VRAM into shared GPU memory and remains slow/unstable. |
| Windows | `diffusers_leaf_offload_compat` | Official Diffusers leaf group offload with stream and record stream | `33.3s` | `setup_transformer_group_offload 0.038s` | `92.82s` | `101.3s` | `1.9s` | `138.5s` | `6.11 GB` | `16.80 GB` | `torch_alloc 0.15 GiB`, `reserved 0.96 GiB` | Best SDNQ memory/latency result in this pair. For quantized models, prefer Diffusers group-offload compatibility presets. |

### Experimental DDO SDNQ runtime adapter

The following pairs were run in the dedicated SDNQ DDO runner. Although the first console label used "Diffusers Leaf Offload Compact", both pasted logs identify `offload=ddo_sdnq_runtime` and call `setup_transformer_sdnq_dynamic_offload`; it is therefore recorded here as the compact **zero-resident SDNQ-runtime baseline**, not as an official Diffusers leaf result.

| Quantization | Route | Resident budget | DDO setup | Denoise | Pass 1 | Total | Peak VRAM | Peak RAM | Readout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| int8 | Compact zero-resident baseline | `0 GiB` | `0.30s` | `92.89s` | `100.1s` | `130.7s` | `6.08 GB` | `26.19 GB` | Steps 2-8 were roughly `10.9-11.3s`; it streams SDNQ weights every layer execution. |
| int8 | `auto` validation probe | `6.00 GiB` probe; saved `4.84 GiB` | `5.20s` | `69.70s` | `82.2s` | `117.4s` | `6.97 GB` | `26.19 GB` | `25.0%` less denoise and `10.2%` less total time than zero-resident. The driver sampler observed `8.16 GiB`, so the profile retained the 1 GiB headroom by lowering the next-run recommendation to `4.84 GiB`. |
| int4 | Compact zero-resident baseline | `0 GiB` | `0.53s` | `66.67s` | `74.5s` | `105.8s` | `6.14 GB` | `26.19 GB` | Steps 2-8 were roughly `7.0-7.3s`. |
| int4 | `auto` steady profile | `4.80 GiB` | `4.18s` | `37.15s` | `48.4s` | `79.2s` | `6.97 GB` | `26.19 GB` | `44.3%` less denoise and `25.1%` less total time than zero-resident; steps 2-8 were roughly `4.3-4.6s`. |

Interpretation: the dedicated adapter keeps SDNQ's own forward rather than trying to replace its packed/quantized linear implementation. On this RTX 3060 Ti Windows workload, increasing whole-module residency is beneficial, especially at int4 where the `4.80 GiB` auto profile is the current best tested result. Int8 also improves, but the reported `6 GiB` run is a calibration probe; evaluate the saved `4.84 GiB` profile as the steady-state int8 result. This evidence updates the SDNQ recommendation from "compatibility only" to "experimental dedicated DDO adapter is a supported, measured option for SDNQ"; it does not generalize that claim to other quantization backends.

### LTX video negative probe: do not generalize image residency

On the Windows RTX 3060 Ti LTX 2.3 video int4 workload (`1280x704`, 5 s, 8 steps), official Diffusers leaf offload completed at about `26 s/it` (`213.9 s` denoise). The experimental native-leaf hybrid that kept SDNQ transformer blocks resident on CUDA was rejected and removed: a `1.00 GiB` request retained `0.85 GiB` across four blocks and slowed to `44.35 s/it` (`357.3 s` denoise); even one contiguous `0.22 GiB` block was already about `54.6 s/it` by step 3. SDNQ reconstructs BF16 weights for each forward, so residency does not remove its dequantization cost and instead consumes the VRAM headroom needed by the leaf streaming path and its activations.

Recommendation: `sdnq_runtime` remains an experimental, image-validated option for smaller workloads where its profile has demonstrated a gain. It is not recommended for LTX video from this evidence; use `diffusers_leaf_offload_compat` there. Other quantized backends remain compatibility-route candidates unless independently validated.

The logs currently report `CPU pinned parameters: 0.00 GiB (0 tensors)` for both routes. This is not evidence that SDNQ tensors were CPU-pinned: the adapter's parameter scan did not find SDNQ weight tensors to pin in this model representation (they may be held as buffers or other backend state). Residency alone accounts for the measured gain. Pinning support for SDNQ-held non-parameter tensors remains an investigation item, not a claimed optimization.

## Quantized Follow-Up

SDNQ int4/int8 now have a Windows dedicated-adapter baseline. The next quantized work should first run the saved int8 `4.84 GiB` steady profile and a direct official Diffusers leaf baseline in the same dedicated runner. WSL/native Ubuntu runs are useful only if quantized portability becomes part of the Diffusers discussion; other backends such as bitsandbytes, TorchAO, or Quanto remain compatibility-route candidates until they receive their own validated adapter.

For quantized backends, fixes should remain model-agnostic and preserve backend-specific forward logic. DDO dynamic linear streaming should only patch standard dense `nn.Linear` weights with the expected 2D `(out_features, in_features)` layout; a dedicated adapter may instead stage backend-owned tensors without replacing its forward.

## Validation Matrix

Use these as the next closed test ladder. Keep Windows as the development gate; repeat WSL/Ubuntu only after a Windows
change produces a useful result.

Windows baseline:

```powershell
$env:DDO_PRESET="auto"
$env:DDO_RUNNER_WIDTH="1280"
$env:DDO_RUNNER_HEIGHT="704"
$env:DDO_RUNNER_STEPS="8"
$env:DDO_RUNNER_SEED="43"
$env:DDO_RUNNER_GENERATION_REPEATS="1"
$env:DDO_RUNNER_FAKE_PROMPT="0"
$env:DDO_RUNNER_METRICS_LEVEL="1"
$env:DDO_SHOW_PROFILE="0"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_RUN="1"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER="1"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER="1"
python run_dynamic_modular_distilled.py
```

Windows RAM-constrained planner check:

```powershell
$env:DDO_AVAILABLE_SYSTEM_RAM_GB="32"
python run_dynamic_modular_distilled.py
Remove-Item Env:DDO_AVAILABLE_SYSTEM_RAM_GB -ErrorAction SilentlyContinue
```

Linux/Ubuntu baseline:

```bash
export DDO_PRESET=auto
export DDO_RUNNER_WIDTH=1280
export DDO_RUNNER_HEIGHT=704
export DDO_RUNNER_STEPS=8
export DDO_RUNNER_SEED=43
export DDO_RUNNER_GENERATION_REPEATS=1
export DDO_RUNNER_FAKE_PROMPT=0
export DDO_RUNNER_METRICS_LEVEL=1
export DDO_SHOW_PROFILE=0
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_RUN
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER
python run_dynamic_modular_distilled.py
```

WSL compatibility fallback:

```bash
export DDO_PRESET=wsl_compat
export DDO_RUNNER_WIDTH=1280
export DDO_RUNNER_HEIGHT=704
export DDO_RUNNER_STEPS=8
export DDO_RUNNER_SEED=43
export DDO_RUNNER_GENERATION_REPEATS=1
export DDO_RUNNER_FAKE_PROMPT=0
export DDO_RUNNER_METRICS_LEVEL=1
export DDO_SHOW_PROFILE=0
python run_dynamic_modular_distilled.py
```

Official Diffusers fallback comparison:

```powershell
$env:DDO_PRESET="diffusers_offload_compat"
python run_dynamic_modular_distilled.py
```
## Traditional Diffusers Pipeline Probes

These probes compare DDO in non-modular LTX2ImagePipeline flows. The full-pipeline variant validates compatibility with old-style Diffusers pipelines, but it cannot insert cleanup between internal text-encoder, transformer, and VAE phases.

| Platform | Runner | Preset / route | Component policy | Load / setup | Pipeline call / denoise | Total | Peak VRAM | Peak RAM | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Windows | `run_dynamic_old_distilled.py` | `auto -> one_shot_fast` | Full `LTX2ImagePipeline`; text encoder, connectors, and VAE use Diffusers leaf group offload; transformer uses DDO dynamic offload | `26.5s` load/setup; transformer dynamic setup `22.1146s` | `279.2730s` pipeline call; first denoise callback `154.02s`, later steps around `12.6-13.7s` | `314.1s` | `6.97 GB` | `48.07 GB` | Uses `from_pretrained_with_dynamic_offload(..., apply_dynamic=False)` to load the transformer before pipeline construction. The full pipeline cannot purge/flush between prompt encoding, transformer, and VAE decode. |
| Windows | `run_dynamic_old_distilled.py` | `diffusers_offload_compat` | Full `LTX2ImagePipeline`; text encoder, connectors, transformer, and VAE use official Diffusers group offload | `4.0s` load/setup; transformer group-offload setup `0.0176s` | `658.4367s` pipeline call; first denoise callback `297.84s`, later steps around `35.9-39.4s` | `668.0s` | `6.97 GB` | `49.62 GB` | Same full-pipeline construction and isolated transformer loader as the DDO run. Group offload has a much smaller setup cost but is transfer-bound during the pipeline call. |
| Windows | `run_dynamic_old_staged_distilled.py` | `auto -> one_shot_fast` | Staged traditional pipeline; text encoder uses Diffusers leaf group offload, connectors use Diffusers leaf group offload inside the denoise pipeline, transformer uses DDO dynamic offload, VAE decoded in a separate manual stage | Pass 0 `55.6s`; transformer dynamic setup `23.6657s`; Pass 1 setup/build/cleanup included `61.5s` | `32.4312s` denoise pipe call; first callback `18.01s`, later steps around `1.7-2.2s` | `132.5s` | `6.97 GB` | `28.88 GB` | Controlled-memory comparison with cleanup boundaries between phases. Still slower than the best modular run because connectors execute inside the denoise pipeline and old-pipeline overhead remains. |
| Windows | `run_dynamic_old_staged_distilled.py` | `diffusers_offload_compat` | Staged traditional pipeline; text encoder, connectors, and transformer use official Diffusers group offload | Pass 0 `53.9s`; transformer group-offload setup `0.0225s`; Pass 1 `231.7s` | `228.8404s` denoise pipe call; first callback `60.64s`, later steps around `23.5-24.9s` | `295.5s` | `4.45 GB` | `33.83 GB` | Same staged flow as the DDO run. It uses less VRAM than DDO staged, but is `2.23x` slower overall, `7.06x` slower in denoise, and uses more host RAM. |
## Native file-window integration — 2026-09-29 (not a performance result)

### Approved integrated capacity and LTX gate — 2026-09-30

### Cache-aware prefetch validation — 2026-09-30

New preparation: full-unit GPU look-ahead is implemented in the submodule
adapter, including inherited pins, skipped-unit cancellation and upload-error
cleanup. **90 CPU tests passed, 15 deselected in 6.51 s**; test lint passed.
The approved CUDA gate passed **32 tests, 5 deselected in 5.98 s**, including
the new public GPU look-ahead contracts and protected adapter regressions.
No budget, DLL, source-cache admission
or protected runtime changed. Adapter SHA256:
`30D533F1C342436B8148164649287DFAADB7483F37010EA348B561BAE1998E2A`.
The proposed consolidated gate and exact commands are recorded in
`docs/native-source-cache-plan.md`, under full-unit GPU look-ahead correction.
Earlier inference timings do not measure this change.

Approved full-unit GPU look-ahead A/B, DDO `.venv-vmm`, same fixed workload:
BF16, 1280x704, seed 43, fake prompt, no PAG, local files, 4 steps, one
generation/prepare, 4 GiB arena, 1.5 GiB total staging and 0/512 MiB source
cache. DLL SHA256 remains
`8029EF09D9D0F83F8B3D5ADF4E766FC3131BAAE105BD6B6BDF457A69DA520682`.

| Source cache GiB | Load s | DDO setup s | Denoise s | Steps 2–4 mean s | Total s | Peak VRAM/RAM GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.3203 | 1.1658 | 34.4810 | 8.4744 | 41.4431 | 6.3017 / 2.7360 |
| 0.5 | 0.3767 | 1.3526 | 39.2374 | 9.5619 | 46.5399 | 6.2059 / 3.2379 |

Step times: cache off `[9.0575, 8.4626, 8.4138, 8.5468]`; cache on
`[10.5514, 9.6601, 9.3997, 9.6259]`. Both exited 0, with 188 GPU-prefetched
units and 188 consumers, zero native-capacity skips/cancellations, zero
fault/read failures, temporary fallbacks and execution-order mismatches.
Both images preserve SHA256
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.

H2D remains **93,927,995,136 bytes** in either case. Cache off read that same
amount through the ring; cache on read **93,430,144,752 ring bytes** plus
**536,870,912 source-fill bytes**, with **23 cache hits** and
**497,850,384 cache DMA bytes**. Native arena counters show 2,663 evicted
pages and 360 signature hits in the cache-off case; do not interpret the
legacy generic `evictions=0` field as no native page recycling.

These runs do not demonstrate a speedup: the preceding pair measured
31.8246/35.9451 s denoise. Full GPU look-ahead is functionally exercised, but
unchanged repeated transfer volume and low retained-source coverage remain.
Each case ran once; standby purge was skipped because the token lacks the
required privilege, so OS cache/environment variation prevents a causal
before/after timing attribution. No further run, capacity increase or tuning
was performed. Prior JSON/images remain in sibling output folder
`before_gpu_lookahead_20260930`; current case names and output directory are
unchanged. The performance target is not achieved.

Later preparation, not part of these measured results: native read-only
residency/signature observation to avoid unnecessary speculative reads,
with consumer fault/recheck retained and no residency-policy change.
Release rebuild succeeded; **104 CPU tests passed in 5.15 s**. New DLL
SHA256 `8029EF09D9D0F83F8B3D5ADF4E766FC3131BAAE105BD6B6BDF457A69DA520682`;
The approved CUDA gate passed: **23 tests in 4.92 s**, including read-only
signature observation, unchanged counters and eviction/generation recovery.
It used the DDO `.venv-vmm` and the rebuilt DLL above. Earlier CUDA and LTX numbers below refer to
the preceding DLL/code, not this newly built version. Source-cache admission
usefulness remains unresolved; no new LTX measurement was made here.

Approved consolidated CUDA battery: **22 passed in 5.56 s**, using
`test_source_cache.py`, `test_vmm_native_transport.py`, `test_vmm_fast_runtime.py`,
`-m vmm -q -x -p no:cacheprovider --basetemp .pytest-cache-prefetch-cuda`,
DDO `.venv-vmm`, unchanged ABI 7 DLL. Followed by the approved 4-step cache-off/on
repeat using the same benchmark commands and manifest documented below.
Previous 4-step JSONs/PNGs were preserved in the sibling host's
`outputs/ltx_image_modular_vmm/before_cache_prefetch_20260930/` before repetition.

| Source cache (GiB) | Load (s) | Setup (s) | Denoise (s) | Mean steps 2+ (s/it) | Total (s) | Peak VRAM (GiB) | Peak process RAM (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3550 | 1.4936 | 31.8246 | 7.1973 | 41.2486 | 6.1151 | 2.7352 |
| 0.5 | 0.3432 | 1.2810 | 35.9451 | 8.7490 | 43.1526 | 6.1371 | 3.2391 |

Both cases completed without fault/read failures, temporary fallbacks or
execution-plan mismatches. Output SHA256 remains
`87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0` in
both cases, matching the earlier four-step outputs. Cache memory remains
512 MiB retained/registered + 1536 MiB staging, zero live cache tickets at
snapshot, zero native admission denials or pressure evictions.

Scheduling contract achieved: cache-on now reports **346 prefetch hits,
0 skips, 188 attempts**, equal to cache-off. Partial cache still has 23 source
hits and 497,850,384 cache-DMA bytes, unchanged from the earlier run. Its bypass
count is now 5594, ring source reads 93,933,584,112 bytes, cache fills
536,870,912 bytes. These source-operation bytes include speculative reads;
they are not measurements of physical disk traffic. Staging waits:
345 cache-on versus 174 cache-off. Summed read-ticket times are not wall time.
H2D payload remains **93,927,995,136 bytes** in either case.

Performance contract still **not achieved**: cache-on is 12.95% slower than
this round's control. The control itself shifted from 27.6858 to 31.8246 s;
one sample per condition and unequal OS cache state do not establish a
causal before/after speedup. Restored prefetch alone does not solve useful
retention, repeated transfer volume or the final target. Native admission
usefulness and speculation that becomes unnecessary after signature checks
need investigation; no higher-capacity or 8-step run was performed.

The preparation and initial gate notes below are historical. The current
cache-aware adapter is validated by the 22-test battery and this A/B, not
by the older 21-test battery alone.

Follow-up preparation after the performance regression: cache-aware speculative
read tickets added in the native adapter, with distinct read/DMA ownership,
cache-first admission, bounded bypass reads and cancellation. CPU suite:
**103 passed, 7 deselected in 6.03 s**. CUDA and benchmark follow-up are pending;
the measurements below belong to the earlier cache-prefetch-disabled code.

Capacity contract `test_model_windows_1536_mib_ring_plus_512_mib_cache`:
**1 passed in 7.72 s**. Validated lazy retention of 512 MiB, second-pass
source reuse, rejection above the 2 GiB aggregate host-registration ceiling
and zero ledger after close. Separate test file: 512 MiB. No large-model host
cache allocation was approved or performed.

Four subsequent public-runtime harness cases completed with exit code zero,
using DDO `.venv-vmm` and the ABI 7 DLL recorded below. Manifest: BF16,
1280x704, seed 43, fake prompt, no PAG, one generation/preparation,
4 GiB active residency, 32 MiB arena pages, 1.5 GiB total native staging,
local-only model `elismasilva/ltx2.3-image-distilled-1.1`. Loader: AutoModel,
resolved local transformer class; bootstrap/local module-cache route preserved.
Standby purge was requested by existing settings but skipped because the
process token lacks the required privilege. File-read counters do not identify
physical disk traffic versus OS-cached reads. Each case ran once, in table order;
OS cache state was not equalized. No statistical speedup claim is supported.

| Steps | Source cache (GiB) | Transformer load (s) | DDO setup (s) | Denoise (s) | Mean steps 2+ (s/it) | Total (s) | Peak VRAM (GiB) | Peak process RAM (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 0.4419 | 1.4700 | 15.9284 | 6.4916 | 25.2002 | 6.3203 | 2.7346 |
| 2 | 0.5 | 0.4222 | 1.3367 | 15.9116 | 7.3343 | 23.0282 | 6.2355 | 3.2385 |
| 4 | 0 | 0.3053 | 1.0021 | 27.6858 | 6.8056 | 33.8766 | 6.1073 | 2.7354 |
| 4 | 0.5 | 0.4623 | 1.4013 | 33.2483 | 8.0562 | 40.4192 | 6.1149 | 3.2386 |

No fault/read failures, execution-plan mismatches or temporary fallbacks were
reported in any case. Cache-on snapshots show exactly 512 MiB cache +
1536 MiB staging registered, zero admission denials, zero pressure evictions
and zero live source tickets at the pre-cleanup snapshot. Runtime close
completed; the separate capacity contract verifies the post-close ledger.
Cache-off/on PNG SHA256 values match within each step count:
2 steps `EC7A1EAEFDBFCD312E988C0D1BAFA9631039EC26E80D7756DBA3CB98223F3521`;
4 steps `87337014D76A937A3B0C9071F978970FB2FC53559F3AD7A3940D19F5788C32D0`.
This is output parity for the tested workload, not exhaustive model correctness.

Performance gate **not achieved**. In the 4-step case cache-on denoise was
20.09% slower. The code explicitly disables inter-unit speculative prefetch
when the source cache is active: 0 hits/188 skips versus 346 hits/0 skips
without cache. This is a confirmed scheduling difference, not proof that it
alone accounts for the timing gap. Read-first batching within a unit does
not close inter-unit read/DMA/compute overlap.

The partial cache reports 23 hits and 497,850,384 DMA bytes at both 2 and 4
steps; no additional retained-source reuse is observed in the longer run.
At 4 steps there are 5549 bypasses and 93,430,144,752 bytes read through the
ring, plus 536,870,912 window-fill bytes. H2D payload remains
93,927,995,136 bytes with or without cache. Retention alone has not removed
the large per-step transfer volume. Investigate useful native admission and
cache-aware prefetch lifetimes before increasing capacity. No 8-step,
real-prompt, text-encoder expansion or higher-capacity gate was run.

Commands (each benchmark substitution was separately part of the approved
four-case sequence: steps 2/4, source cache 0/0.5):

```powershell
$env:DDO_VMM_CAPACITY_GATE='1'
& 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe' -m pytest ddo-vmm-allocator/tests/test_source_capacity.py::test_model_windows_1536_mib_ring_plus_512_mib_cache -q -x -p no:cacheprovider --basetemp .pytest-model-window-capacity
```

From the sibling host directory, with DDO source/native-library paths:

```powershell
& 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe' run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 0.5 --width 1280 --height 704 --seed 43 --steps 4 --fake-prompt --no-pag --local-files-only --metrics-level 2 --show-ddo-profile
```

Host JSON/PNG artifacts use suffixes
`steps{2,4}_seed43_ddo_public_cache{0,0.5}gb` under
`outputs/ltx_image_modular_vmm` (JSON in `metrics/`). The following preparation
notes are historical; the results above supersede their pending-run status.

2026-09-30 preparation: sibling public-harness opt-in added, with explicit
diagnostic-to-public staging conversion (0.5 GiB per diagnostic slot ->
1.5 GiB total), block pattern/page/DXGI configuration parity and distinct
cache-off/on output names. CPU configuration suite: **31 passed in 4.54 s**.
New 2 GiB aggregate host-registration contract is prepared but unexecuted.
Neither the modified harness nor the larger capacity has been run on CUDA;
the passing 21-test gate below predates this harness integration.

Approved consolidated CUDA attempt used `test_source_cache.py`,
`test_vmm_native_transport.py`, and `test_vmm_fast_runtime.py`, `-m vmm -x`,
DDO `.venv-vmm` and the ABI 7 Release DLL recorded below. Result:
**1 passed, 1 failed in 4.70 s**, stopping at the first failure. The new
window/neighbor/EOF test passed its DMA data comparisons but expected two
misses after explicitly issuing a third, cross-window bypass request. Native
miss accounting includes that rejected request. The test expectation was
corrected to three misses, one hit and one bypass; no runtime/DLL change.
Approved repeat, same DLL and limits, with the corrected expectation:
**21 passed in 4.98 s**, using `--basetemp .pytest-integrated-source-cuda-repeat`
and stopping on first failure (`-x`). This closes the selected small source
cache and public-runtime CUDA contracts, including window sharing, EOF,
DMA correctness, cleanup and existing fallback controls. It does not validate
large host capacity, LTX throughput or a performance regression baseline.

- Prepared source-cache ABI 7/config v4: shared aligned windows, EOF clamping,
  bounded descriptor batching before DMA, legacy exact-range configuration
  retained. Model-capacity ceilings are opt-in and have not been allocated.
- Release build: passed using DDO's existing native build configuration.
- DLL SHA256: `029F6E840FA1BE377ABF17A6720578B17A0FE874605B9EFAE8CDBC886FBC0AF5`.
- Grouped CPU contracts: **100 passed, 7 deselected in 4.94 s**, using DDO
  `.venv-vmm`, after isolating real CUDA device queries in planner/profile
  tests. Earlier grouped attempts had 99 passes and one no-CUDA assertion
  failure caused by that test isolation issue; those are not passing gates.
- New CUDA window/neighbor/EOF/DMA contract passed in the approved 21-test repeat.
  Harness parity, larger registered capacity and LTX performance are pending.
  No new denoise measurement; previous baselines remain unchanged.

## Linux handoff preparation — 2026-10-03 (CPU only)

Prepared an isolated native CPU CMake project and a reviewable Linux build
helper, with separate `build-linux-cpu` / `build-linux` outputs and no automatic
GPU probe execution. Explicit library selection preserves span pool ON, source
dispatcher OFF and lifecycle fault hooks OFF. No Windows DLL rebuild/promotion.

- Seven shared C++ CPU contracts compiled with MSVC and passed: CTest 0.08s.
  The Linux-only eighth POSIX source contract is prepared, not executed here.
- 66 Python build-recipe/bootstrap/protected-route contracts passed in5.28s.
  Initial temporary-directory permission errors were resolved by using a new
  build-local pytest directory; the failing attempt is not a passing gate.
- Ruff for the new helper/tests passed. Candidate/default DLL SHA256 remained
  unchanged from the support checkpoint.
- No CUDA or model run; no new performance comparison. Native Linux host
  storage/cache/ring/admission/batch work remains pending for the real compiler.

Resume through [Linux checklist](../docs/linux-handoff.md); selected Windows
performance remains accepted within [documented scope](../docs/native-support-status.md).
