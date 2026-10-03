# Native lifecycle hardening checkpoint

## Consolidated checkpoint and next full-model gate

Small-gate regression is complete for the latest upload diagnostic edit (six
CUDA PASS, 5.16 s). The next proposed check is one real-workload bootstrap
process with the lifecycle OFF candidate, three eight-step generations and
unchanged benchmark limits. Executed once after fresh approval: PASS, exit zero;
warm 1.401990/1.397915/1.394595 s/it, no source rereads or storage/registration
growth, final tickets/transfers zero, no native fallback/quarantine/monitor error.
Effective staging is the unchanged three-slot host conversion: 0.5 GiB CLI per
slot, 1.5 GiB total DDO capacity. Approval exhausted, no promotion. See
[executed command and unchanged acceptance](native-hardened-reuse-gate.md).
This does not promote the DLL or close driver-failure, full-model cancellation,
operator/platform coverage, text-encoder migration or publication work.

## 2026-10-02: upload first-error diagnostics (CUDA regression PASS)

Two execution-buffer upload paths could replace the original upload exception
with a secondary fence/release failure. Both now attempt the same protection
operation while retaining the original exception and attaching cleanup failure
notes. If upload succeeds, fence failure still propagates normally. Unsafe
backing/lease close refusal remains intact; this is not driver recovery or proof
of completion. Shared cache/ticket cleanup diagnostics also preserve a first
error if secondary cleanup raises KeyboardInterrupt. No native C++ change.

70 CPU contracts PASS in 5.05 s. Nine new fake tests cover both execution buffer
types with original RuntimeError/KeyboardInterrupt and secondary native-error/
KeyboardInterrupt, and successful upload followed by fence failure. Existing
Python 3.10 exception-note fallback and source ownership contracts also passed.
Other startup/read cleanup paths and native event error propagation remain
separate audit work; this checkpoint does not claim all errors are normalized.

CUDA regression executed once after explicit approval: four existing public
execution-buffer cases plus two interrupted-forward variants. Six tests PASS,
no skips, 5.16 s, exit zero. All original numeric/resource/reuse/cleanup assertions
passed. Lifecycle OFF DLL SHA256 verified before execution. This exercises
actual native execution slots, unlike an entirely resident tiny model.
Existing case limits remain unchanged: execution-buffer cases use 6 MiB weight
and 2 MiB staging with 64 MiB source cache, interruption cases retain 0.07 GiB
weight and 2 MiB staging with 64 MiB committed/registration ceilings. No model
benchmark change. Approval exhausted; no rerun or parameter adjustment. CUDA
validated healthy operation after the bridge correction; simultaneous upload/
fence failure and first-error retention remain covered by CPU fake contracts,
not real driver fault injection.

Executed command:

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
$uploadErrorGateHash=(Get-FileHash -LiteralPath $env:DDO_VMM_LIBRARY_PATH -Algorithm SHA256).Hash
if ($uploadErrorGateHash -ne '755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729') { throw 'Unexpected DLL hash; stop gate' }
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_execution_buffer.py tests/test_vmm_forward_cancel_cuda.py::test_interrupted_forward_resumes_then_unloads_without_readback -q -x --tb=short -p no:cacheprovider --basetemp=.tmp-upload-first-error-cuda-20261002
```

No rebuild, benchmark or DLL promotion. Fake secondary errors do not establish
poisoned-driver recovery or asynchronous signal safety during FFI.

## 2026-10-02: partial look-ahead cancellation retry (CUDA regression PASS)

Further failure-path audit found cancellation could repeat an already successful
unpin when a later binding failed. Each unit now records its confirmed cancelled
binding count, retains prefetched status until all releases finish and resumes
only outstanding releases on explicit retry. Canonical tensor descriptors are
unchanged. Upload fencing remains mandatory; this is cleanup progress, not
Python cache/residency policy. Native execution-lease failures retain the lease.

61 selected CPU contracts PASS in 5.13 s, including eight new fake contracts
covering first/second unpin failures (native error and KeyboardInterrupt), upload
fence failure, execution-lease cancellation failure and a successful unit before
a failed peer. No CUDA execution, benchmark, rebuild or promotion in this update.

One-invocation CUDA regression executed after explicit approval: two existing
interrupted-forward cases plus existing 128-forward reload case, unchanged
small-case limits. Three tests PASS, no skips, 5.24 s, exit zero. All existing
numeric/source/resource/cleanup assertions passed. Lifecycle OFF DLL SHA256
verified before execution. No rerun, benchmark, rebuild or promotion; approval
exhausted. This validates normal cooperative cancellation/reuse after the edit;
actual driver failures remain represented only by the CPU fake contracts.

Executed command:

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
$partialCancelGateHash=(Get-FileHash -LiteralPath $env:DDO_VMM_LIBRARY_PATH -Algorithm SHA256).Hash
if ($partialCancelGateHash -ne '755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729') { throw 'Unexpected DLL hash; stop gate' }
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_forward_cancel_cuda.py::test_interrupted_forward_resumes_then_unloads_without_readback tests/test_vmm_model_lifecycle_cuda.py::test_generic_model_128_forwards_with_unload_reload_and_source_replacement -q -x --tb=short -p no:cacheprovider --basetemp=.tmp-partial-prefetch-cancel-cuda-20261002
```

CPU mocks do not prove real driver failure or signal timing during FFI. Earlier
gates remain historical results; the three-test execution above is the CUDA
regression validation of this bridge edit.

## 2026-10-02: interrupted-forward gate PASS

Prepared two variants (RuntimeError/KeyboardInterrupt) for a generic three-block
model: warm, interrupt, resume, interrupt then immediate unload. No additional
runtime patch. 36 selected CPU contracts PASS in 5.00 s; two CUDA cases collected
only, not executed. Gate requires observed look-ahead cancellation, numerically
correct resume without source reread, bounded public resources and explicit
hook/registry/file cleanup.

Executed once after approval: both RuntimeError/KeyboardInterrupt cases PASS,
no skips, 4.86 s, exit zero. One look-ahead cancellation observed per case by the
resumed boundary; outputs correct, source reads unchanged at 12,672 bytes,
tickets/transfer events zero, committed/registered source bytes stable at 64 MiB.
Second interruption followed immediately by unload passed registry/hook/file
cleanup. These are cooperative Python exceptions after the first block, not
signals during FFI, driver poisoning or full-model cancellation. No global
post-unload memory gauge, benchmark or promotion. Approval exhausted. See
[executed command and observations](native-forward-cancel-gate.md).

## 2026-10-02: pending unpin ownership correction (CUDA regression PASS)

Audit found the capacity reaper removed the oldest event/ranges before querying
or waiting for completion. Query/wait failure therefore lost queue ownership.
The general collector also suppressed native unpin errors and dropped those
pending ranges. Both paths now retain event/range ownership until confirmed
release, propagate errors, and commit each successful unpin so retry does not
repeat earlier releases. This changes failure handling only; no native allocator
policy, ceilings, bootstrap, linear/NVMe route or DLL promotion was changed.

51 selected CPU contracts PASS in 5.13 s. Ten new fake-event cases cover query
and wait failures, partial native-unpin failure, KeyboardInterrupt, incomplete
nonblocking events and an earlier successful entry followed by failure. These
do not prove real driver error or signal timing during a CUDA call.

CUDA regression sequence executed once after explicit approval: missing-context
gate plus the existing tiny public-model 128-forward A->B->A->B reload gate.
Two tests PASS, no skips, 5.47 s, exit zero. Numeric upload/forward checks and
all existing ownership/resource/cleanup assertions passed. Existing case
parameters unchanged, lifecycle OFF DLL hash verified before execution. No
performance benchmark, rebuild or promotion. This does not prove a poisoned
driver fault or full-model cancellation. Approval exhausted; no automatic rerun.

Executed command:

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
$unpinGateHash=(Get-FileHash -LiteralPath $env:DDO_VMM_LIBRARY_PATH -Algorithm SHA256).Hash
if ($unpinGateHash -ne '755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729') { throw 'Unexpected DLL hash; stop gate' }
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_context_recovery_cuda.py tests/test_vmm_model_lifecycle_cuda.py::test_generic_model_128_forwards_with_unload_reload_and_source_replacement -q -x --tb=short -p no:cacheprovider --basetemp=.tmp-pending-unpin-cuda-20261002
```

## 2026-10-02: DLL compatibility and context ownership audit

DLL loading now reports the selected absolute path and preserves the original
loader/dependency error. Required binding exports produce an explicit package/DLL
compatibility error, without silently selecting another build. Optional streaming
and telemetry exports remain optional. The fresh-process allocator installer
checks malloc/free exports before importing Torch or querying CUDA. These checks
detect missing exports, not incompatible signatures hidden behind matching names.

Native source review confirms cache creation retains the requested device primary
context and rejects a different current context. Cache operations validate the
owning context; cross-device arena copies are rejected. Failed closes retain
ownership. No automatic device/context switch or native policy change was added.
This audit is not real-driver multi-GPU/context-switch validation.

CPU validation: 35 tests passed in 5.06 s, including missing required symbols,
optional symbol absence, dependency load failure, pre-Torch callback validation,
library selection, bootstrap, probe context balance and source failure retention.
An initial attempt encountered pytest temporary-directory permissions and a
missing explicit test DLL; the final run used a workspace temporary directory
and the isolated lifecycle OFF DLL. One new-test expectation was corrected to
the existing empty-dictionary representation of unsupported pool telemetry.

Separate load/export checks passed for the protected default DLL, span-pool
milestone and lifecycle OFF candidate without calling CUDA exports. Their hashes
remain unchanged. No rebuild, CUDA test, benchmark or DLL promotion occurred.
Remaining: actual wrong-context rejection/recovery and multi-device ownership
coverage, plus driver error/cancellation and full-model longevity gates.

Missing-current-context rejection/recovery gate now executed once after explicit
approval: one CUDA test PASS, no skip, 4.68 s, exit zero. Ticket/cache handles and
arena ownership survived rejected cleanup; restoring the context permitted close
and a second correct upload without file reread. Clear accounting reached zero
and file deletion passed. Preparation suite: 24 CPU contracts PASS (4.88 s).
This proves the absent-current-context case only, not a different live context,
multi-GPU ownership or device-lost recovery. See
[executed command and limited coverage](native-context-recovery-gate.md).

## 2026-10-02: audit and CPU failure contracts

Scope: experimental fast VMM source ownership. No native allocator/CUDA policy,
budget, bootstrap, protected linear or NVMe runtime changes. The span-pool
milestone remains the performance reference. No CUDA run in this checkpoint.

Reviewed native source ticket close, source batch partial submission/completion,
cache clear/close and Python ownership bridges. A close failure must retain its
handle and dependencies. Source cache close refuses live tickets/batches; batch
submission retains pending owners and marks unproven completion unsafe. These
are source-review observations, not injected real-driver failure proof.

Added CPU-only fake-ABI contracts in allocator `test_source_lifecycle_cpu.py`:

- Ticket/cache/batch failed close retains handle; ticket arena/ring and batch
  destinations survive until successful explicit cleanup.
- Failed batch submission retains its potential DMA destination.
- Transport cleanup stops at ticket/batch/cache/source failure, without closing
  downstream source/cache/ring owners; a successful retry completes cleanup.
- Cache/ticket context managers preserve the first operation error when cleanup
  also fails, append the cleanup error as an exception note and retain owners.
  Explicit close, or close after a successful body, still raises on failure.

The last item required a Python bridge correction: previously a failing
`__exit__` made cleanup the primary exception. No native handle deletion or
completion semantics changed. Failed context cleanup is NOT successful cleanup;
callers must retain owners and deal explicitly with failure. Destructor fallback
is not a recovery guarantee, particularly after CUDA context failure.

Validation: 44 selected CPU contracts passed in 4.73 s, including the new
13 cases, native file ownership/read/repeated-OS-handle contracts, public timing
and bootstrap tests. Native source tests load the milestone DLL but do not
initialize CUDA; the fresh-process CUDA-initialization guard passed. No DLL
rebuild or promotion. CPU mocks do not establish real asynchronous DMA safety.

## Remaining work before declaring lifecycle complete

Update: the isolated source copy/close refusal gate now passed three CUDA cases
in 4.80 s (no skips), including 16 small batch cycles and scoped event/source
stability. See [executed gate](native-lifecycle-cuda-gate.md). No benchmark or
promotion. Item 1's controlled partial-copy/close contracts are now covered;
actual driver recording/query/poisoned-context cases remain unproven. Item 2
has scoped transfer/pool/batch gauges, not complete global live-event accounting.

Generic public-route lifecycle gate also passed: 128 numerically checked
forwards over four small A->B->A->B model attachments, one test/no skip/5.25 s.
Hooks/registries/files closed, aggregate source accounting zero before subsequent
attachments, and resource/source observations stable within each attachment.
See [executed model gate](native-model-lifecycle-gate.md). This advances item 3's
reload proof for tiny dense models; per-forward collection, no VRAM pressure and
ordinary PyTorch allocation do not establish production full-model longevity or
asynchronous cancellation. Those and actual driver failure cases remain open.

1. Add explicit native failure injection or equivalent controlled contract
   probes for partial upload/event record/query/close failure. Owners must remain
   protected when completion cannot be established. Do not simulate success by
   clearing an unsafe flag or globally synchronizing a failed context.
2. Observe live event ownership/destruction separately from cumulative created
   events. The reuse gate's 736 ->1216 ->1696 creations is not evidence of a leak;
   the free event pool is bounded at 256 and excess events are destroyed. Real
   failure paths and sustained live-resource accounting remain to be proved.
3. Validate same-model longer reuse plus unload/reload and cancellation, with
   numerical/resource checks at equivalent completion boundaries. Three prior
   generations proved source reuse, not long-run stability/model swapping.
4. Audit remaining first-error propagation, native context/device ownership,
   version compatibility and representative operator support separately. Do not
   label the whole runtime hardened from this focused bridge correction.

Before each CUDA sequence: provide exact commands, hypothesis and acceptance,
get user approval, retain current ceilings, stop on the first failed gate or
memory anomaly. Prefer one consolidated sequence; do not rerun the exhausted
three-generation gate or change historical performance criteria.
