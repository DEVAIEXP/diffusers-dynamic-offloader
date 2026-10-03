# Isolated native source lifecycle gate

Status: approved three-case CUDA sequence executed once: **3 PASS, no skips,
4.80 s**, exit 0. Partial-copy/close-refusal ownership and 16-cycle event/source
reuse contracts passed. ON DLL hash verified before execution; milestone and
protected DLL hashes unchanged afterward. No benchmark, retry or promotion;
approval exhausted. This result covers the controlled contracts below, not
actual driver event failures, long-process/model-swap safety or performance.

## Implementation and limits

`DDO_VMM_SOURCE_TEST_HOOKS` is OFF by default. Its test-only native export
configures one cache to refuse a copy after N allowed attempts or refuse one
ticket close before completion proof. It does not provoke a driver error,
discard owners, change memory limits or bypass actual completion drainage.
The configured refusal is consumed once. No runner/API user configuration.
OFF builds omit the export and all injection branches; never publish/promote
the ON DLL as a production runtime.

New optional CPU-only event-owner snapshot locks batch/cache metadata and
counts events attached to tracked DMA transfers, free pool and batch completion
owners. It does not query CUDA, count unsubmitted ticket events, other allocators
or OS handles, and is not a complete global live-resource gauge. Existing ABI
structs remain unchanged; older DLLs omit optional metric fields.

ON build `build-lifecycle-20261002/Release/ddo_vmm.dll`:
SHA256 `9C0313293E3865EF602A5350F209E28D44A09022703333DE43B3B9E3AA7C6740`.
OFF build `build-lifecycle-off-20261002/Release/ddo_vmm.dll`:
SHA256 `755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729`.
Both are isolated, span pool ON/source dispatcher OFF. No performance comparison
or default promotion; preserved milestone/protected DLL hashes unchanged.

ON/OFF compiled, OFF export isolation PASS. Selected CPU suites 48 PASS each
(ON 4.70 s), including native null rejection without CUDA initialization,
optional snapshot binding, file ownership and first-error fallback contract
for Python 3.10 (simulated missing add_note, not a Python 3.10 interpreter run).
Three CUDA cases previously collected only; all three now executed and passed
after explicit approval. The CPU results above remain separate evidence.

## Executed approved sequence (not authorization to rerun)

Hypothesis: actual native partial-copy and close-refusal paths retain handles,
source bytes and destinations until the ordinary completion-based close succeeds;
small repeated batches stabilize tracked event ownership and reuse source bytes.

Run ONE sequence, exactly three small cases, no model benchmark or rerun:

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-20261002\Release\ddo_vmm.dll'
& .venv-vmm/Scripts/python.exe -m pytest `
  ddo-vmm-allocator/tests/test_source_lifecycle_cuda.py `
  -x -q -p no:cacheprovider `
  --basetemp ddo-vmm-allocator/build-lifecycle-20261002/pytest-cuda-gate
if ($LASTEXITCODE -ne 0) { throw 'Native lifecycle gate failed; stop' }
```

Before execution verify ON hash. First two cases use only 4 MiB model payload,
4 MiB retained/registered source, 2 MiB ring; third uses 2 MiB payload/source
and 16 cycles. Require all three PASS with no skip, correct byte values,
expected refusal status/message, cache-close BUSY while batches remain,
retained handles/destinations, successful later close, zero transfer/batch event
owners and tickets after cleanup, free pool <=256, clear returns committed and
registered storage to zero. Repeated case must stabilize created-event count,
pooled-event count, source-read and committed/registered bytes after its first
cycle. Stop at first failure/anomaly; do not enlarge capacities or add cases.

Coverage is deliberately limited: injected copy refusal occurs before a later
DMA, after an earlier real DMA; close refusal occurs before proof. This does NOT
simulate driver cuEventRecord/query errors, poisoned contexts, hangs, compute
slot cancellation, model swaps or sustained large-model inference. Passing it
cannot close the complete robustness plan or establish performance non-regression.
