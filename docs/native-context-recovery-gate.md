# Native source context rejection/recovery gate

Status: executed once after explicit approval; PASS (2026-10-02).

Result: one test passed, no skips, 4.68 s, exit zero. DLL SHA256 verified before
execution: `755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729`.
All three missing-context cleanup operations rejected with status 3 and the
owning-context diagnostic, retaining ticket/cache handles and arena ownership.
After restoring the original context, ticket close and a second upload passed;
both numeric payload checks passed. Source reads remained 2 MiB (no reread),
completed boundaries had zero live tickets/transfer events, clear left zero
committed/registered bytes and source file deletion after close succeeded.
Approval is exhausted; no rerun, benchmark, rebuild or DLL promotion occurred.

Scope: one Windows GPU, native cache/ticket ownership. The test temporarily
detaches the thread's current context and restores its original context in a
finally guard. It never creates/destroys a CUDA context or resets the GPU. No
Torch operation runs while detached. No native implementation or residency policy
change; existing DLL used, without failure-injection exports.

CPU preparation: 24 contracts PASS in 4.88 s, including six detach/restore guard
tests, existing primary-context guard, DLL diagnostics/selection and bootstrap.
Initial collection lacked the submodule PYTHONPATH; final commands set it
explicitly. CUDA test collected only (one case), not executed.

## Executed command (one run, stop on failure)

Run from `E:\ProjetosIA\diffusers-dynamic-offloader`:

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
$contextGateHash=(Get-FileHash -LiteralPath $env:DDO_VMM_LIBRARY_PATH -Algorithm SHA256).Hash
if ($contextGateHash -ne '755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729') { throw 'Unexpected DLL hash; stop gate' }
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_context_recovery_cuda.py -q -x --tb=short -p no:cacheprovider --basetemp=.tmp-context-recovery-cuda-20261002
```

## Hypothesis and acceptance

After a real tiny DMA submission, removing the thread's current context causes
ticket close, cache clear and cache close to reject with CUDA status 3 and the
owning-context diagnostic. Ticket/cache handles and arena ownership must survive.
Restoring the owner context permits explicit close and a fresh ticket/upload from
the same cache; both outputs must be uniformly byte 53. The second request must
not reread the file. At completed boundaries: no live tickets/transfer events;
clear leaves committed/registered bytes zero; file deletion after close succeeds.

Accept only one test PASS, no skip, exit zero, all assertions satisfied. Stop
on first failure or memory anomaly; do not rerun or extend scope automatically.
Source/cache capacity 2 MiB, process source ceiling 6 MiB, arena 2 MiB, plus
ordinary Torch/driver overhead. No full-model benchmark, bootstrap promotion,
production parameter changes or protected-runtime mutation.

Limitations: absent current context is not a different live context or a second
GPU. This does not prove device-lost recovery, event-driver failure handling,
cancellation, production custom-allocator longevity or large-model pressure.
