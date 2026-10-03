# Interrupted forward, resume and unload gate

Status: executed once after explicit approval; PASS (2026-10-02).
Preparation: 36 CPU contracts PASS in 5.00 s; two CUDA cases collected only.

Execution: two tests PASS (RuntimeError/KeyboardInterrupt), no skips, 4.86 s,
exit zero. Lifecycle OFF DLL SHA256 verified before execution:
`755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729`.
Both warm/resumed outputs matched CPU reference. Each case recorded one
look-ahead cancellation by the resumed boundary, unchanged source reads of
12,672 bytes, zero tickets/transfer events and stable committed/registered
bytes of 67,108,864. Pooled events were stable within each case: 2 for the
RuntimeError variant, 6 for KeyboardInterrupt; these distinct bounded values
do not establish a global leak. Final interruption followed by immediate unload
passed registry closure, all-hook removal and source-file deletion assertions.
No post-unload global memory gauge was measured.

Artifacts: `.tmp-forward-cancel-cuda-20261002/test_interrupted_forward_resum0/`
and `test_interrupted_forward_resum1/`, each containing `cancel_observations.json`.
Approval exhausted. No rerun, benchmark, runtime patch, rebuild or DLL promotion.

Two variants in one invocation: RuntimeError and KeyboardInterrupt. A generic
three-block ModuleList model (Linear32/SiLU) completes one reference-checked
warm forward, interrupts after block one, resumes with reference-checked output,
then interrupts again and immediately calls public remove_dynamic_offload.
No output readback or metrics collection occurs between the final interruption
and unload. These are cooperative Python exceptions after queued work, not
asynchronous signals inside an FFI call, process termination or driver poisoning.

Require actual look-ahead cancellation observed in the public DDO report,
unchanged source reads on resume (12,672 payload bytes), six file-backed bindings
and no retained CPU source bindings. At the resumed completed boundary: zero
tickets/transfer events, bounded event pool <=256 and source committed/registered
storage <=64 MiB. No unexpected native fault/fallback/admission denial/bypass.
Unload must close registry/remove all hooks; source file deletion must succeed.
Partial forward intentionally changes execution order; this gate does not
require an unchanged execution-plan mismatch counter. Report JSON records warm
and resumed observations under each pytest temporary case directory.

Use the existing tiny-model limits: 0.07 GiB weight ceiling, 2 MiB staging,
2 MiB arena page, one 64 MiB persistent-source window and registration ceiling.
No benchmark parameter changes or runtime/DLL promotion. Ordinary PyTorch
allocation plus native weight arena, not production global-allocator bootstrap.

Executed command, from the DDO checkout (one invocation, stop at first failure):

```powershell
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
$forwardCancelGateHash=(Get-FileHash -LiteralPath $env:DDO_VMM_LIBRARY_PATH -Algorithm SHA256).Hash
if ($forwardCancelGateHash -ne '755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729') { throw 'Unexpected DLL hash; stop gate' }
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_forward_cancel_cuda.py::test_interrupted_forward_resumes_then_unloads_without_readback -q -x --tb=short -p no:cacheprovider --basetemp=.tmp-forward-cancel-cuda-20261002
```

Accept two PASS, no skips, exit zero, all assertions satisfied. Stop on first
failure or memory anomaly; do not repeat, adjust limits or expand scope without
approval. Passing is bounded interrupted-forward proof only, not full-model
long-run cancellation, event-driver failure or multi-GPU support.
