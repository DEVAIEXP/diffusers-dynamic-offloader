# Explicit native idle device-cache trim candidate

Status: compiled Windows candidate, nine mocked CPU contracts and the approved
small CUDA trim gate passed on 2026-10-11. Qwen performance acceptance is pending.
Historical joined-spans DLL and default runtime behavior are preserved.

Approved command used DDO .venv-vmm and scripts/vmm_idle_trim_probe.py with the
candidate DLL. Before trim: mapped 35,651,584 bytes, one live 1,024-byte lease,
33,554,432 pending/whole-idle bytes. After trim: mapped 2,097,152 bytes, live
lease unchanged, pending/quarantined bytes zero. Active payload remained exact.
After dropping the live lease, repeated trim and allocation-after-trim passed;
final mapped/live/pending/quarantined bytes and failures were zero. Native wait
observation total was 0.0002512 s. This small ownership gate is not a model or
general multi-process/platform acceptance.

Candidate SHA256: 468A854EB550C6299C3DE785B65ECD4F4DF382C488B6F1C83AA6899983F12CFE.
Preserved historical SHA256: EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.

## Selected Qwen leaf comparison

User completed both same-candidate 1024x1024/20-step base BF16 runs:
`qwen21_vae_leaf_trim_control` and `qwen21_vae_leaf_trim_enabled` in the host
outputs. Both PNG hashes match the accepted reference exactly; no monitor/native
failures or temporary tensors were reported. Decode peak/time: control
5.05419921875 GiB / 1.8069 s, trim 3.92724609375 GiB / 1.8943 s. Explicit trim
phase took 0.1100 s. These are single sequential observations, not repeatability.

Important limitation: trim snapshot mapped bytes fell only from 1,247,805,440
to 1,245,708,288 (2 MiB). One live 8,519,680-byte lease remained before/after;
the containing backing cannot safely be released. Quarantined bytes/failures
were zero. This is not full device-cache release. The later decode peak reduction
must not be described as 1.13 GiB directly freed by this trim. Investigate the
live lease owner and allocator reuse before claiming causality or complete
cleanup. Do not force-free it or expand the VAE VMM route yet.

## cuBLAS workspace hypothesis: small CUDA gate passed

The observed live request is exactly 8,519,680 bytes (8.125 MiB), matching
the documented default cuBLAS workspace (4096 KiB * 2 + 16 KiB * 8).
PyTorch retains workspaces after model references are dropped. This strongly
suggests the owner, but size matching alone does not prove the Qwen lease's origin.
The installed Torch exposes `_C._cuda_clearCublasWorkspaces`; CUDA remained
uninitialized during read-only binding inspection.
See [PyTorch CUDA workspace documentation](https://github.com/pytorch/docs/blob/site/stable/notes/cuda.md).

Optional `clear_cublas_workspaces=True` in the DDO helper synchronizes completed
work, records the native pool snapshot before workspace cleanup, invokes the
guarded private Torch binding, then trims whole-idle backing. This is explicitly
single-visible-GPU, non-capture, quiescent compute-thread usage; no concurrent
compute may run. Multi-GPU/capture/missing binding routes refuse cleanup.
No workspace size or GEMM policy changes; future GEMM can allocate a workspace
again. Errors propagate. Default trim behavior is unchanged, and the DLL does
not require another rebuild. Thirteen mocked CPU contracts passed.

On 2026-10-11 the user-approved small probe passed using DDO .venv-vmm and
the same candidate DLL hash above. Tiny GEMM retained exactly 8,519,680 requested
bytes in one live lease, with 10,485,760 bytes mapped. Workspace cleanup retired
that lease; subsequent native trim reduced mapped/live/pending/quarantined bytes
to zero, without failures. GEMM after cleanup reproduced the exact output and
workspace reallocation/second cleanup succeeded. Final pool was again empty.
First helper elapsed time: 0.1020755 s; total observed native eviction wait:
0.0000267 s. This proves the mechanism in the small probe, not yet ownership
of the surviving Qwen allocation. The real Qwen generation is still pending.

Small proposed gate: `scripts/vmm_blas_trim_probe.py`. Generate a tiny exact GEMM,
drop its tensors, show a retained live workspace, clear it and require zero
live/mapped/quarantined bytes or failures. Repeat GEMM and cleanup to verify
workspace recreation. Run only with explicit approval. After that, the Qwen
runner's `--clear-cublas-workspaces-before-vae` can be evaluated in combination
with `--release-native-cache-before-vae`; neither flag is automatic.

## Selected Qwen cuBLAS cleanup acceptance

User completed `qwen21_vae_leaf_blas_trim_enabled` with the same candidate DLL
and selected 1024x1024/20-step leaf VAE workload. PNG bytes match the accepted
reference exactly, with no monitor/native failures or temporary fallback.
Before cuBLAS cleanup: one live 8,519,680-byte lease, 1,247,805,440 mapped bytes.
After cleanup: zero live/requested/mapped/pending/quarantined bytes and failures.
The subsequent explicit trim was a no-op: freeing the workspace already caused
normal pool retirement to release backing. This confirms the workspace owner
in the real selected workload without force-freeing any live allocation.

Cleanup phase: 0.1201 s (helper 0.1191073 s). VAE-load sampled absolute driver
peak: 1.07763671875 GiB, versus 2.23779296875 GiB with simple trim alone.
Decode: 2.1736 s / 4.74560546875 GiB absolute sampled driver peak; simple trim
was 1.8943 s / 3.92724609375 GiB. Full cleanup is thus successful but not the
lowest decode peak in these sequential observations. Pool recreation/reuse
may affect execution peak; no allocation-shape causal proof is claimed.
Do not promote full cleanup as automatically better or rewrite the earlier
lower-peak result. Selected single-GPU boundary only; no Linux or multi-GPU gate.

`release_native_vmm_idle_cache(enabled=True, phase="before_vae", device=...)`
is an explicit completed-phase helper, not a per-layer policy. Call only after
detaching the previous component and dropping caller-owned references. The helper
collects Python garbage, synchronizes once and uses the actual installed native
library path, never a newly changed environment path. Disabled/uninitialized
routes do not load native code or initialize CUDA. Missing exports are unsupported,
not a successful release. Native proof/release failures propagate.

Optional ABI `ddo_vmm_torch_trim_idle_v1` invokes the existing native checked
whole-backing eviction transaction. It selects no backing with live leases or
quarantined bytes, proves pending event completion, and checks physical eviction
before committing retirement. Partial backing, live tensors, model arenas and
host source storage are outside this operation. It does not change budgets or
purge Windows standby RAM. A completed trim does not promise zero process VRAM.
Before/after pool snapshots are observations, not an atomic concurrent census.

Candidate: `ddo-vmm-allocator/build-idle-trim-cu128-20261011/Release/ddo_vmm.dll`,
CUDA toolkit 12.8, span pool/joined spans/allocation diagnostics enabled, source
dispatcher disabled. No historical binary was overwritten. Linux requires a
separate build and validation; the added portable source is not Linux acceptance.

Before real-model execution, obtain approval for `scripts/vmm_idle_trim_probe.py`:
pending whole-idle backing releases, active neighbor payload stays exact, repeated
trim is safe, subsequent allocation works, and no pool failures/quarantine occur.
Only after that gate compare the Qwen leaf VAE with/without explicit trim using
the same candidate DLL for both runs. This avoids confounding a DLL change with
the lifecycle intervention. Preserve the accepted image and capacity settings.
