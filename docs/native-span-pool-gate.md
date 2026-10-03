# Completed consolidated CUDA gate: native span pool (2026-10-02)

Subsequent separately approved two-run repeatability gate also PASS:
warm GPU 1.412275 and 1.375524 s/it (both <=1.46), identical images, zero
errors/quarantine/fallbacks or denoise eviction/wait delta. Same DLL and
budgets; only output directories `span_pool_repeat_1/2` changed. Details in
the experiment sidecar. Both approvals are exhausted; no automatic reruns.

Executed once with user confirmation: three small contracts PASS, model exit 0,
identical image, warm 1.387751 s/it, no pool failures/quarantine or denoise
eviction/wait delta. Details in the experiment sidecar. Approval is exhausted;
commands below are historical and must not be rerun automatically. Use the DDO `.venv-vmm`, isolated
candidate SHA256 `8CAAD072...`, not the protected DLL. Stop at the first nonzero
exit, ownership failure, corruption or memory anomaly; do not repeat cases.

Hypothesis: native split/reuse preserves interior-pointer safety and stream
ordering, with less backing churn than exact-size rejection. Memory reduction
does not by itself prove faster inference.

## Small contracts (three fresh processes)

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$python = 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$env:PYTHONPATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-span-pool-20261002\Release\ddo_vmm.dll'

& $python ddo-vmm-allocator/scripts/vmm_span_pool_probe.py --library $env:DDO_VMM_LIBRARY_PATH
if ($LASTEXITCODE -ne 0) { throw 'Interior span gate failed' }

& $python ddo-vmm-allocator/scripts/vmm_torch_allocator_probe.py `
  --library $env:DDO_VMM_LIBRARY_PATH --megabytes 16 --cache-mib 512 `
  --verify-stream-lifetime --verify-nonblocking-reuse
if ($LASTEXITCODE -ne 0) { throw 'Stream/nonblocking reuse gate failed' }

& $python ddo-vmm-allocator/scripts/vmm_torch_allocator_probe.py `
  --library $env:DDO_VMM_LIBRARY_PATH --megabytes 16 --cache-mib 0
if ($LASTEXITCODE -ne 0) { throw 'Whole backing retirement gate failed' }
```

Pass criteria: all PASS, payload equality/noninterference, 512-byte-aligned
interior pointers, one shared mapping for small leases, zero leaked leases,
nonblocking repeated cross-stream reuse and zero cached physical backing in
the cache-disabled retirement process. Cache=0 is confined to that small
process; the model retains its previously approved 512 MiB cache configuration.
Mocked failure contracts already passed in CPU; this sequence does not inject
real driver errors or prove all pressure/fallback behavior.

## One model case, only after all small contracts pass

```powershell
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:HF_HOME = 'F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$env:DDO_VMM_TORCH_CACHE_MIB = '512'
& $python run_dynamic_modular_vmm_distilled.py `
  --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto `
  --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 `
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --no-pag --local-files-only --metrics-level 2 `
  --memory-monitor rss --ddo-gpu-timing `
  --output-dir outputs/ltx_image_modular_vmm/span_pool_20261002
if ($LASTEXITCODE -ne 0) { throw 'Model gate failed' }
```

Real prompt/default BF16, existing requested weight ceiling 4 GiB/effective
staging 1.5 GiB/source and registration 24.625 GiB; no profiler/fine timeline/
step waits. Require correct selected DLL, identical PNG, completed timing,
zero pool failures/quarantine, no new native fallback/fault/admission errors,
final source tickets zero and valid partition accounting. Compare warm GPU
steps against protected 1.712871 s/it, eviction/wait interval deltas against
exact-size 3,182 attempts/12.709596 s, and existing memory observations without
claiming sampled peak equals safe headroom. Stop on memory anomaly; no capacity
change or automatic promotion follows a single result.
