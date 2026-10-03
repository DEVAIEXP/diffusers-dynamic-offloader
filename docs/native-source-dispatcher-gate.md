# Native source dispatcher gate (2026-10-02)

Executed once after explicit user confirmation. Four source pytest cases PASS
(4.73 s), span probe PASS with zero failures/quarantine, model exit 0 with the
expected image hash. Performance gate FAILED: warm 1.390072 s/it passes <=1.46,
but Generate 25.8173 s fails <=22.5 (milestone repeats 23.2125/23.2392 s).
Prepare 7.5827 s, first GPU step 1.693964 s, GPU denoise 11.424502 s, host
denoise 10.7700 s, total 46.6713 s. Smaller first step is not whole-phase gain.
Candidate `24A2F4F7...` is NOT promoted; protected/milestone hashes unchanged.
The approved sequence is exhausted; no retry or parameter/default change.

Before denoise all 25,846,148,360 source bytes were already read, with zero
source registrations/DMA. Source preparation took 6.627046 s. Code inspection
found a concrete serialization: `ddo_vmm_host_ring_wait_file_read` holds the
ring mutex during `future.get()`, while `commit_storage` requires that mutex.
This prevents concurrent admissions during each read wait. It is a supported
bottleneck diagnosis, not a measured attribution of every elapsed second.
Correction must retain slot/read ownership through completion and clear/close;
simply unlocking around a consumable future is not a safe lifetime contract.

Commands below document the executed approval, not authorization to rerun.

Hypothesis: asynchronous native source filling can overlap setup/compute,
lowering whole Generate without changing residency, registration ceilings,
DMA bytes or warm throughput. Smaller first step alone is not acceptance.

## Small gates: four pytest cases, then one span probe

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$python = 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$env:PYTHONPATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-source-dispatcher-20261002\Release\ddo_vmm.dll'
$env:DDO_VMM_TORCH_CACHE_MIB = '512'
& $python -m pytest ddo-vmm-allocator/tests/test_source_dispatcher.py ddo-vmm-allocator/tests/test_source_cache.py::test_persistent_storage_survives_registration_recycling_without_reread -x -q -p no:cacheprovider --basetemp ddo-vmm-allocator/build-source-dispatcher-20261002/pytest-cuda-gate
if ($LASTEXITCODE -ne 0) { throw 'Source lifecycle gate failed' }
# Inspect result: require four PASS, no skips; do not continue on a skip.
& $python ddo-vmm-allocator/scripts/vmm_span_pool_probe.py --library $env:DDO_VMM_LIBRARY_PATH
if ($LASTEXITCODE -ne 0) { throw 'Span pool gate failed' }
```

Require source-only preparation with no DMA/new registration, numerical payload
equality after file-owner close, bounded-cache native ring fallback, independent
registration recycling without reread, clear while reads may still be active,
zero storage/tickets after clear and fresh correct payload after restart. Small
probe must preserve span isolation/stream lifetime. These are real small gates,
not exhaustive failure injection, queue-saturation or platform coverage.

## One unchanged model case, only after all small gates pass

```powershell
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:HF_HOME = 'F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
& $python run_dynamic_modular_vmm_distilled.py `
  --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto `
  --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 `
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --no-pag --local-files-only --metrics-level 2 `
  --memory-monitor rss --ddo-gpu-timing `
  --output-dir outputs/ltx_image_modular_vmm/source_dispatcher_20261002
if ($LASTEXITCODE -ne 0) { throw 'Model gate failed' }
```

Same real-prompt BF16 case, DDO Python, cache 512 MiB, weight 4 GiB/effective
staging 1.5 GiB/source and registration 24.625 GiB; no profiler/fine timeline/
step waits. Verify selected DLL and source-dispatcher capability in the public
report. Require identical `877876A4...` PNG, complete timing, final tickets zero,
zero new native errors/fallbacks/quarantine, no denoise eviction/wait delta and
conserved memory accounting. Warm <=1.46 s/it; Generate <=22.5 s is the proposed
performance criterion versus prior repeats 23.2125/23.2392 s. Record prepare,
first step, source-ready wait and memory; include prepare in Generate. A failed
performance criterion does not justify changing budgets or promoting the DLL.
No second run or default change is included in this proposed approval.
