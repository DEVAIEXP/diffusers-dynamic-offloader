# Cold start and same-model generation reuse

Status: approved sequence executed once; source reuse PASS, strict warm gate
FAIL on generation 1 (1.473632 >1.46). Exit 0, three complete CUDA reports,
source reads/storage/registrations stable after generation 1, final tickets
704 -> 0, zero native/monitor errors. First GPU steps 7.627204 / 1.479055 /
1.420242 s; warm 1.473632 / 1.437565 / 1.408667 s/it. No rerun/promotion;
approval exhausted. Results in `experiments/dynamic_offload_results.md`.
Preserve the span-pool
milestone `8CAAD072...` and protected runtime `4AC3D8B...`. Do not promote the
source-dispatcher candidate: its previous whole-Generate performance failed.

## Decision and bounded scope

Investigate cold preparation/first step and reuse before further performance
changes, then return to lifecycle/failure/compatibility hardening. The previously
fixed global read-wait lock and whole-window barrier are not new hypotheses.
No capacity/worker changes or new Python scheduling/residency policy.

The host already supports repeated denoise generations with one attached model.
It now persists a public DDO/timing snapshot for every generation after the
existing CPU latent readback, without an extra wait or intermediate collection.
Final resource collection remains unchanged. This is same-model reuse, NOT
full-pipeline repetition, model swap, unload/reload or a long-run stability proof.

`generation_observations[].elapsed_seconds` covers latent preparation through
CPU latent readback. It excludes shared transformer setup, report snapshotting,
final cleanup and VAE decode; never compare it directly with whole Generate.
The CLI staging value is per slot: 0.5 GiB here becomes three slots/1.5 GiB total
in the public DDO configuration. This is the existing host conversion, not a
later ceiling increase; retain this distinction when interpreting metadata.
CUDA step/phase scopes remain those of the public timing report. Native counters
remain cumulative: compare adjacent public before/after snapshots, never sum
nested source/readiness/registration timers as independent phases.

## Executed approved command (not authorization to rerun)

Hypothesis: retained native source bytes eliminate additional source file reads
on generations 2 and 3, reducing the cold first-step penalty without growing
storage/registration/pool ownership or degrading warm throughput. Use the proven
milestone first, not another speculative allocator correction.

```powershell
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$ddoPython = 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$env:PYTHONPATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-span-pool-20261002\Release\ddo_vmm.dll'
$env:DDO_VMM_TORCH_CACHE_MIB = '512'
$env:HF_HOME = 'F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
& $ddoPython run_dynamic_modular_vmm_distilled.py `
  --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto `
  --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 `
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --width 1280 --height 704 --seed 43 --steps 8 --generation-repeats 3 `
  --no-pag --local-files-only --metrics-level 2 `
  --memory-monitor rss --ddo-gpu-timing `
  --output-dir outputs/ltx_image_modular_vmm/generation_reuse_gate
if ($LASTEXITCODE -ne 0) { throw 'Reuse gate failed; stop' }
```

Before execution verify selected DLL hash equals the preserved milestone and
confirm effective configuration. Approval covers ONE three-generation process,
not reruns, candidates, higher limits or extra tests. Keep current monitoring;
stop on error or observed memory anomaly. Real prompt/BF16; encoder runs once.
The existing generator advances between generations. Only the last latent is
decoded; do not expect the saved PNG to match the one-generation seed43 image
or claim per-generation numerical parity from timing/resource observations.

## Evaluation and next action

- Require three complete eight-step CUDA reports; pending/missing timing is
  inconclusive, never zero. Warm <=1.46 s/it for each generation.
- Compare first step, complete GPU phase and readback-complete generation wall
  time separately. Count shared setup once when reporting amortized cost.
- Expect no additional native source-read bytes after generation 1 if retained
  source capacity covers the model. DMA can remain nonzero; a source-cache hit
  is not GPU residency and does not prove WDDM migration.
- Inspect source committed/registered bytes, live tickets and unique pool
  backing at equivalent boundaries. No exceeded ceilings, unexpected fallback,
  quarantine, mismatches or faults. Pending metadata alone is not a leak.
- Final existing nonblocking collection must retire tickets to zero. Intermediate
  snapshots deliberately do not collect; investigate growth, do not hide it by
  inserting cleanup into each generation.
- If cold cost disappears on reuse: prioritize safe persistent lifecycle, not
  another steady-state optimization. If it persists: use phase/counter evidence
  to identify whether source, registration, DMA or execution remains critical;
  propose one targeted native correction only with a concrete mechanism.
- Three generations cannot prove long-run stability or establish performance
  variance. No automatic promotion. Subsequent issue-derived gates remain separate.

Record actual results in `experiments/dynamic_offload_results.md` after approval.
