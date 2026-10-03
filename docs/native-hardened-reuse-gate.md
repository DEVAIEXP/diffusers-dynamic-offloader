# Hardened bridge real-model reuse regression gate

Status: executed once after explicit approval; PASS (2026-10-02), process exit 0.
Approval exhausted; no rerun or promotion.

## Executed result

Three complete eight-step CUDA reports, each warm below the unchanged 1.46 s/it
limit:

| Generation | First GPU step (s) | Warm (s/it) | GPU denoise phase (s) | Prepare-latents through CPU readback (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7.528940 | 1.401990 | 17.342902 | 17.356360 |
| 2 | 1.395665 | 1.397915 | 11.181079 | 11.196348 |
| 3 | 1.390656 | 1.394595 | 11.152851 | 11.166465 |

Source read bytes 25,846,148,360, committed/registered bytes 26,038,239,232 and
388 registrations remained unchanged across all generations. Native source
readiness cumulative 6.720228/6.766770/6.813429 s is nested timing, not additional
exclusive I/O. DMA cumulative 187,588,136,576/372,488,764,032/557,389,391,488 bytes;
each additional generation still copies 184,900,627,456 bytes.

At readback snapshots: transfer owners/tickets 736/704/704, pooled events
0/32/32 and batch completion owners 103/103/103. These snapshots deliberately
retain completed ticket metadata until ordinary cleanup. Final nonblocking
collection retired 704 tickets to zero; transfer owners zero, free pool 256,
batch completion owners 103 while the attached model still owned immutable
batches. This is scoped tracking, not global live CUDA handles. No post-unload
global native-resource gauge was taken.

Unique mapped allocator pool bytes remained 2,543,845,376; quarantine zero.
CPU source bindings zero/file-backed bindings 1,152. Faults, temporary fallback,
execution mismatch, cache admission denial/bypass and pressure eviction zero.
All memory monitor error fields null. Standby purge skipped safely due to
missing process privilege; existing Diffusers tracing/deprecation warnings were
not native errors. Source and registration ceilings respected.

Transformer offload preparation 0.7483 s; transformer load separately 0.3657 s.
Whole Generate 48.0874 s includes THREE denoises and shared setup/cleanup, not
one eight-step sampler. Text encode phase 20.2336 s, VAE phase 1.5945 s, total
69.9961 s. Peak GPU delta from baseline 6.3214 GiB (not absolute GPU use), peak
RSS 29.5053 GiB (not USS/shared WDDM).

Effective staging clarification: CLI 0.5 GiB denotes a slot; the unchanged host
conversion supplies THREE slots, 1.5 GiB total capacity to DDO. Public report
and native staging accounting confirm 1,610,612,736 bytes. The previous reuse
gate used the same conversion. No runtime parameter changed during this run.

Candidate hash verified before/after:
`755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729`.
Protected default `4AC3D8B...` and milestone `8CAAD072...` hashes unchanged.
Prior milestone reuse failure remains historical FAIL; this passing candidate
observation does not prove the hardening caused a speedup or establish variance.
No default promotion, rebuild, benchmark rerun or budget recalibration.

Artifact in sibling host:
`outputs/ltx_image_modular_vmm/hardened_reuse_gate/metrics/ltx23_image_distilled_modular_vmm_bf16_vmm4gb_text_encoder_original_nopag_1280x704_steps8_seed43_ddo_public_cache24.625gb_registered24.625gb.json`.

The latest bridge corrections passed 70 CPU contracts and six small CUDA cases:
required DLL diagnostics, failed completion owner retention, partial unpin and
look-ahead cancellation progress, and upload first-error diagnostics. Cooperative
interrupt/resume/unload and 128-forward tiny reload gates passed separately.
Real driver poisoning, second live context/multi-GPU and full-model cancellation
remain outside proved support. This checkpoint does not close the entire roadmap
or approve text-encoder VMM migration, compatibility removal or publication.

## Proposed real-workload validation

One fresh bootstrap process, three eight-step denoise generations, real prompt,
BF16, 1280x704, seed43. Use DDO `.venv-vmm`, sibling LTX harness and local caches.
Only the transformer uses native VMM; text encoder keeps its existing route and
runs once. The generator advances across generations and only the final latent
is decoded, so saved-image parity across repeats is not established.

Same benchmark limits/monitoring as the prior generation-reuse gate: 4 GiB
weight ceiling, 512 MiB allocator cache, 0.5 GiB per staging slot (1.5 GiB total), source storage and
registration ceilings 24.625 GiB, RSS monitor and public GPU step timing. No
extra collection between generations or disabled sampling. Lifecycle OFF DLL
is the candidate, not a promoted replacement for the span-pool milestone.
Its native policy remains span-pool ON/source-dispatcher OFF; added event-owner
exports permit scoped resource observations. Hash checked before execution.

Hypothesis: safety/diagnostic bridge changes preserve throughput and same-model
source reuse without growing native resource ownership. This is a controlled
candidate check, not an inference that small-gate timing predicts LTX timing.

## Executed command (one process, stop on failure/anomaly)

```powershell
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$ddoPython='E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
$hardenedReuseHash=(Get-FileHash -LiteralPath $env:DDO_VMM_LIBRARY_PATH -Algorithm SHA256).Hash
if ($hardenedReuseHash -ne '755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729') { throw 'Unexpected DLL hash; stop gate' }
$env:DDO_VMM_TORCH_CACHE_MIB='512'
$env:HF_HOME='F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& $ddoPython run_dynamic_modular_vmm_distilled.py `
  --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto `
  --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 `
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --width 1280 --height 704 --seed 43 --steps 8 --generation-repeats 3 `
  --no-pag --local-files-only --metrics-level 2 `
  --memory-monitor rss --ddo-gpu-timing `
  --output-dir outputs/ltx_image_modular_vmm/hardened_reuse_gate
if ($LASTEXITCODE -ne 0) { throw 'Hardened reuse gate failed; stop' }
```

## Acceptance and interpretation

- Three complete eight-step CUDA reports, warm <=1.46 s/it for EACH generation,
  unchanged historical strict criterion. The prior milestone reuse run failed
  generation1 at 1.473632 and passed generations2/3 at 1.437565/1.408667.
  Do not convert that earlier failure into PASS or silently relax the criterion.
- No source reread after generation1; committed/registered storage and
  registrations stable after generation1, within approved ceilings. DMA remains
  necessary and cache reuse does not prove WDDM migration or GPU residency.
- Compare live/scoped event ownership at equivalent completion boundaries.
  Pooled events <=256; cumulative created events and pending metadata are not
  sufficient leak evidence. Final public completion collection retires tickets
  to zero. Do not insert per-generation cleanup to conceal growth.
- No native/monitor error, unexpected fallback/quarantine or memory anomaly.
  GPU peak delta is not absolute GPU use; RSS is not USS or shared WDDM usage.
- Report transformer preparation, first step, warm steps, CUDA denoise phase,
  readback-complete generation wall, whole Generate and shared text encoding
  separately. Do not compare three denoises with a one-generation sampler or
  add nested readiness/registration timers as independent costs.

Approval covers ONE specified process, no rerun, changed limits or different
candidate. Any failed or inconclusive criterion remains failed/inconclusive.
No automatic promotion even on PASS; protected default and performance milestone
DLLs remain untouched. Record actual results in the compact experiment sidecar.
