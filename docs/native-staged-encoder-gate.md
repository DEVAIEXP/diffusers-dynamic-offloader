# Phase-staged native encoder integration (PASS, 2026-10-03)

The actual encoder-only numerical gate passed the selected real prompt and
two calls, with unchanged limits/DLL. The host now offers explicit
`--text-encoder-native-vmm`, default false. Requires public native runtime and
real prompt; conflicting explicit compatibility controls are rejected.

DDO owns loaded-source inspection, native ownership/fault/transfer/lifecycle.
The host only selects the component config, constructs the existing prompt
pipeline and orchestrates phases. Encoder config uses full selection, page2MiB,
bounded buffers1MiB, no transformer profile calibration; all resource ceilings
are inherited unchanged. Transformer config/attachment and default encoder
compatibility remain untouched. Encoder reports use the public metrics object.
Its unload runs in finally around encoding; cleanup errors preserve the original
forward error. Successful output readback, model/pipeline deletion and ordinary
cleanup occur before connectors or transformer load, not simultaneously.

Preparation:52 CPU contracts PASS in5.77s (host config/flag/phase ordering,
existing timing callback, component adapter, backend/profiles). Runner source
syntax compiled only; full CUDA pipeline NOT executed in this preparation.

## Executed consolidated full-pipeline execution

ONE fresh bootstrap process, ONE eight-step generation, existing real cat prompt,
BF16,1280x704,seed43,no PAG,local checkpoint/cache. Native encoder then transformer,
normal VAE decode/image save. This is a single-generation integration gate, not
the previous three-generation reuse benchmark; do not compare their whole
Generate/total times as equal workloads. No native encoder warm extra call.

Same candidate DLL SHA256
755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729.
Limits:4GiB active,512MiB global allocator cache,CLI staging0.5GiB per slot
converted to1.5GiB total,source/registration24.625GiB. Encoder page2MiB and
buffers1MiB; transformer page/selection/guards unchanged. RSS monitor0.1s,
public CUDA step timing retained. No default promotion or budget adjustment.

```powershell
Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$ddoPython='E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
if ((Get-FileHash -LiteralPath $env:DDO_VMM_LIBRARY_PATH -Algorithm SHA256).Hash -ne '755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729') { throw 'Unexpected DLL hash; stop gate' }
$env:DDO_VMM_TORCH_CACHE_MIB='512'
$env:HF_HOME='F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& $ddoPython run_dynamic_modular_vmm_distilled.py --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto --text-encoder-native-vmm --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --width 1280 --height 704 --seed 43 --steps 8 --generation-repeats 1 --no-pag --local-files-only --metrics-level 2 --memory-monitor rss --ddo-gpu-timing --output-dir outputs/ltx_image_modular_vmm/native_encoder_staged_gate_20261003
if ($LASTEXITCODE -ne 0) { throw 'Staged native encoder gate failed; stop' }
```

Acceptance requires checking saved public reports, not just process exit0:

- One complete8step CUDA report; transformer warm mean<=1.46s/it, unchanged
  strict criterion. First step/setup reported separately; no silent threshold
  relaxation. Total/phase times observed, not assigned an unsupported speedup.
- Correct1280x704 image saved and inspected; encoder numerical support comes
  from the preceding reference gate. No pixel-identical image claim against
  the earlier three-generation artifact (different advanced RNG state).
- Encoder1065 native file bindings/zero CPU source bindings; no source denial,
  bypass, unexpected fault/fallback/quarantine or monitor errors in either phase.
- Native process source committed/registered counters must remain within the
  fixed24.625GiB ceiling, including at the transformer boundary. No overlapping
  encoder/transformer cache population; compare process/model counters where
  available, without implying post-unload global driver gauges exist.
- Observed peak RSS<=40GiB and absolute driver GPU use<=7.5GiB (add reported
  baseline to GPU delta). These sampled checks are not instantaneous hard caps
  and do not quantify shared WDDM. Stop on any visible memory anomaly/failure.
- Normal lifecycle collection/cleanup completes; evaluate final ticket owners
  at the same existing public completion boundary, no added per-step cleanup.

No rerun, changed cases/ceilings, native cancellation injection, compatibility
removal, history cleanup, publication or DLL promotion is approved by this
execution scope. A failing performance criterion remains FAIL even if image completes.

## Executed outcome

The approved command ran once with DDO `.venv-vmm`, fresh bootstrap and the
unchanged candidate hash above; exit0. Public reports and saved PNG inspected:
**PASS for this single-generation staged integration gate**, not general
multimodal/long-run support or promotion of the default runtime.

| Measurement | Observed |
| --- | ---: |
| Encoder load | 0.7951 s |
| Encoder native attachment | 2.5477 s |
| Encoder call event | 7.4341 s |
| Entire encoder phase | 12.6624 s |
| Transformer load / prepare event | 0.3965 / 0.6261 s |
| First transformer CUDA step | 6.729011 s |
| Transformer warm CUDA mean, steps 2–8 | 1.395602 s/it |
| Transformer CUDA phase including dependencies | 16.498256 s |
| Prepare-latents through CPU latent readback | 16.517878 s |
| Entire Generate phase including connectors/setup | 24.2141 s |
| VAE phase / image save | 1.5570 / 0.0760 s |
| RunTracker total | 38.5135 s |
| Sampled peak RSS | 27.3479 GiB |
| Sampled peak GPU delta above baseline | 5.9403 GiB |

The printed GPU baseline was1.03GiB: sampled absolute driver use was approximately
6.97GiB, below7.5GiB. Baseline is rounded in console and not persisted in this
JSON; no exact absolute peak or shared-WDDM accounting is claimed. All four
phase monitors have null error_type. RSS remains below40GiB. RunTracker total
excludes interpreter/import/bootstrap startup; it is not process-launch time.
Encoder call event includes the host's native report/unload boundary, not a
separately measured pure GPU encode or explicit unload duration.

Encoder native file bindings1065/CPU0; source read/committed/registered
23,555,211,264 bytes; DMA25,545,825,792 bytes. Completion collection38→0 tickets,
zero transfer-owned events. At the public pre-denoise transformer snapshot,
process source committed/registered are both0; staging registration is
1,610,612,736 bytes. At transformer completion model/process committed and
registered match at26,038,239,232 bytes, below24.625GiB, confirming no retained
encoder cache population at those observed boundaries. Transformer source
read25,846,148,360 bytes, DMA187,588,136,576 bytes, file bindings1152/CPU0;
completion736→0 tickets, zero transfer-owned events. Source process total limit
28,051,505,152 bytes includes the separate1.5GiB staging allocation.
Fault/admission/read failures, temporary fallback, source denials/bypasses,
execution-plan mismatches and pool quarantine/failures are0. No post-unload
global driver gauge or leak-free arbitrary reuse claim.

PNG header1280x704 verified and image visually inspected: plausible cat/flower
crown/park matching the prompt, no visible corruption; not a pixel-equality test.
The report's top-level preset booleans still describe default group-offload
settings; actual `text_encoder` public report proves vmm_fast/component ownership.
Do not interpret those legacy preset booleans as the executed encoder route.
Standby purge safely skipped because this process lacks
SeProfileSingleProcessPrivilege; this is not a memory-monitor failure.
Protected default and candidate DLL hashes were checked unchanged afterwards.
No retry, DLL promotion, budget/default change or compatibility removal.

Artifacts in sibling host repository, under
`outputs/ltx_image_modular_vmm/native_encoder_staged_gate_20261003/`:
`ltx23_image_distilled_modular_vmm_bf16_vmm4gb_text_encoder_native_nopag_1280x704_steps8_seed43_ddo_public_cache24.625gb_registered24.625gb.png`
and the same stem in `metrics/*.json`.

Next: consolidated robustness/protected-route regression, then public reporting
and documentation closure. Additional CUDA cases need their own named command,
hypothesis, acceptance criteria and approval. Historical three-generation total
is a different workload and cannot establish a full-pipeline speedup here.
