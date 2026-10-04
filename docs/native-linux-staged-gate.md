# Linux staged model gate — PASS (2026-10-04)

The user confirmed the consolidated sequence below; all three cache contracts
and the single staged real-model generation passed outside the GPU-restricted
sandbox. No fresh probe/full test matrix/repeated generation/parameter tuning
was added. Future GPU execution requires its own confirmation.

## Executed result

Corrected library SHA256 matches the fixed manifest below. Three selected CUDA
contracts: **3 passed in3.10s**, no skips. Preflight: overcommit mode0,
MemAvailable56,991,768kB, GPU628MiB used and7212MiB free; output directory absent.
The one generation then completed exit0, using DDO .venv-linux, fresh allocator
bootstrap, offline local weights and native BF16 encoder/transformer routes.

| Measured scope | Linux | Accepted Windows reference |
| --- | ---: | ---: |
| Encoder load / attach | 0.6240 / 1.2323s | 0.7951 / 2.5477s |
| Encoder call including report/unload boundary | 7.7168s | 7.4341s |
| Entire prompt phase | 11.7280s | 12.6624s |
| Transformer load / prepare | 0.3953 / 0.7808s | 0.3965 / 0.6261s |
| First CUDA step | 5.788787s | 6.729011s |
| CUDA steady state, steps2–8 | 1.050970s/step | 1.395602s/step |
| CUDA phase, compute stream with dependencies | 13.145578s | 16.498256s |
| Entire Generate phase | 19.0590s | 24.2141s |
| VAE / save phases | 1.2131 / 0.2129s | 1.5570 / 0.0760s |
| RunTracker total, excluding bootstrap/import startup | 32.2142s | 38.5135s |

All eight CUDA intervals are complete, timing.failed=false. Host callback times
are not used as CUDA timings; nested prepare scopes must not be summed.
This single Linux observation is about24.7% lower steady-state time and16.4%
lower tracker time than the selected Windows reference, not a statistical or
cross-platform speed guarantee.

Sampled memory at100ms: baseline GPU0.722290GiB, peak baseline-relative
delta5.7773GiB, persisted absolute driver peak6.499573GiB, peak RSS27.3482GiB.
All phase monitor error_type fields are null. Both sampled acceptance bounds
(7.5GiB GPU /40GiB RSS) pass; these are not instantaneous enforced caps.

Native ownership/cache observations:

- Encoder1065 parameter groups, six buffer groups/34818 bytes,82 execution
  units, lookahead disabled; source committed/registered21.9375GiB.
- Encoder tickets31→0; before transformer denoise, source committed,
  registered, process committed/registered and live tickets are all zero.
- Transformer source committed/registered24.25GiB, below24.625GiB; staging
  registration1.5GiB remains separate (aggregate25.75GiB below26.125GiB).
  Tickets411→0 after explicit completion collection.
- Transformer source read25,846,148,360 bytes versus187,588,136,576 DMA bytes;
  cache10780 hits/388 misses. This is aggregate reuse evidence, not a per-step
  file-read trace. No source-read failures, cache/batch bypass, admission denials,
  ring fallback or cache host-copy bytes; no temporary tensors, fault/admission
  failures, execution-plan mismatches, allocator failures or quarantine bytes
  in either component's completion report.

Image inspected: plausible cat/flower crown/park,1280×704. This does not prove
pixel equality or numerical encoder equivalence across platforms.

Artifacts in the sibling host:
`outputs/ltx_image_modular_vmm/native_linux_staged_gate_20261004/`.
Stem: `ltx23_image_distilled_modular_vmm_bf16_vmm4gb_text_encoder_native_nopag_1280x704_steps8_seed43_ddo_public_cache24.625gb_registered24.625gb`.
PNG SHA256: dfbf135a833b5a77f4574a98cebd3be84d19fc569b79de8dd5f18457c52d64c1.
JSON in metrics/, SHA256:07c14628d7982f2319ec5f9b37ea9684c1d0c7a09f00910221e97d6fbdc707f4.
No source/parameters changed during execution. No additional GPU runs, Windows
regression benchmark, default promotion, commit/push or history cleanup.
Linux evidence covers this selected staged text-only workload, not arbitrary
models, multimodal inputs, driver-loss recovery, WSL or macOS.

## Fixed manifest and preflight

Use DDO .venv-linux, the existing sibling modular harness and fresh-process
bootstrap. Native BF16 text encoder → unload → native BF16 transformer → VAE/save.
One generation, real default cat prompt,1280x704,seed43,eight steps,no PAG.
Retain512MiB global allocator cache,4GiB native active weight budget,1.5GiB
total staging (host CLI0.5GiB x3),24.625GiB source/registration ceilings.
No model-specific DDO code or Linux kernel settings are changed.

Actual readable local snapshot:
`/run/media/eliseu/IA/hf_cache/hub/models--elismasilva--ltx2.3-image-distilled-1.1/snapshots/5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9`.
Read only safetensor headers in preparation: four transformer shards,
26,248,632,576 payload bytes; one encoder shard,24,374,650,080 payload bytes.
These raw checkpoint totals are NOT measured selected native bindings/commit;
the encoder includes payload outside the validated text-only route.
The home-directory HF cache has only encoder material; don't use it as if complete.
Keep host .hf_modules for any dynamic module cache; installed pinned Diffusers
already exposes this transformer's class and its config has no auto_map.
Do not replace loaders/pipeline or download a new checkpoint silently.

Preparation measured mode0 with54.5GiB available host RAM; corrected admission
retains finite RAM/container/process limits. Values must be checked again before
the model; they do not reserve capacity against competing processes. Minimum
full-profile host demand27.125GiB (source + staging + native default margin),
plus live component/loader/driver demands. Stop for insufficient headroom,
registration denial, bypass, anomalous memory or a failed small gate; don't
reduce capacity, increase limits or tweak sysctl to force execution.

Library SHA256 a87979f8773a010ddc2c781a37f3ef55112ca558a9d93083b972f51246a2b245.
Span pool ON, dispatcher/hooks/probes OFF; Windows artifacts preserved.

## Approved commands executed

From DDO, first run ONLY the three source/cache contracts affected by admission:

```bash
export DDO_VMM_LIBRARY_PATH="$PWD/ddo-vmm-allocator/build-linux/libddo_vmm.so"
export DDO_VMM_TORCH_CACHE_MIB=512
.venv-linux/bin/python -m pytest \
  ddo-vmm-allocator/tests/test_source_cache.py::test_persistent_storage_survives_registration_recycling_without_reread \
  ddo-vmm-allocator/tests/test_source_cache.py::test_immutable_batch_retains_sources_recycles_events_and_guards_close \
  ddo-vmm-allocator/tests/test_source_cache.py::test_live_tickets_protect_cache_close_and_clear \
  -q -x -p no:cacheprovider
```

After all three PASS with no skips, verify library hash, available host/driver
memory and that the output directory below does not exist. Do not launch after
failed preflight; stop and report. Then ONE model command in the sibling host:

```bash
cd /run/media/eliseu/AI2/ProjetosIA/ltx2.3-image-diffusers
export HF_HOME=/run/media/eliseu/IA/hf_cache
export HF_MODULES_CACHE="$PWD/.hf_modules"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/run/media/eliseu/AI2/ProjetosIA/diffusers-dynamic-offloader/.venv-linux/bin/python \
  run_dynamic_modular_vmm_distilled.py \
  --model-path /run/media/eliseu/IA/hf_cache/hub/models--elismasilva--ltx2.3-image-distilled-1.1/snapshots/5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9 \
  --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto \
  --text-encoder-native-vmm --no-fake-prompt \
  --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 \
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 \
  --width 1280 --height 704 --seed 43 --steps 8 --generation-repeats 1 \
  --no-pag --local-files-only --metrics-level 2 --save-metrics \
  --memory-monitor rss --ddo-gpu-timing \
  --output-dir outputs/ltx_image_modular_vmm/native_linux_staged_gate_20261004
```

## Hypothesis and acceptance

The corrected guard permits an integral retained-source profile within actual
Linux capacity while preserving native transfer/events and staged ownership.
Require the three existing numerical/registration/source/batch ownership gates
PASS first; then complete native encoder/transformer phases, eight CUDA step
intervals, correct-size visually plausible output, reports saved, no unexpected
fault/fallback/quarantine, source reread after warm-up, admission denial/bypass,
execution mismatch, collected transfer owners or monitor errors. Inspect source
process counters at the encoder/transformer boundary for no retained encoder
population and at transformer completion for configured ceilings. Sources remain
bounded; absence of a historical field is unavailable, not zero.

Keep the Windows sampled safety checks: peak RSS<=40GiB and sampled absolute
driver GPU use<=7.5GiB (use persisted baseline + delta/absolute fields, not delta
alone). These are sampled observations, not instantaneous enforced peak caps.
Stop on a failure or visible memory anomaly; no retry/alternate manifest.

Record load/setup/cold-step/warm/phase/total independently. Windows warm
1.395602s/it and tracker38.5135s are comparison references, not a Linux speed
guarantee; do not relax a claimed performance acceptance retrospectively. If
Linux is slower, record that rather than claiming the performance target passed.
Do not assert pixel equality across platforms or numerical encoder equivalence
solely from plausible images. No arbitrary model/multimodal/driver-loss support,
Windows regression benchmark, default promotion, commit/push or history cleanup.
