# Dynamic Offload Experiment Notes

## Executive Summary

All headline comparisons below use the same LTX 2.3 distilled image workload unless noted: BF16, real prompt encoding, `1280x704`, `8` steps, seed `43`, native attention.

| Environment / comparison | Baseline | DDO / staged result | Total latency change | Denoise change | Peak RAM change | Management readout |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Windows modular, DDO vs official Diffusers block-level group offload | `315.1s` total, `222.88s` denoise, `28.98 GB` RAM | `133.2s` total, `14.83s` denoise, `27.38 GB` RAM | `57.7%` lower total time (`2.37x` faster) | `93.3%` lower denoise time (`15.03x` faster) | `5.5%` lower peak RAM | DDO is a major win on this Windows low-VRAM setup when enough system RAM is available for pinned CPU weights. |
| Windows old full pipeline vs staged traditional pipeline | `423.6s` total, `51.71 GB` RAM | `126.8s` total, `28.89 GB` RAM | `70.1%` lower total time (`3.34x` faster) | Not directly comparable because the full pipeline hides internal phases | `44.1%` lower peak RAM | Staging old-style Diffusers pipelines matters: it gives DDO cleanup boundaries between components. |
| Windows staged traditional pipeline, DDO vs official Diffusers block-level group offload | `321.4s` total, `255.39s` denoise, `33.83 GB` RAM, `4.45 GB` VRAM | `126.8s` total, `29.70s` denoise, `28.89 GB` RAM, `6.97 GB` VRAM | `60.5%` lower total time (`2.53x` faster) | `88.4%` lower denoise time (`8.60x` faster) | `14.6%` lower peak RAM; VRAM `56.6%` higher | Same old-style staged flow, so this is the cleanest non-modular DDO-vs-Diffusers comparison. DDO trades more VRAM for much lower latency and lower host RAM. |
| WSL modular, DDO `wsl_compat` vs official Diffusers block-level group offload | `218.1s` total | `144.4s` total | `33.8%` lower total time (`1.51x` faster) | DDO denoise `97.80s` vs Diffusers `172.00s` | DDO peak RAM `23.47 GB` vs Diffusers `38.39 GB` | WSL benefits from DDO's dynamic path with pinning disabled, but remains slower and less stable than native Linux/Windows. |
| Native Ubuntu modular, DDO `one_shot_fast` vs best recorded official Diffusers group offload | `34.9s` total for Diffusers leaf group offload | `59.3s` total for DDO | DDO was `69.9%` slower in this recorded one-shot run | DDO denoise was `45.9%` faster (`13.06s` vs `24.14s`), but setup/encode made total latency worse | Similar recorded RAM | Native Ubuntu is not a DDO one-shot win in the recorded baseline; official Diffusers leaf group offload was strongest overall. |
| Quantized SDNQ Windows, offload off vs Diffusers leaf group offload | `250.8s` total, `21.46 GB` RAM | `138.5s` total, `16.80 GB` RAM | `44.8%` lower total time (`1.81x` faster) | `92.82s` denoise with group offload vs `189.00s` without | `21.7%` lower peak RAM | Quantized backends should preserve their own forward logic; prefer Diffusers group-offload compatibility presets for SDNQ. |

High-level conclusion: DDO's dense-linear streaming path is most valuable on Windows low-VRAM BF16 workloads where official Diffusers group offload is dominated by repeated transfers. For native Linux and quantized backends, DDO should act as a routing/preset layer and preserve official Diffusers or backend-specific paths when those are faster or safer.

### Evidence Coverage

Closed, apples-to-apples comparisons:
- Windows modular BF16: DDO `auto -> one_shot_fast` vs official Diffusers group-offload compatibility presets.
- WSL modular BF16: DDO `wsl_compat` vs official Diffusers block-level group offload.
- Native Ubuntu modular BF16: DDO `one_shot_fast` vs official Diffusers block/leaf group offload.
- Windows SDNQ int8: no transformer offload vs official Diffusers leaf group offload.
- Windows staged traditional pipeline BF16: DDO `one_shot_fast` vs official Diffusers block-level group offload in the same staged runner.

Exploratory comparisons, useful but not headline product claims:
- Old full pipeline vs old staged pipeline: shows the value of staging/cleanup boundaries, not a pure DDO-vs-Diffusers comparison.
- Simulated 32 GB RAM: useful planner behavior probe, but not a substitute for a physical 32 GB machine.
- Warm-process/server mode: useful for cache behavior, but totals include multiple generation repeats and should be reported separately from one-image latency.

Numbers still worth adding before sharing broadly:
- Optional WSL/native Ubuntu runs for `run_dynamic_old_staged_distilled.py`, only if we want old-pipeline portability claims.
- Quantized SDNQ on WSL/native Ubuntu, if quantized portability becomes part of the Diffusers discussion.

Baseline unless noted:
- Model: LTX 2.3 distilled image modular runner
- Resolution: 1280x704
- Steps: 8
- Seed: 43
- Dtype: bfloat16
- Prompt mode: fake prompt for transformer-only comparisons
- Attention backend: native
- Windows standby purge before transformer enabled for Windows tests

## Windows Transformer Comparisons

| Scenario | Preset / route | RAM policy | Setup seconds | Denoise seconds | Copy seconds | Peak VRAM | Peak RAM | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dynamic offload, enough RAM | `one_shot_fast` | detected available RAM 46.84 GB, full pin selected | 34.67 | 14.38 | 0.56 | 6.97 GB | 25.15 GB | Correct fast path: `snap_to_full_pin`, 444 patched modules, 11 resident blocks. |
| Dynamic offload, simulated 32 GB RAM | `one_shot_fast` | pin skipped, insufficient RAM | 7.72 | 198.2 total pass, denoise dominated by copies | 174.12 | 6.97 GB | 25.87 GB | Correct safe path: avoids pinning when usable RAM is below required RAM. |
| Low RAM fallback | `low_ram_safe` | no pin, smaller resident budget | 3.78 | 194.10 | 187.81 | 3.28 GB reserved during denoise | not captured in pasted slice | Lower VRAM, but slower than `one_shot_fast` simulated 32 GB because fewer resident modules and more runtime copies. |
| Diffusers group offload | `off` + transformer `leaf_level`, stream, record stream, low CPU mem usage off | official Diffusers group offload | 0.07 group setup | 315.29 | not tracked by dynamic offload | 6.49 GB | 26.60 GB | Very compatible/official path, but much slower in this low-VRAM 8-step transformer scenario. |
| Diffusers group offload | `off` + transformer `leaf_level`, stream, record stream, low CPU mem usage on | official Diffusers group offload | 0.05 group setup | 320.10 | not tracked by dynamic offload | 6.56 GB | 26.60 GB | Same shape as low CPU off; setup is tiny, but denoise remains copy/offload bound. |
| Diffusers group offload | `off` + transformer `leaf_level`, no stream, no record stream, low CPU mem usage on | official Diffusers group offload | 0.03 group setup | 507.83 | not tracked by dynamic offload | 6.58 GB | 39.37 GB | Worst official offload probe so far; disabling stream made denoise much slower and increased peak RAM. |
| Diffusers group offload | `off` + transformer `block_level`, `num_blocks_per_group=1`, stream, record stream, low CPU mem usage on | official Diffusers group offload | 0.02 group setup | 265.47 | not tracked by dynamic offload | 6.49 GB | 26.79 GB | Better than leaf, but still far from dynamic offload fast path. `torch_alloc` stayed near 0.03 GiB while reserved grew to 2.11 GiB. |
| Diffusers group offload | `off` + transformer `block_level`, `num_blocks_per_group=1`, stream, no record stream, low CPU mem usage on | official Diffusers group offload | 0.01 group setup | 312.03 | not tracked by dynamic offload | 6.51 GB | 26.78 GB | Worse than record stream on; reserved VRAM was lower, but denoise returned to leaf-level timing. |

## Current Interpretation

`one_shot_fast` is the default general preset. It pins CPU weights only when the measured or supplied available RAM can cover the model weight copy, resident GPU modules, and configured system headroom. On machines with enough RAM, it pays setup time once and keeps denoise fast. On constrained RAM, it chooses safety over speed.

`low_ram_safe` is not the recommended 32 GB preset for this model. It is a fallback for tighter VRAM cases where the user accepts slow denoise to reduce accelerator memory pressure.

The Diffusers group offload path is useful as an official compatibility baseline, but in this test it is not close to
the dynamic offload fast path. `block_level` with one block per group and `record_stream=1` was the best official
offload probe, but it still remained dominated by repeated offload movement. `leaf_level` without stream is not viable
for this model/shape.

## Preset Direction

Recommended defaults:
- `auto` -> `one_shot_fast`
- `one_shot_fast` -> balanced planner, RAM-aware pinning
- `low_ram_safe` -> explicit fallback only
- `wsl_compat` -> explicit WSL fallback when stream/pin behavior is unstable
- `warm_process` -> process-lifetime cache comparisons and server-like usage
- `diffusers_offload_compat` -> official Diffusers group-offload compatibility fallback
- `diffusers_leaf_offload_compat` -> official Diffusers leaf-level fallback for models where finer granularity may fit better

Inspect the current preset contract without loading model weights:

```powershell
$env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS="1"
python run_dynamic_modular_distilled.py
Remove-Item Env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS -ErrorAction SilentlyContinue
```

Closed report sections:
- one-shot setup cost vs denoise throughput
- RAM-aware pinning behavior
- Windows standby cache impact
- Diffusers official group offload baseline
- Windows, WSL, and native Ubuntu validation matrix
- Quantized model compatibility risks and follow-up results


## Platform Validation Runs

These runs use real prompt encoding unless noted. Keep width `1280`, height `704`, steps `8`, seed `43`, BF16, native attention, and generation repeats `1` fixed across platforms.

| Platform | Preset / route | Prompt | Platform cleanup | Encode pass | Transformer setup | Denoise | Pass 1 | VAE | Total | Peak VRAM | Peak RAM | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Windows | `auto -> one_shot_fast` | real | standby purge before run, after text encoder, before transformer | `72.0s` | `29.76s` DDO setup, `21.47s` pin, `7.80s` resident modules | `14.83s` | `52.6s` | `4.9s` | `133.2s` | `6.45 GB` | `27.38 GB` | Run 1 after package split. Planner detected `43.53 GB` available RAM and selected `snap_to_full_pin` for `18.5181 GB`; runtime copy `0.3153s / 148.1445 GB`. |
| Windows | `auto -> one_shot_fast` | real, simulated 32 GB RAM | standby purge before run, after text encoder, before transformer | `79.6s` | `8.43s` DDO setup, pin skipped, `7.93s` resident modules | `114.70s` | `132.2s` | `2.6s` | `217.2s` | `6.41 GB` | `28.22 GB` | RAM-aware safety path. Planner used `available_system_ram_gb=32.0`, usable `24.0 GB`, required `30.0234 GB`, selected `skip_insufficient_ram`; runtime copy rose to `110.1362s / 148.1445 GB`. |
| Windows | `diffusers_offload_compat` | real | standby purge before run, after text encoder, before transformer | `76.5s` | `0.066s` Diffusers group-offload setup | `222.88s` | `232.9s` | `2.5s` | `315.1s` | `6.09 GB` | `28.98 GB` | Official Diffusers `block_level`, `num_blocks_per_group=1`, stream and record stream. First denoise step `83.68s`, later steps around `19-20s`; much slower than DDO pinned fast path but compatible baseline. |
| Windows | `diffusers_leaf_offload_compat` | real | standby purge before run, after text encoder, before transformer | `77.9s` | `0.030s` Diffusers group-offload setup | `233.39s` | `242.7s` | `2.7s` | `326.4s` | `6.21 GB` | `28.79 GB` | Official Diffusers `leaf_level`, stream and record stream. First denoise step `86.75s`, later steps around `20-21s`; slightly slower than block-level fallback in this Windows run. |
| Windows | `low_ram_safe` | real | standby purge before run, after text encoder, before transformer | `81.4s` | `4.00s` DDO setup, no pin, `3.36s` resident modules | `136.49s` | `149.6s` | `2.5s` | `236.5s` | `6.11 GB` | `28.22 GB` | Explicit VRAM-pressure fallback. Denoise torch allocation stayed near `2.96 GiB`, but more modules were patched (`516`) and runtime copy rose to `132.5071s / 172.1680 GB`; slower than simulated-32GB `auto` for this model. |
| WSL Ubuntu | `low_ram_safe` | real | no Windows standby purge | `36.2s` | `1.77s` DDO setup, no pin, `3 GB` resident budget | `122.88s` | `131.6s` | `2.0s` | `170.4s` | `5.99 GB` | `23.56 GB` | Corrected WSL low-RAM route: text encoder now uses Diffusers `leaf_level` group offload with stream disabled, avoiding the earlier `device not ready` connector failure. Transformer stays low VRAM around `2.96 GiB` allocated, but denoise is copy-bound without pinned CPU weights. |
| WSL Ubuntu | `wsl_compat` | real | no Windows standby purge | `36.2s` | `3.66s` DDO setup, no pin, `6 GB` resident budget | `97.80s` | `105.8s` | `1.8s` | `144.4s` | `6.30 GB` | `23.47 GB` | Stable WSL compatibility run after text-encoder routing fix. Text encoder uses non-streaming Diffusers group offload; transformer uses DDO dynamic offload with pin disabled. Denoise steps were consistent around `11.8-12.5s`, slower than Windows full-pin but faster than `low_ram_safe` and Diffusers group offload on WSL. |
| WSL Ubuntu | `diffusers_offload_compat` | real | no Windows standby purge | `35.8s` | `0.012s` Diffusers group-offload setup | `172.00s` | `180.1s` | `1.8s` | `218.1s` | `6.10 GB` | `38.39 GB` | Official Diffusers `block_level` group offload baseline on WSL. First denoise step `39.36s`, later steps around `18-19s`; faster than the Windows Diffusers block-level run but still slower than DDO WSL dynamic-offload fallbacks for this model/shape. |
| WSL Ubuntu | `diffusers_leaf_offload_compat` | real | no Windows standby purge | not completed | `0.011s` Diffusers group-offload setup | not completed | not completed | not completed | not completed | not captured | not captured | Official Diffusers `leaf_level` group offload loaded and prepared, then the WSL process exited/crashed immediately after `Starting denoise loop` before the first denoise callback. Treat as unstable on this WSL stack. |
| Native Ubuntu | `auto -> one_shot_fast` | real | none | `19.7s` | `19.30s` DDO setup, full-pin fast path | `13.06s` | `37.6s` | `1.6s` | `59.3s` | `6.29 GB` | `1.45 GB` | Native Linux fast DDO path. Denoise steps stayed around `1.61s`; transformer setup is the main remaining cost. |
| Native Ubuntu | `low_ram_safe` | real | none | `32.3s` | `2.61s` DDO setup, no pin, `3 GB` resident budget | `37.47s` | `48.4s` | `1.6s` | `82.7s` | `5.98 GB` | `23.45 GB` | Low-RAM fallback completed. First step paid `20.16s`; later steps were around `2.45s`, giving a much better Linux low-RAM result than WSL. |
| Native Ubuntu | `diffusers_offload_compat` | real | none | `13.5s` | `0.014s` Diffusers group-offload setup | `27.16s` | `29.4s` | `0.6s` | `43.8s` | `5.96 GB` | `1.53 GB` | Official Diffusers `block_level` group offload is very strong on native Linux, with steady `3.36s` denoise steps. It beats DDO one-shot total here because setup and encode are much lower. |
| Native Ubuntu | `diffusers_leaf_offload_compat` | real | none | `7.8s` | `0.026s` Diffusers group-offload setup | `24.14s` | `26.2s` | `0.6s` | `34.9s` | `5.96 GB` | `1.45 GB` | Fastest native Ubuntu run so far. Official Diffusers `leaf_level` completed cleanly and outperformed block-level for this platform run. |
| Native Ubuntu | `auto -> one_shot_fast`, simulated `32 GB` RAM | real | none | `7.8s` | `0.73s` DDO setup | `18.38s` | `21.4s` | `0.6s` | `30.0s` | `6.24 GB` | `1.48 GB` | Simulated 32 GB RAM decision on native Linux. This run was unusually fast and likely benefited from OS/file/cache warmth after earlier tests; keep as warm/native Linux data, not as cold one-shot baseline. |
| Native Ubuntu | `warm_process` | real | none | `19.7s` | prepare 1 `20.04s`, prepare 2 `0.90s` after cache | repeat 1 `12.98s`, repeat 2 `12.89s` | `52.4s` | `1.5s` | `73.9s` | `6.27 GB` | `1.43 GB` | Corrected warm-process route completed. Text encoder uses Diffusers group offload; transformer dynamic offload benefits from process-lifetime cache on the second prepare. Total includes two generation repeats, so this is a server/warm comparison rather than one-image latency. |


## Quantized SDNQ Validation

These runs use the SDNQ int8 LTX 2.3 image transformer path with the same runner, resolution `1280x704`, steps `8`, seed `43`, real prompt encoding, and text encoder Diffusers leaf group offload. The goal is memory behavior and compatibility, not DDO dense-linear acceleration.

| Platform | Preset / route | Transformer behavior | Encode pass | Transformer setup/load | Denoise | Pass 1 | VAE | Total | Peak VRAM | Peak RAM | Denoise allocation | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Windows | `off` | No transformer offload; transformer moved to CUDA/shared GPU memory | `32.3s` | `load_transformer_to_cuda 17.86s` | `189.00s` | `214.9s` | `1.9s` | `250.8s` | `6.97 GB` | `21.46 GB` | `torch_alloc 12.63 GiB`, `reserved 13.10 GiB` | Lower RAM than BF16, but denoise spills beyond 8 GB dedicated VRAM into shared GPU memory and remains slow/unstable. |
| Windows | `diffusers_leaf_offload_compat` | Official Diffusers leaf group offload with stream and record stream | `33.3s` | `setup_transformer_group_offload 0.038s` | `92.82s` | `101.3s` | `1.9s` | `138.5s` | `6.11 GB` | `16.80 GB` | `torch_alloc 0.15 GiB`, `reserved 0.96 GiB` | Best SDNQ memory/latency result in this pair. For quantized models, prefer Diffusers group-offload compatibility presets. |

Interpretation: SDNQ significantly reduces system RAM pressure versus BF16, but DDO dynamic linear streaming does not accelerate packed/quantized linears because their backend-specific forward must be preserved. For SDNQ on this Windows machine, Diffusers leaf group offload is the recommended compatibility path so far.

## Quantized Follow-Up

SDNQ int8 now has an initial Windows compatibility/memory baseline. The next quantized work should compare the same SDNQ pair on WSL and native Ubuntu only if needed, then add other backends such as bitsandbytes, TorchAO, or Quanto when matching models are available.

For quantized backends, fixes should remain model-agnostic and preserve backend-specific forward logic. DDO dynamic linear streaming should only patch standard dense `nn.Linear` weights with the expected 2D `(out_features, in_features)` layout.

## Validation Matrix

Use these as the next closed test ladder. Keep Windows as the development gate; repeat WSL/Ubuntu only after a Windows
change produces a useful result.

Windows baseline:

```powershell
$env:DDO_PRESET="auto"
$env:DDO_RUNNER_WIDTH="1280"
$env:DDO_RUNNER_HEIGHT="704"
$env:DDO_RUNNER_STEPS="8"
$env:DDO_RUNNER_SEED="43"
$env:DDO_RUNNER_GENERATION_REPEATS="1"
$env:DDO_RUNNER_FAKE_PROMPT="0"
$env:DDO_RUNNER_METRICS_LEVEL="1"
$env:DDO_SHOW_PROFILE="0"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_RUN="1"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER="1"
$env:DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER="1"
python run_dynamic_modular_distilled.py
```

Windows RAM-constrained planner check:

```powershell
$env:DDO_AVAILABLE_SYSTEM_RAM_GB="32"
python run_dynamic_modular_distilled.py
Remove-Item Env:DDO_AVAILABLE_SYSTEM_RAM_GB -ErrorAction SilentlyContinue
```

Linux/Ubuntu baseline:

```bash
export DDO_PRESET=auto
export DDO_RUNNER_WIDTH=1280
export DDO_RUNNER_HEIGHT=704
export DDO_RUNNER_STEPS=8
export DDO_RUNNER_SEED=43
export DDO_RUNNER_GENERATION_REPEATS=1
export DDO_RUNNER_FAKE_PROMPT=0
export DDO_RUNNER_METRICS_LEVEL=1
export DDO_SHOW_PROFILE=0
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_RUN
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER
unset DDO_RUNNER_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER
python run_dynamic_modular_distilled.py
```

WSL compatibility fallback:

```bash
export DDO_PRESET=wsl_compat
export DDO_RUNNER_WIDTH=1280
export DDO_RUNNER_HEIGHT=704
export DDO_RUNNER_STEPS=8
export DDO_RUNNER_SEED=43
export DDO_RUNNER_GENERATION_REPEATS=1
export DDO_RUNNER_FAKE_PROMPT=0
export DDO_RUNNER_METRICS_LEVEL=1
export DDO_SHOW_PROFILE=0
python run_dynamic_modular_distilled.py
```

Official Diffusers fallback comparison:

```powershell
$env:DDO_PRESET="diffusers_offload_compat"
python run_dynamic_modular_distilled.py
```
## Traditional Diffusers Pipeline Probes

These probes compare DDO in non-modular LTX2ImagePipeline flows. The full-pipeline variant validates compatibility with old-style Diffusers pipelines, but it cannot insert cleanup between internal text-encoder, transformer, and VAE phases.

| Platform | Runner | Preset / route | Component policy | Load / setup | Pipeline call / denoise | Total | Peak VRAM | Peak RAM | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Windows | `run_dynamic_old_distilled.py` | `auto -> one_shot_fast` | Full `LTX2ImagePipeline`; text encoder, connectors, and VAE use Diffusers leaf group offload; transformer uses DDO dynamic offload | `25.0s` load/setup; transformer dynamic setup `21.2353s` | `392.3639s` pipeline call; first denoise callback delayed `189.48s`, later steps unstable | `423.6s` | `6.97 GB` | `51.71 GB` | Functional but not viable for low-RAM one-shot use. Because the old full pipeline owns all phases internally, DDO cannot purge/flush between prompt encoding, transformer, and VAE decode. Keep this as a compatibility baseline; use staged or modular runners for controlled memory cleanup. |
| Windows | `run_dynamic_old_staged_distilled.py` | `auto -> one_shot_fast` | Staged traditional pipeline; text encoder uses Diffusers leaf group offload, connectors use Diffusers leaf group offload inside the denoise pipeline, transformer uses DDO dynamic offload, VAE decoded in a separate manual stage | Pass 0 `55.6s`; transformer dynamic setup `23.5855s`; Pass 1 setup/build/cleanup included `58.2s` | `29.7005s` denoise pipe call; first callback `15.66s`, later steps around `1.8s` | `126.8s` | `6.97 GB` | `28.89 GB` | Good controlled-memory comparison. Releasing DDO result references before cleanup fixed the earlier `~50 GB` RAM retention. Still slower than the best modular run because connectors execute inside the denoise pipeline and old-pipeline overhead remains. |
| Windows | `run_dynamic_old_staged_distilled.py` | `diffusers_offload_compat` | Staged traditional pipeline; text encoder, connectors, and transformer use official Diffusers group offload | Pass 0 `51.9s`; transformer group-offload setup `0.0711s`; Pass 1 `258.5s` | `255.3886s` denoise pipe call; first callback `73.78s`, later steps around `25-26s` | `321.4s` | `4.45 GB` | `33.83 GB` | Apples-to-apples staged old baseline. It uses less VRAM than DDO staged, but is `2.53x` slower overall, `8.60x` slower in denoise, and uses more host RAM. |
