# diffusers-dynamic-offloader

Model-agnostic dynamic offload helpers for Diffusers-style low-VRAM inference.

DDO is a Python compatibility-first layer: it can keep selected modules resident on the accelerator, stream large weights at runtime, optionally pin CPU tensors when system RAM allows it, and expose repeatable presets through `DDO_*` environment variables.

This repository was split from the LTX 2.3 modular image experiments so the offload manager can evolve independently from any single model family. The LTX runners and custom blocks stay in the host experiment repository and import this package with an editable install.

## Install For Local Experiments

From the host project environment:

```powershell
uv pip install -e E:\ProjetosIA\diffusers-dynamic-offloader
```

Then host code can import:

```python
from diffusers_dynamic_offloader import DynamicOffloadSettings, enable_offload
```

## Basic API

The shortest integration path is to let DDO load preset defaults and choose the correct route for each component:

```python
from diffusers_dynamic_offloader import enable_offload

transformer_offload = enable_offload(
    transformer,
    preset="auto",
    component="transformer",
)
transformer = transformer_offload.module
```

`preset="auto"` currently resolves to `one_shot_fast`. For the LTX 2.3 BF16 benchmark this means: keep selected transformer blocks resident, pin streamed CPU weights only when system RAM has enough headroom, and use Windows standby-list cleanup at the lifecycle points declared by the preset.

Use an explicit settings object only when the host wants to share one resolved configuration across components or override devices:

```python
from diffusers_dynamic_offloader import DynamicOffloadSettings, enable_offload

settings = DynamicOffloadSettings.from_env(
    execution_device="cuda:0",
    offload_device="cpu",
    default_preset="auto",
)

text_encoder = enable_offload(text_encoder, settings=settings, component="text_encoder").module
transformer = enable_offload(transformer, settings=settings, component="transformer").module
```

For `one_shot_fast`, `component="transformer"` currently routes to DDO dynamic offload. For `diffusers_offload_compat`, it routes to the official Diffusers group offload helper. Component-specific preset values such as `DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD=1` are handled inside DDO, so application code should not need to choose between DDO and Diffusers hooks manually.

Explicit keyword arguments override preset and environment values:

```python
enable_offload(
    transformer,
    settings=settings,
    component="transformer",
    max_resident_module_budget_gb=6.0,
    pin_cpu_workers=4,
)
```

The lower-level `enable_dynamic_offload(...)`, `enable_diffusers_group_offload(...)`, and `apply_dynamic_offload(...)` APIs remain available for experiments, but runners should prefer `enable_offload(...)`.

## Minimal Host Runner

The LTX host repository includes `run_dynamic_minimal.py` as a compact integration example. It should run with no DDO environment variables set:

```powershell
python run_dynamic_minimal.py
```

The script still allows optional overrides such as `DDO_PRESET=low_ram_safe`, but the default path is resolved by DDO itself through `DynamicOffloadSettings.from_env(default_preset="auto")` and `enable_offload(...)`.

## Environment Prefixes

Library settings use `DDO_*`.

Host-runner settings should use `DDO_RUNNER_*`.

```powershell
$env:DDO_PRESET="auto"
$env:DDO_RUNNER_WIDTH="1280"
$env:DDO_RUNNER_HEIGHT="704"
$env:DDO_RUNNER_STEPS="8"
$env:DDO_RUNNER_SEED="43"
$env:DDO_RUNNER_METRICS_LEVEL="1"
```

## Current Presets

- `auto`: resolves to the recommended platform/default policy.
- `one_shot_fast`: default performance path when enough system RAM is available.
- `low_ram_safe`: explicit fallback for tighter memory budgets.
- `wsl_compat`: WSL/driver fallback when pinning or streams are unstable.
- `warm_process`: process-lifetime cache/server-style comparison mode.
- `diffusers_offload_compat`: official Diffusers block-level group offload baseline.
- `diffusers_leaf_offload_compat`: official Diffusers leaf-level group offload baseline.

Inspect the current preset contract from a host runner without loading model weights:

```powershell
$env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS="1"
python run_dynamic_modular_distilled.py
Remove-Item Env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS -ErrorAction SilentlyContinue
```

## Experiment Documents

- `experiments/dynamic_offload_results.md` is the compact, report-ready benchmark file. Keep current preset tables, Diffusers group-offload comparisons, and platform summaries there.
- `experiments/ltx2_image_experiments.md` is the full historical LTX lab notebook. Keep raw notes, dead ends, old logs, and implementation chronology there.

The final public-facing report should be distilled from `dynamic_offload_results.md`, with links back to the historical log only when useful.

## Platform Recommendations

For the current LTX 2.3 BF16 1280x704, 8-step benchmark:

- Windows: use `auto` / `one_shot_fast`. It is the best compatibility-first path found so far when enough system RAM is available. If DDO detects insufficient RAM, it skips pinned CPU weights and remains safe, but denoise becomes much slower.
- WSL Ubuntu: use `wsl_compat`. It disables pinned CPU memory and streaming group-offload paths that were unstable on this WSL stack. It is slower than Windows full-pin but stable.
- Native Ubuntu: `diffusers_leaf_offload_compat` is currently the fastest measured preset for this exact benchmark. Keep `one_shot_fast` as the DDO dynamic-offload comparison path, especially for warm/server-style runs.
- Tight VRAM/RAM fallback: use `low_ram_safe` only when lower accelerator pressure matters more than latency.
- Server or repeated generation: use `warm_process` to measure process-lifetime cache behavior; it is not a one-image cold-start latency preset.

## Current Evidence Snapshot

For the LTX 2.3 image BF16 1280x704, 8-step benchmark on an 8 GB NVIDIA GPU:

- Windows `auto -> one_shot_fast`: total `133.2s`, generation pass `52.6s`, denoise `14.83s`, peak VRAM `6.45 GB`, peak RAM `27.38 GB`.
- Windows simulated 32 GB RAM: DDO correctly skipped pinned CPU weights; total `217.2s`, generation pass `132.2s`, denoise `114.70s`.
- Windows official Diffusers baselines: `diffusers_offload_compat` total `315.1s`, `diffusers_leaf_offload_compat` total `326.4s`.
- WSL Ubuntu: `wsl_compat` was the best stable DDO path measured, total `144.4s`, denoise `97.80s`; Diffusers leaf-level group offload crashed after denoise start on this WSL stack.
- Native Ubuntu: official Diffusers leaf-level group offload was fastest in the latest matrix, total `34.9s`; Diffusers block-level total `43.8s`; DDO `one_shot_fast` total `59.3s`.
- Warm process: DDO plan/pinned cache reduced the second transformer prepare on native Ubuntu from about `20.04s` to `0.90s`; use this only for server-style comparisons.

## Validation Roadmap

1. Keep the current Windows/WSL/native-Ubuntu BF16 matrix as the first report baseline.
2. Validate quantized LTX next, starting with SDNQ, then broader quantization paths such as bitsandbytes when a matching model is available.
3. If quantized loading exposes class/config or parameter-wrapper incompatibilities, keep fixes inside the generic loader/offload layer rather than adding model-specific code.
4. Only revisit native VBAR-style behavior if Python/Diffusers-compatible paths are exhausted and hardware compatibility risks are explicitly accepted.

## Compatibility Position

DDO intentionally stays above native VBAR-level hooks for now. The current priority is a portable PyTorch/Diffusers implementation that improves performance where possible while keeping fallback behavior understandable on different CUDA, WSL, and Linux configurations.
