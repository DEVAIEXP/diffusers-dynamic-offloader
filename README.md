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
from diffusers_dynamic_offloader import enable_dynamic_offload
```

## Basic API

Use the high-level helper when you already have a Diffusers/Transformers module instance:

```python
from diffusers_dynamic_offloader import enable_dynamic_offload

result = enable_dynamic_offload(
    transformer,
    preset="auto",
    execution_device="cuda:0",
    record_event=record_event,  # optional
)
transformer = result.module
```

Explicit keyword arguments override preset and environment values:

```python
enable_dynamic_offload(
    transformer,
    preset="one_shot_fast",
    max_resident_module_budget_gb=6.0,
    pin_cpu_workers=4,
)
```

The official Diffusers group offload path is also exposed through DDO so runners do not need to import Diffusers hooks directly:

```python
from diffusers_dynamic_offloader import enable_diffusers_group_offload

enable_diffusers_group_offload(
    transformer,
    offload_type="block_level",
    use_stream=True,
    record_stream=True,
)
```

The lower-level `DynamicOffloadSettings` and `apply_dynamic_offload` APIs remain available for experiments, but runners should prefer the high-level helpers.
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
python run_modular_distilled.py
Remove-Item Env:DDO_RUNNER_PRINT_DYNAMIC_OFFLOAD_PRESETS -ErrorAction SilentlyContinue
```

## Experiment Documents

- `experiments/dynamic_offload_results.md` is the compact, report-ready benchmark file. Keep current preset tables, Diffusers group-offload comparisons, and platform summaries there.
- `experiments/ltx2_image_experiments.md` is the full historical LTX lab notebook. Keep raw notes, dead ends, old logs, and implementation chronology there.

The final public-facing report should be distilled from `dynamic_offload_results.md`, with links back to the historical log only when useful.

## Current Evidence Snapshot

For the LTX 2.3 image BF16 1280x704, 8-step benchmark on an 8 GB NVIDIA GPU:

- Windows with enough usable RAM: DDO `one_shot_fast` keeps denoise around `14-15s`, with cold setup usually around `30-35s`.
- Windows with simulated 32 GB RAM: the RAM-aware planner correctly skips pinned CPU weights; setup drops under `10s`, but denoise becomes copy-bound around `~198s` for this model.
- Official Diffusers group offload is much slower in this specific low-VRAM transformer test: `block_level` with stream/record stream was the best official path so far around `~265s`, while `leaf_level` was `~315-320s`.
- WSL needs explicit validation because pinned-memory failures can poison the CUDA context on some driver setups.
- Native Ubuntu previously accepted pinned DDO and was the strongest platform in the early matrix, but needs a final repeated pass after the current preset cleanup.

## Validation Roadmap

1. Freeze Windows preset behavior with `auto`, simulated 32 GB RAM, and official Diffusers fallback comparisons.
2. Repeat the same matrix on WSL only after Windows changes are stable.
3. Repeat the same matrix on native Ubuntu before declaring platform defaults.
4. Add a quantized-model section after BF16 presets are stable, starting with SDNQ and then other quantization paths if available.

## Compatibility Position

DDO intentionally stays above native VBAR-level hooks for now. The current priority is a portable PyTorch/Diffusers implementation that improves performance where possible while keeping fallback behavior understandable on different CUDA, WSL, and Linux configurations.