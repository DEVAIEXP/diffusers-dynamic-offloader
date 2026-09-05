# diffusers-dynamic-offloader

Model-agnostic dynamic offload helpers for Diffusers-style low-VRAM inference.

DDO is a Python compatibility-first layer: it can keep selected modules resident on the accelerator, stream large weights at runtime, optionally pin CPU tensors when system RAM allows it, and expose repeatable presets through `DDO_*` environment variables.

This repository was split from the LTX 2.3 modular image experiments so the offload manager can evolve independently from any single model family. The LTX runners and custom blocks should remain in the host experiment repository and import this package with an editable install.

## Install For Local Experiments

From the host project environment:

```powershell
uv pip install -e E:\ProjetosIA\diffusers-dynamic-offloader
```

Then the host runner can import:

```python
from diffusers_dynamic_offloader import DynamicOffloadSettings, apply_dynamic_offload
```

## Environment Prefixes

Library settings use `DDO_*`.

Example or host-runner settings should use `DDO_RUNNER_*`.

```powershell
$env:DDO_PRESET="auto"
$env:DDO_RUNNER_WIDTH="1280"
$env:DDO_RUNNER_HEIGHT="704"
$env:DDO_RUNNER_STEPS="8"
$env:DDO_RUNNER_SEED="43"
$env:DDO_RUNNER_METRICS_LEVEL="1"
```

## Current Presets

- `one_shot_fast`: default performance path when enough system RAM is available.
- `low_ram_safe`: explicit fallback for tighter memory budgets.
- `wsl_compat`: WSL/driver fallback when pinning or streams are unstable.
- `warm_process`: process-lifetime cache/server-style comparison mode.
- `diffusers_offload_compat`: official Diffusers block-level group offload baseline.
- `diffusers_leaf_offload_compat`: official Diffusers leaf-level group offload baseline.

See `experiments/dynamic_offload_results.md` for the current benchmark notes.
