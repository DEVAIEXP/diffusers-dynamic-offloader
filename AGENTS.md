# Repository Guidance

This repository is focused on `diffusers-dynamic-offloader` (DDO), a generic Diffusers-style dynamic offload manager for low-VRAM inference.

## Project Direction

- Keep the implementation model-agnostic. It must not depend on LTX-specific class names or component types.
- Prefer Diffusers conventions and architecture. Code should be easy to upstream or adapt into Diffusers internals later.
- Keep runtime logic self-contained in the offload manager. Host runners should mainly parse/pass parameters and orchestrate components.
- Avoid native or hardware-specific low-level memory hooks unless explicitly chosen later. The current priority is compatibility first, then performance. If comparing with native approaches, describe it only as an inspiration or external reference, not as something DDO implements.
- Keep Windows standby purge exposed by DDO as a Windows-only lifecycle helper. It should be safe/no-op outside Windows and must not become a generic Linux requirement.

## Experiment Tracking

- `experiments/dynamic_offload_results.md` is the compact benchmark sidecar for current comparison tables and report-ready notes.
- `experiments/ltx2_image_experiments.md` is local-only historical LTX scratch material and should stay ignored/untracked in this repository.
- Historical results may stay in experiment docs, but stale recommendations must be marked as historical or corrected.
- Use DDO environment names in new docs and commands:
  - `DDO_*` for library settings.
  - `DDO_RUNNER_*` for host/example runner settings.

## Current Preset Meaning

- `auto` resolves to `one_shot_fast` on Windows, Linux, and WSL.
- `one_shot_fast` is the default benchmark path: RAM-aware balanced planner, auto resident-module budget from model size and CUDA VRAM headroom, and pinned CPU weights only when enough usable system RAM is available.
- `low_ram_safe` is an explicit low-VRAM fallback. It reduces accelerator pressure but can make denoise copy-bound and very slow.
- `wsl_compat` is an explicit WSL/driver fallback for cases where pinned-memory or stream behavior is unstable.
- `warm_process` is for process-lifetime cache/server-like comparisons.
- `diffusers_offload_compat` uses official Diffusers block-level group offload as a compatibility baseline, not the current performance baseline.
- `diffusers_leaf_offload_compat` keeps the official Diffusers leaf-level group offload path available for quantized backends, native-Linux comparisons, and smaller/different models.

## Working Rules

- Do not add LTX runners or custom blocks back into this repository unless they are explicitly turned into small, generic examples.
- Keep LTX-specific execution experiments in the sibling host repository and install DDO there with `pip install -e` or `uv pip install -e`.
- Use `apply_patch` for manual edits when the sandbox permits it.
- Before changing experiment conclusions, check the latest pasted benchmark numbers and avoid resurrecting older assumptions.

- Keep README examples short and link longer usage patterns from `docs/`.
- Document both full-pipeline and phase-staged usage. Full-pipeline examples are for minimal integration; phase-staged examples are for memory-sensitive runs and benchmark comparisons.
- Quantized backends should preserve backend-specific forward logic and usually route through Diffusers group-offload compatibility presets.
