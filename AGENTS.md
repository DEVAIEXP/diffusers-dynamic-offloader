# Repository Guidance

This repository is focused on `diffusers-dynamic-offloader` (DDO), a generic Diffusers-style dynamic offload manager for low-VRAM inference.

## Project Direction

- Keep the implementation model-agnostic. It must not depend on LTX-specific class names or component types.
- Prefer Diffusers conventions and architecture. Code should be easy to upstream or adapt into Diffusers internals later.
- Keep runtime logic self-contained in the offload manager. Host runners should mainly parse/pass parameters and orchestrate components.
- Preserve the existing PyTorch-backed DDO runtime paths. Native VMM work is explicitly scoped to the experimental `vmm_runtime` route and should not replace or destabilize the existing `linear_runtime` presets.
- Keep Windows standby purge exposed by DDO as a Windows-only lifecycle helper. It should be safe/no-op outside Windows and must not become a generic Linux requirement.

## Runtime Architecture

- `linear_runtime` is the stable DDO runtime used by `one_shot_fast`, `low_ram_safe`, `wsl_compat`, and `warm_process`. It uses normal PyTorch CUDA allocation and must remain intact while native VMM work evolves.
- `diffusers_offload_compat`, `diffusers_leaf_offload_compat`, and `off` remain compatibility/control presets that route through plan-mode behavior rather than `linear_runtime`.
- `vmm_runtime` is the only route intended to use the native C++ VMM allocator in `ddo-vmm-allocator`. Consolidate VMM experiments into this route instead of adding parallel runner-only paths.
- The native VMM target flow must install the C++ PyTorch allocator in a fresh child process before CUDA is initialized, then load the model under that allocator. The allocator owns virtual address reservation, physical residency, mapping/unmapping, pressure response, budget guards, and transfer accounting.
- Avoid adding Python-side residency heuristics that try to pin a prefix, preserve a suffix, or otherwise simulate allocator pressure from hook order. Such experiments may be kept only as explicitly marked diagnostics, not as the target VMM path.
- Do not remove or rewrite the working non-VMM runtime while implementing VMM. Once `vmm_runtime` is validated, add profile support for it separately.

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
