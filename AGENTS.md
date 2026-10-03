# Repository Guidance

This repository is focused on `diffusers-dynamic-offloader` (DDO), a generic Diffusers-style dynamic offload manager for low-VRAM inference.

## Project Direction

- Keep the implementation model-agnostic. It must not depend on LTX-specific class names or component types.
- Prefer Diffusers conventions and architecture. Code should be easy to upstream or adapt into Diffusers internals later.
- Keep runtime logic self-contained in the offload manager. Host runners should mainly parse/pass parameters and orchestrate components.
- Preserve the existing PyTorch-backed DDO runtime paths. Native VMM work is explicitly scoped to the experimental `vmm_fast_runtime` route; the legacy `vmm_runtime` is the protected NVMe baseline. Neither may destabilize the existing `linear_runtime` presets.
- Keep Windows standby purge exposed by DDO as a Windows-only lifecycle helper. It should be safe/no-op outside Windows and must not become a generic Linux requirement.

## Runtime Architecture

- `linear_runtime` is the stable DDO runtime used by `one_shot_fast`, `low_ram_safe`, `wsl_compat`, and `warm_process`. It uses normal PyTorch CUDA allocation and must remain intact while native VMM work evolves.
- `diffusers_offload_compat`, `diffusers_leaf_offload_compat`, and `off` remain compatibility/control presets that route through plan-mode behavior rather than `linear_runtime`.
- `vmm_fast_runtime` is the target native C++ VMM route in `ddo-vmm-allocator`: one model-address arena with native range faulting, signatures, unpinning, fallback, and accounting. `vmm_runtime` is the protected legacy NVMe baseline. Consolidate new VMM work into `vmm_fast_runtime`, never into runner-only paths.
- The native VMM target flow must install the C++ PyTorch allocator in a fresh child process before CUDA is initialized, then load the model under that allocator. The allocator owns virtual address reservation, physical residency, mapping/unmapping, pressure response, budget guards, and transfer accounting.
- `ddo-vmm-allocator` is a pinned Git submodule. DDO consumes its public API; C++/CUDA and native adapters belong in the submodule, not in host runners.
- Avoid adding Python-side residency heuristics that try to pin a prefix, preserve a suffix, or otherwise simulate allocator pressure from hook order. Such experiments may be kept only as explicitly marked diagnostics, not as the target VMM path.
- Do not remove or rewrite the working non-VMM runtime while implementing VMM. Add VMM capacity profiles separately after the transformer route is validated.

## Experiment Tracking

- `experiments/dynamic_offload_results.md` is the compact benchmark sidecar for current comparison tables and report-ready notes.
- `experiments/ltx2_image_experiments.md` is local-only historical LTX scratch material and should stay ignored/untracked in this repository.
- Historical results may stay in experiment docs, but stale recommendations must be marked as historical or corrected.
- Use DDO environment names in new docs and commands:
  - `DDO_*` for library settings.
  - `DDO_RUNNER_*` for host/example runner settings.
- Keep external-reference notes and unredacted historical investigation material out of deliverable docs. Before publication, audit active files and private history for prohibited provenance references and validate the cleaned result again.

## Current Preset Meaning

- `auto` resolves to `one_shot_fast` on Windows, Linux, and WSL.
- `one_shot_fast` is the default benchmark path: RAM-aware balanced planner, auto resident-module budget from model size and CUDA VRAM headroom, and pinned CPU weights only when enough usable system RAM is available.
- `low_ram_safe` is an explicit low-VRAM fallback. It reduces accelerator pressure but can make denoise copy-bound and very slow.
- `wsl_compat` is an explicit WSL/driver fallback for cases where pinned-memory or stream behavior is unstable.
- `warm_process` is for process-lifetime cache/server-like comparisons.
- `diffusers_offload_compat` uses official Diffusers block-level group offload as a compatibility baseline, not the current performance baseline.
- `diffusers_leaf_offload_compat` keeps the official Diffusers leaf-level group offload path available for quantized backends, native-Linux comparisons, and smaller/different models.

## VMM Route Contract

- A host runner uses `--ddo-backend` and `--ddo-preset`; it must not require users to construct `DDO_BACKEND` or native allocator settings by hand.
- `linear/auto` preserves the existing `one_shot_fast` resolution. `vmm/auto` selects `vmm_fast`; `vmm/nvme_baseline` selects the protected VMM fallback.
- `vmm_fast` starts through `ddo_vmm_bootstrap.ensure_vmm_fast_process()` as the first executable action, before CUDA is initialized. Direct initialization is diagnostic-only.
- DDO owns generic VMM setup, weight-source lifecycle, hooks, cleanup, fallback, profiling, and telemetry. A host only parses arguments, loads Diffusers components, and orchestrates pipeline phases.
- VMM profiles are capacity observations per component, not Python-side residency budgets. Validate the transformer first, then reuse the mechanism for text encoders.
- Runners consume `DynamicOffloadMetrics`; they must not read hook state, planner internals, or VMM statistic attributes directly.

## VMM Development and Gates

- Current user-approved closure is lean: accept the selected Windows performance and critical ownership/capacity/cancellation gates, without an open-ended long-run or model/platform test matrix. Extend tests when a critical implementation change or a reported defect requires it; do not automatically repeat the full benchmark.
- Next platform block is native Linux `.so` portability, not a claim that changing the library suffix enables the existing Windows-only source/cache transport. Preserve the accepted Windows build/route; Linux execution remains experimental until the user tests it. WSL is not a substitute for native Linux validation.
- For a new Linux session, read `docs/linux-handoff.md` first. Use a separate `.venv-linux` and `build-linux`; the isolated native CPU recipe never authorizes CUDA execution. Complete native transport with a real Linux compiler before proposing the small CUDA gate.

- `.[vmm-dev]` is the Windows CUDA development profile. It is optional; ordinary DDO installs must not acquire CUDA development dependencies.
- Windows is the initial supported VMM platform. Linux validation follows a successful Windows route. Do not promise WSL or macOS VMM support.
- Before any CUDA test, benchmark, benchmark-parameter change, compatibility removal, or history cleanup, show the command, hypothesis, and approval criterion and wait for user confirmation.
- Prefer consolidated, explicitly approved validation sequences over fragmented micro-runs. Approval may cover a capacity gate followed by named benchmark cases; stop at the first failed gate or memory anomaly, and do not extend the approved parameters or scope.
- Each incremental VMM block must compile, pass its unit/CUDA contracts, and retain the NVMe baseline before the next block. Do not recalibrate residency budgets during the persistent-source through fallback blocks.
- Record deliverable benchmark results in `experiments/dynamic_offload_results.md`.
- Follow `docs/native-source-cache-plan.md` for the native persistent-source work.
  The registered source cache owns native bytes, not loader CPU tensors. Host
  cache capacity is separate from GPU residency; no Python cache-admission policy.
  The persistent host buffer also separates committed storage from CUDA
  registration: unregistering must not discard bytes or trigger file rereads.
  Never increase either ceiling automatically or treat a compiled change as
  a passing CUDA/performance gate.
- Persistent-source execution uses native reusable slots and immutable native
  source batches. Slot recycling must wait for the last compute/upload event;
  partial submission failures retain ownership until completion is proven.
  Keep per-window request/copy/reap scheduling out of the normal Python route.

## Working Rules

- Native validated checkpoints and limitations are summarized in `docs/native-support-status.md`. Preserve the selected staged performance and small robustness gates; do not generalize them to arbitrary models, driver-loss recovery or automatic component opt-in.
- Host memory reports must distinguish baseline-relative GPU deltas from sampled absolute driver usage, and requested controls from executed component routes. Missing historical fields are unavailable, not zero; do not rewrite old benchmark artifacts.

- Do not add LTX runners or custom blocks back into this repository unless they are explicitly turned into small, generic examples.
- Keep LTX-specific execution experiments in the sibling host repository and install DDO there with `pip install -e` or `uv pip install -e`.
- Use `apply_patch` for manual edits when the sandbox permits it.
- Before changing experiment conclusions, check the latest pasted benchmark numbers and avoid resurrecting older assumptions.

- Keep README examples short and link longer usage patterns from `docs/`.
- Document both full-pipeline and phase-staged usage. Full-pipeline examples are for minimal integration; phase-staged examples are for memory-sensitive runs and benchmark comparisons.
- Quantized backends should preserve backend-specific forward logic and usually route through Diffusers group-offload compatibility presets.
