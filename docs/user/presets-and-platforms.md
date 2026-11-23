# Runtimes and presets

DDO has two execution families. **Linear** uses PyTorch to move supported weights
between CPU memory and the GPU. **VMM** uses the optional native allocator:
it keeps the GPU addresses used by tensors stable while assigning and reusing
physical GPU memory behind those addresses. Both leave model computation to
Diffusers and the selected backend.

A **backend** chooses the family. A **preset** chooses its offload policy.
The **runtime** is the implementation that actually runs. A saved **profile**
is different: it records a capacity recommendation for a particular workload.

## Backend and preset selection

These are the public runner selections registered by
`add_dynamic_offload_arguments()`:

| Backend | Preset argument | Resolved preset | Runtime | Use |
| --- | --- | --- | --- | --- |
| `linear` | `auto` | `one_shot_fast` | `linear_runtime` for supported dense modules | Default PyTorch route |
| `linear` | `one_shot_fast` | `one_shot_fast` | `linear_runtime` for supported dense modules | One-shot generation with RAM-aware CPU pinning |
| `linear` | `low_ram_safe` | `low_ram_safe` | `linear_runtime` for supported dense modules | CPU pinning disabled; measure the memory/latency trade-off |
| `linear` | `wsl_compat` | `wsl_compat` | `linear_runtime` for supported dense modules | WSL configuration with pinning disabled and conservative stream settings |
| `linear` | `warm_process` | `warm_process` | `linear_runtime` for supported dense modules | Reuse within one process; compare repeated generations separately |
| `vmm` | `auto` | `vmm_fast` | `vmm_fast_runtime` | VMM Fast: native weight arena and retained-source transport |
| `vmm` | `nvme_baseline` | `nvme_baseline` | `vmm_runtime` | Experimental storage-backed alternative |

For example, `--ddo-backend vmm --ddo-preset auto` selects **VMM Fast**.
`vmm_fast` is the resolved name, not a preset argument accepted by the VMM CLI.
`auto` under `linear` always resolves to `one_shot_fast`; it does not silently
switch to VMM on a particular GPU or platform.

The table describes dense transformer execution. The result also depends on
the component and supported backend. DDO detects supported packed SDNQ modules
and preserves their computation through `sdnq_runtime` when that dynamic route
is explicitly selected. Other quantized modules can fall back to official
Diffusers offload. Inspect the report rather than inferring execution from the
requested preset alone.

## Compatibility and manual placement

These selections use the `linear` backend interface but choose a different
component route:

| Preset | Component route | Use |
| --- | --- | --- |
| `diffusers_offload_compat` | Official Diffusers block-level group offload | Official hooks through the DDO interface |
| `diffusers_leaf_offload_compat` | Official Diffusers leaf-level group offload | Fine-grained official hooks and backend-preserving controls |
| `off` | No offload hooks; the high-level API moves the component to the execution device | Resident execution when it fits |

Compatibility hooks are not `linear_runtime`. `resolve_dynamic_offload_backend()`
selects a family/configuration; `enable_offload()` resolves the actual component
route. Reports distinguish both.

## Component routes

An encoder, transformer and VAE can use different strategies.
Selecting VMM Fast for the transformer does not make the entire pipeline native.
The VAE defaults and explicit alternatives are defined once in
[component strategies](component-strategies.md). Native encoder or VAE ownership
requires explicit supported sources and its own capacity setup.

Use [integration](integration.md) for both families, [dynamic loading](dynamic-loading.md)
for checkpoint association and [capacity profiles](capacity-profiles.md) for
calibration. A profile recommends a weight budget; it is not a total GPU-memory cap.

## Choosing a preset

Start with a measured recipe close to your model and workload in
[DDO examples](https://devaiexp.github.io/ddo-examples/docs/benchmarks/).
Compare full-run latency as well as denoising, VAE decode and memory peaks.
`low_ram_safe` describes a policy, not a guarantee that it uses less RAM or fits
a smaller machine. CPU pinning, resident modules, activations and source storage
all affect memory use.

`nvme_baseline` remains an experimental alternative and may be discontinued
in a future release. It is maintained as a protected fallback today. Its lower
source-memory use in some workloads can come with much longer runtime;
validate it on the intended storage and hardware.

## Platform notes

| Platform | What changes |
| --- | --- |
| Windows | Optional standby purge is off by default. Enable only the phases needed by the recipe. VMM profiles also observe the Windows local GPU budget. |
| Native Linux | VMM admission respects available RAM, cgroup/process limits and the actual overcommit mode. Use a Linux native wheel. |
| WSL | Use a separate Linux environment and the Windows-provided GPU driver. Pinned-memory and stream constraints differ from native Linux; selected recipes use `wsl_compat` or bounded VMM capacities. |

No preset or successful smoke test establishes every-model acceptance.
See [support](../developer/support.md) and the platform-specific benchmark notes.
