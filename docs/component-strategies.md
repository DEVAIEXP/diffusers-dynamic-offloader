# Component strategies and memory trade-offs

DDO aims to improve low-VRAM inference throughput. A faster route can increase
the sampled GPU memory peak; offloading weights does not remove activation or
workspace costs. Choose a strategy per component instead of assuming one runtime
is optimal for prompt encoding, repeated denoise and final decode alike.

## VAE routing defaults

These defaults apply to the named `vae` component in `enable_offload` and
`enable_pipeline_offload`, not to arbitrary class-name detection or the low-level
`enable_dynamic_offload` API. Transformer and text-encoder policies are unchanged.

| Resolved preset | VAE route |
| --- | --- |
| `auto` (resolves to `one_shot_fast`), `one_shot_fast`, `low_ram_safe` | Official Diffusers leaf, no stream |
| `vmm_fast` settings from the native route | Official Diffusers leaf, no stream |
| `warm_process`, `wsl_compat`, `nvme_baseline` | Resident, preserving historical staged-host behavior |
| `off` | Unmanaged/resident when applied |

The VAE leaf policy uses `low_cpu_mem_usage=False`, matching the accepted
no-stream host calls. Native preset settings must come from the native setup;
the policy does not install an allocator or bypass fresh-process bootstrap.
It does not make WSL/native Linux VAE execution validated.

## Minimal application calls

```python
from diffusers_dynamic_offloader import enable_offload

# Default VAE strategy; do not implement transfers in the host.
vae_result = enable_offload(
    vae, component="vae", preset="auto", execution_device="cuda:0",
    use_environment=False,
)

# Explicit alternative, in a separate normal-PyTorch-allocator run:
linear_result = enable_offload(
    other_vae.eval(), component="vae", route="linear_convolution",
    execution_device="cuda:0", use_environment=False,
)
```

`linear_convolution` builds the isolated eager configuration inside DDO: CPU
offload, optional pinned weights (enabled by this route), zero residency, no
automatic budget or persistent pinned cache. It preserves the original dense
convolution forward and dtype. Explicit config overrides remain subject to its
preflight. See [convolution contracts](linear-convolution-offload.md).
It is not supported with the installed native VMM allocator or quantized modules.

Use `route="diffusers_group_offload", offload_type="leaf_level", use_stream=False`
to explicitly select leaf, or `route="none"` for resident execution. A shared
settings object can set `component_policies={"vae": {"route": "linear_convolution"}}`
without changing the transformer's policy. An explicit route wins; explicit
configs are not silently replaced by a preset default. Native VMM VAE remains
an explicit source/capacity opt-in, not an automatic class-based conversion.

## Phase boundaries and reporting

After denoise, detach the transformer, drop all pipeline/state references, and
perform the completed phase cleanup before loading the VAE. In an installed
native allocator, ordinary `torch.cuda.empty_cache()` does not prove its idle
pool was returned to the driver. The selected Qwen low-memory combination also
uses the explicit `release_native_vmm_idle_cache` helper after cleanup; this
requires a supporting DLL and a quiescent phase boundary. Routing alone never
automatically trims that pool, clears cuBLAS workspaces or purges Windows standby.

Report every executed route with `vae_result.metrics_snapshot()`. Use
`cleanup_on_error()` for DDO's managed error boundary; official group hooks remain
owned by Diffusers, not by `remove_dynamic_offload`. Drop those module/pipeline
references when releasing the component. Staged decode must explicitly receive
the execution device when weights reside on CPU.

## Selected evidence, not a universal ranking

All selected Qwen Image 2.1 BF16 1024-square, 20-step cases preserved the accepted
PNG exactly. With a native transformer and explicit idle trim, VAE leaf measured
3.93 GiB sampled absolute driver peak and 1.89 s decode; native VMM VAE measured
5.95 GiB and 1.43 s. With a linear transformer, leaf measured 6.47 GiB/1.60 s and
DDO linear convolution 6.50 GiB/1.36 s. The latter pair is the matched comparison;
do not rank all four as if their process allocators were identical.

For memory-sensitive final decode, the selected native-transformer plus leaf
combination is preferable to saving a fraction of a second with a larger peak.
Compatibility is accepted for the selected VAE, not every VAE or platform.
Detailed pipeline evidence belongs in the sibling
[ddo-examples benchmarks](https://github.com/DEVAIEXP/ddo-examples), not in a
generic library performance promise. The new preset routing is CPU-contract
validated; a new real-model run requires separate approval.

The selected CPU regression suite completed with 126 tests and two subtests
passing (7.00 s), covering component routing/precedence, the official helper
arguments, existing presets/backends/profiles, convolution ownership, and Qwen
runner/block contracts. Changed runner/policy tests passed Ruff and both
repositories passed `git diff --check`. No CUDA generation, DLL rebuild,
compatibility removal or historical-artifact rewrite accompanied this migration.
