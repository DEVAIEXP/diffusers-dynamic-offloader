# Configuration Reference

DDO is designed to work with a small public surface:

- call `enable_offload(module_or_pipeline, preset="auto")` for the default behavior;
- pass `DynamicOffloadSettings` only when application code wants explicit control;
- use `DDO_*` environment variables as optional overrides for experiments, launch scripts, or CI.

Python arguments should be preferred for reusable application code. Environment variables are best treated as runtime overrides: useful while benchmarking, but less explicit than code.

## Override Order

This catalog includes linear defaults and the protected legacy VMM route.
For the current experimental native route, start with
[native VMM usage](native-vmm.md): runner backend/preset selection, fresh-process
bootstrap and independent host/GPU capacities are part of its contract.
The retained-source configuration does not turn VMM into the linear default.

Configuration is resolved in this order:

1. Built-in preset defaults.
2. Explicit `DynamicOffloadSettings` / `DynamicOffloadConfig` values passed by Python code.
3. Recognized `DDO_*` environment variables, when `DynamicOffloadSettings.from_env(...)` is used.
4. Component-specific policy overrides passed to `enable_pipeline_offload(...)` or `enable_offload(...)`.

Component-specific environment variables can override a route for one component without changing the rest of the pipeline.

## DynamicOffloadSettings

`DynamicOffloadSettings` is the high-level resolved settings object. Most users do not need to instantiate it directly.

| Field | Meaning |
| --- | --- |
| `requested_preset` | Preset requested by the caller or environment. |
| `effective_preset` | Preset after `auto` and platform-specific resolution. |
| `enabled` | Whether DDO should apply an offload route. |
| `plan` | Whether `DDO_PLAN` enabled the plan-mode dynamic route. |
| `config` | The resolved `DynamicOffloadConfig`. |
| `requested_pin_cpu_memory` | The pinning choice before platform safety overrides. |
| `effective_pin_cpu_memory` | The final pinning choice used by DDO. |
| `disable_pin_on_wsl` | Whether WSL should disable pinned CPU memory automatically. |
| `running_on_wsl` | Whether the current process was detected as WSL. |
| `component_policies` | Per-component route overrides. |

## DynamicOffloadConfig

`DynamicOffloadConfig` controls how DDO patches or routes an individual module.

The defaults below are the `DynamicOffloadConfig(...)` dataclass defaults. `DynamicOffloadSettings.from_env(...)` resolves a few different operational defaults when no environment override or preset supplies one: available system RAM is detected, the small-tensor threshold is `1024 KiB`, pinning uses `4` workers, the built-in always-resident patterns are used, and profile printing is off. A preset can override these resolved values.

| Field | Default | Environment variable | Notes |
| --- | --- | --- | --- |
| `execution_device` | `"cuda:0"` | Python only | Device where active compute runs. |
| `offload_device` | `"cpu"` | Python only | Device where inactive weights are kept. |
| `target_module_classes` | `torch.nn.Linear`, `torch.nn.Embedding` | Python only | Module classes eligible for dynamic linear runtime patching. |
| `skip_modules_pattern` | `()` | Python only | Regex patterns for modules DDO should ignore. |
| `always_resident_modules_pattern` | `()` | `DDO_ALWAYS_RESIDENT_MODULE_PATTERNS` | Comma- or semicolon-separated regex patterns kept resident. `from_env(...)` supplies patterns for common projection, embedding, and normalization modules unless overridden. |
| `small_tensor_threshold_bytes` | `16 KiB` | `DDO_SMALL_TENSOR_THRESHOLD_KB` | Small tensors are moved directly instead of dynamically copied. `from_env(...)` resolves this to `1024 KiB` before preset overrides. |
| `execution_mode` | `"plan"` | `DDO_EXECUTION_MODE` | `plan` builds the placement plan; `linear_runtime` enables DDO's dense-linear runtime path. Experimental `vmm_runtime` streams standard dense BF16/FP16 linear weights through CUDA virtual-memory mappings. |
| `linear_runtime_strategy` | `"functional"` | `DDO_LINEAR_RUNTIME_STRATEGY` | Dense-linear execution strategy: `functional` calls the functional operator with a transient weight copy. Experimental `swap` temporarily attaches the CUDA copy to the module parameter and restores the CPU source immediately after the original forward. |
| `vmm_resident_weight_budget_gb` | `0.0` | `DDO_VMM_RESIDENT_WEIGHT_BUDGET_GB` | Required positive physical-weight cache budget for `vmm_runtime`. VMM reserves virtual addresses separately; this budget limits currently mapped CUDA backing. Requires `pip install diffusers-dynamic-offloader[vmm]`. |
| `vmm_staging_buffer_gb` | `0.625` | `DDO_VMM_STAGING_BUFFER_GB` | Bounded reusable pinned host buffer for VMM group uploads. It must fit the largest execution group, not the full model. |
| `pin_cpu_memory` | `False` | `DDO_PIN_CPU_MEMORY` | Enables pinned CPU tensor copies when supported and safe. |
| `allow_pin_memory_fallback` | `True` | `DDO_ALLOW_PIN_MEMORY_FALLBACK` | Fall back cleanly if pinning fails. |
| `pin_cpu_workers` | `1` | `DDO_PIN_CPU_WORKERS` | Positive worker count for pinning preparation. `from_env(...)` resolves `4` before preset overrides. |
| `cache_plan` | `True` | `DDO_CACHE_PLAN` | Cache structural planning in-process. |
| `cache_pinned_weights` | `False` | `DDO_CACHE_PINNED_WEIGHTS` | Reuse pinned tensors in the same process when possible. |
| `pinned_weight_cache_namespace` | `""` | `DDO_PINNED_WEIGHT_CACHE_NAMESPACE` | Separates pinned tensor caches between model/config groups. |
| `auto_budget_policy` | `"off"` | `DDO_AUTO_BUDGET_POLICY` | Supported values: `off`, `balanced`. |
| `max_resident_module_budget_gb` | `6.0` | `DDO_MAX_RESIDENT_MODULE_BUDGET_GB` | Upper cap used by auto budgeting; `0` means uncapped. |
| `auto_vram_headroom_gb` | `0.0` | `DDO_AUTO_VRAM_HEADROOM_GB` | Extra VRAM margin reserved by auto budgeting. |
| `profile_vram_headroom_gb` | `1.0` | `DDO_PROFILE_VRAM_HEADROOM_GB` | VRAM margin held back when a zero-resident capacity-profile calibration calculates its next-run resident budget. |
| `profile_name` | `""` | Python only | User-chosen workload name. DDO looks up its recommendation during setup; empty disables lookup. |
| `build_profile` | `False` | Python only | Calibrate with zero resident budget; use the returned `profile_run()` context to measure and save. Requires a name. |
| `auto_full_pin_min_model_to_vram_ratio` | `4.0` | `DDO_AUTO_FULL_PIN_MIN_MODEL_TO_VRAM_RATIO` | When usable system RAM fits all eligible weights and the model-to-VRAM ratio meets this threshold, balanced auto selects full CPU pinning. Set `0` to disable this automatic choice. |
| `max_pin_weight_budget_gb` | `0.0` | `DDO_MAX_PIN_WEIGHT_BUDGET_GB` | Upper cap for auto pinning; `0` means uncapped. |
| `available_system_ram_gb` | `0.0` | `DDO_AVAILABLE_SYSTEM_RAM_GB` | `from_env(...)` detects available system RAM unless this is overridden; direct config construction uses the supplied value. |
| `system_ram_headroom_gb` | `6.0` | `DDO_SYSTEM_RAM_HEADROOM_GB` | RAM held back from auto pinning decisions. |
| `pin_weight_budget_gb` | `0.0` | `DDO_PIN_WEIGHT_BUDGET_GB` | Explicit pinned weight budget. |
| `pin_weight_budget_ratio` | `0.0` | `DDO_PIN_WEIGHT_BUDGET_RATIO` | Fraction of candidate weights to pin; values are clamped to the range `0.0` through `1.0`. |
| `pin_weight_selection` | `"spread"` | `DDO_PIN_WEIGHT_SELECTION` | Supported values: `first`, `spread`, `largest`. |
| `resident_module_budget_gb` | `0.0` | `DDO_RESIDENT_MODULE_BUDGET_GB` | Explicit resident module budget. |
| `resident_module_patterns` | `()` | `DDO_RESIDENT_MODULE_PATTERNS` | Comma-separated regex patterns; `auto` lets DDO infer common block patterns. |
| `resident_module_selection` | `"spread"` | `DDO_RESIDENT_MODULE_SELECTION` | Supported values: `first`, `spread`, `largest`. |
| `load_safetensors_backend` | `""` | `DDO_LOAD_SAFETENSORS_BACKEND` | Loading backend: `""` preserves the library default; `mmap` and `pread` are the supported overrides. Only use an override after measuring the target host. |
| `verbose` | `False` | `DDO_VERBOSE` | Prints additional setup details. |
| `show_profile` | `True` | `DDO_SHOW_PROFILE` | Prints the dynamic offload profile summary. `from_env(...)` resolves this to false unless enabled. |

`DDO_SAFETENSORS_BACKEND` is also accepted as a compatibility alias for `DDO_LOAD_SAFETENSORS_BACKEND`.

### Experimental CUDA VMM runtime

The following subsection describes the protected `vmm_runtime` / NVMe route,
not the current native retained-source route. Both are preserved; the linear
preset family is not being removed or replaced.

`vmm_runtime` is an opt-in, CUDA-only route for standard dense `nn.Linear`
weights. It reserves a stable CUDA virtual-address slot for each weight, then
maps bounded physical backing only while that weight is needed. It is intended
to avoid eagerly pinning a full transformer just to obtain fast asynchronous
copies.

Install the optional dependency, then choose a positive mapped-weight budget:

```python
from diffusers_dynamic_offloader import DynamicOffloadConfig, enable_dynamic_offload

result = enable_dynamic_offload(
    transformer,
    config=DynamicOffloadConfig(
        execution_mode="vmm_runtime",
        vmm_resident_weight_budget_gb=4.0,
    ),
    use_environment=False,
)
```

The route currently accepts CUDA devices reporting virtual-memory-management
support and CPU-resident contiguous BF16 or FP16 `nn.Linear` weights. It does
not support quantized/custom linear forwards, embeddings, or a zero mapping
budget. VMM reduces weight-residency pressure; activation and KV/cache memory
must still fit independently.

### Native VMM fast configuration

Use `--ddo-backend vmm --ddo-preset auto` to select `vmm_fast_runtime`.
`auto` here selects a route, not a calibrated residency or performance preset.
There are no additional named native tuning presets. Choose capacities for the
actual component/workload; the accepted benchmark manifest is not a universal
default. `vmm/nvme_baseline` retains the protected legacy route.

All fields below belong to `DynamicOffloadConfig`. Defaults are dataclass
defaults, not the LTX harness's chosen benchmark settings. Size suffixes `gb`
mean GiB, `mb` mean MiB. Python config applies directly; environment entries
are consumed by `DynamicOffloadSettings.from_env(...)`.

| Config field | Default | Environment | Generic DDO CLI / meaning |
| --- | --- | --- | --- |
| `vmm_native_transport` | `False` | `DDO_VMM_NATIVE_TRANSPORT` | `--ddo-vmm-native-transport`; native reusable registered-host windows. Source retention also selects this transport. |
| `vmm_resident_weight_budget_gb` | `0.0` | `DDO_VMM_RESIDENT_WEIGHT_BUDGET_GB` | `--ddo-vmm-resident-weight-budget-gb`; active weight capacity, not complete driver GPU usage. CLI helper default4GiB; activations/auxiliaries need additional headroom. |
| `vmm_staging_buffer_gb` | `0.625` | `DDO_VMM_STAGING_BUFFER_GB` | `--ddo-vmm-staging-buffer-gb`; TOTAL host staging, rounded to native windows, not per slot. |
| `vmm_source_cache_gb` | `0.0` | `DDO_VMM_SOURCE_CACHE_GB` | `--ddo-vmm-source-cache-gb`; native retained-storage ceiling, separate from GPU and registration. Zero disables this large retained-source configuration. |
| `vmm_source_registered_gb` | `0.0` | `DDO_VMM_SOURCE_REGISTERED_GB` | `--ddo-vmm-source-registered-gb`; registration within retained storage; zero uses storage ceiling, explicit values require whole64MiB segments within storage capacity. |
| `vmm_module_pattern` | `.*` | `DDO_VMM_MODULE_PATTERN` | Config/ENV; select native attachment units, not model architecture detection. No generic CLI flag currently. |
| `vmm_page_mb` | `0` | `DDO_VMM_PAGE_MB` | Config/ENV; zero requests native/default geometry. Explicit page size must satisfy native alignment/support checks. No generic CLI flag currently. |
| `vmm_dxgi_local_headroom_gb` | `0.0` | `DDO_VMM_DXGI_LOCAL_HEADROOM_GB` | Config/ENV; Windows local budget reserve, not a Linux requirement. |
| `vmm_dxgi_shared_headroom_gb` | `0.0` | `DDO_VMM_DXGI_SHARED_HEADROOM_GB` | Config/ENV; Windows non-local budget reserve, not measured shared-memory usage. |
| `vmm_dxgi_system_headroom_gb` | `0.0` | `DDO_VMM_DXGI_SYSTEM_HEADROOM_GB` | Config/ENV; Windows system reserve. Linux uses native host/cgroup/process admission instead. |
| `vmm_component_tensor_ownership` | `False` | Not currently parsed from ENV | `--ddo-vmm-component-tensor-ownership` or Python; opt-in whole-component ownership, native fast only, not automatic encoder support. |
| `vmm_component_buffer_limit_mb` | `1` | Not currently parsed from ENV | `--ddo-vmm-component-buffer-limit-mb` or Python; explicit once-copied device-buffer payload ceiling, not total GPU usage. |
| `vmm_native_source_cache_slot_mb` | `0` | `DDO_VMM_NATIVE_SOURCE_CACHE_SLOT_MB` | `--ddo-vmm-native-source-cache-slot-mb`; legacy explicit micro-cache geometry, distinct from large retained-source config. |
| `vmm_native_source_cache_slots` | `0` | `DDO_VMM_NATIVE_SOURCE_CACHE_SLOTS` | `--ddo-vmm-native-source-cache-slots`; paired with slot size; zero preserves cache-off selection. |

`vmm_host_cache_gb` / `DDO_VMM_HOST_CACHE_GB` describe the older CPU-source
registration cache, not the native retained file-source buffer. Likewise,
CPU pinning and `vmm_pin_source_weights` must not be conflated with native
source registration. Keep legacy controls available for the routes that use them.

CLI, config and environment are not three universally interchangeable inputs
today: `build_config_from_args()` constructs config from the namespace and does
not itself merge `DDO_*` variables. `use_environment=False` deliberately avoids
environment overrides. Some host parsers replace environment values with their
own defaults. Application code must explicitly select one resolution path and
document precedence; do not assume an environment value wins over a supplied
config or runner argument. Changes to that precedence need separate contracts.

Bootstrap controls are process setup, not residency presets/config fields:

| Control | Meaning |
| --- | --- |
| `DDO_VMM_LIBRARY_PATH` / bootstrap `library_path` | Matching platform native library; chosen before CUDA initialization. |
| Bootstrap `allocator_cache_mib` | Global native PyTorch allocator cache; default512MiB. Bootstrap writes `DDO_VMM_TORCH_CACHE_MIB` for the child; it does not automatically honor a conflicting pre-existing cache environment value. |
| `DDO_VMM_ALLOCATOR_ROOT` | Host-harness checkout locator where supported, not a universal `DynamicOffloadConfig` option. |

See [native usage](native-vmm.md) for bootstrap/lifecycle, and
[metrics](metrics.md) for requested capacities versus executed routes and observed
bytes. The LTX fixture's historical staging CLI is per logical slot and is
converted to total capacity; do not copy that numerical value to generic DDO CLI.

The sibling LTX VMM harness also exposes `--ddo-vmm-module-pattern`,
`--ddo-vmm-page-mb` and `--ddo-vmm-dxgi-{local,shared,system}-headroom-gb`.
These map to the matching config fields above; they are host options, not
additional flags in the generic DDO argument helper. Its diagnostic adapter
is explicitly selected by `--ddo-vmm-diagnostic-adapter`; other diagnostic
controls use `--ddo-vmm-diagnostic-*`. Those controls are not native presets
and do not change the public route defaults. The renamed host options have
no compatibility aliases; older diagnostic command lines must be migrated.

### Full-pin zero-resident behavior

When balanced auto resolves to `full_pin_zero_resident`, DDO offloads even the default
`always_resident_modules_pattern` leaves and includes them in the pinned-weight pool. This
keeps the plan genuinely zero-resident and preserves CUDA headroom for activations. Set
an explicit `DDO_RESIDENT_MODULE_BUDGET_GB` to override DDO's automatic or stored-profile
choice. A stored capacity profile can internally supply a conservative residual CUDA budget
while preserving full CPU pinning; it is not a separate environment setting.

## Global Environment Variables

These variables are recognized directly by DDO:

| Variable | Purpose |
| --- | --- |
| `DDO_PRESET` | Selects the preset when code calls `DynamicOffloadSettings.from_env()` or does not pass an explicit preset. |
| `DDO_PLAN` | Boolean that enables DDO when `DDO_EXECUTION_MODE=plan`; either runtime mode also enables DDO without this flag. |
| `DDO_AVAILABLE_SYSTEM_RAM_GB` | Overrides detected available system RAM. Useful for simulating smaller RAM machines. |
| `DDO_DISABLE_PIN_ON_WSL` | Disables pinned CPU memory on WSL by default. Set to `0` to force pinning experiments. |
| `DDO_PURGE_WINDOWS_STANDBY_<PHASE>` | Enables Windows standby-list purge for a named phase. Non-Windows platforms no-op safely; on Windows it requires `SeProfileSingleProcessPrivilege`, normally from an elevated Administrator terminal. |
| `DDO_PROFILE_DIR` | Optional directory for persistent capacity profiles. Defaults to `~/.ddo/profiles`; see [capacity profiles](capacity-profiles.md). |

Boolean variables accept `1`, `true`, `yes`, or `on` as true values. Every other value, including `0`, `false`, `no`, `off`, and an empty value, resolves to false.

## Component Overrides

For pipeline-level setup, DDO can route each component independently. Component names are uppercased in environment variables.

For a component named `transformer`, DDO recognizes:

| Variable pattern | Purpose |
| --- | --- |
| `DDO_TRANSFORMER_ROUTE` | Canonical values: `auto`, `dynamic_offload`, `diffusers_group_offload`, or `none`. Aliases: `dynamic`/`ddo`, `diffusers`/`group`/`group_offload`, and `cuda`/`standard`/`off`. |
| `DDO_TRANSFORMER_GROUP_OFFLOAD` | Enables the Diffusers group offload route for this component. |
| `DDO_TRANSFORMER_DYNAMIC_OFFLOAD` | Enables the DDO dynamic offload route for this component. |
| `DDO_TRANSFORMER_OFFLOAD_STREAM` | Boolean Diffusers group-offload `use_stream` override. DDO forces it off on WSL when `DDO_DISABLE_PIN_ON_WSL` is enabled. |
| `DDO_TRANSFORMER_OFFLOAD_RECORD_STREAM` | Boolean Diffusers group-offload `record_stream` override. It is also forced off by that WSL safety rule. |
| `DDO_TRANSFORMER_OFFLOAD_TYPE` | Value forwarded to Diffusers group offload. DDO examples use `block_level` and `leaf_level`; compatibility depends on the installed Diffusers version. |
| `DDO_TRANSFORMER_OFFLOAD_LOW_CPU_MEM_USAGE` | Boolean Diffusers group offload low-CPU-memory mode. |
| `DDO_TRANSFORMER_NUM_BLOCKS_PER_GROUP` | Integer group size forwarded for `block_level`; it is ignored for `leaf_level`. |

The same pattern works for other component names, for example `DDO_TEXT_ENCODER_ROUTE`, `DDO_VAE_ROUTE`, or `DDO_CONNECTORS_ROUTE`.

`DDO_RUNNER_<COMPONENT>_*` is accepted as a compatibility alias for existing experiments, but new applications should use `DDO_<COMPONENT>_*` or Python `component_policies`.

## Preset Selection

The common public presets are:

| Preset | Intent |
| --- | --- |
| `auto` | Platform-aware default. Resolves to a safe DDO preset for the detected environment. |
| `one_shot_fast` | Optimized for one-shot generation when enough system RAM is available. |
| `low_ram_safe` | Lower VRAM/RAM pressure, slower than `one_shot_fast`. |
| `wsl_compat` | WSL-friendly route that avoids pinned CPU memory by default. |
| `diffusers_offload_compat` | Uses official Diffusers group offload compatibility routing. |
| `diffusers_leaf_offload_compat` | Uses official Diffusers leaf-level group offload compatibility routing. |
| `off` | Disables offloading and moves unmanaged modules to the execution device when hooks are applied. |

For quantized backends with custom forward paths, prefer `diffusers_offload_compat` or `diffusers_leaf_offload_compat` unless a backend-specific DDO adapter has been validated.

## Windows Standby Purge

DDO exposes standby-list purge as an optional Windows-only lifecycle helper. It is useful when a staged application frees one large component before loading another.

Example:

```python
from diffusers_dynamic_offloader import maybe_purge_windows_standby_cache

maybe_purge_windows_standby_cache("before_transformer", settings=settings)
```

The helper is safe to call on Linux, native Windows without purge support, and WSL. On unsupported platforms it returns no-op metadata. On Windows, the process token must be able to enable `SeProfileSingleProcessPrivilege`; this normally requires starting the terminal as Administrator. If that privilege is unavailable, DDO returns `purged: false` with `reason: PermissionError` and continues the run safely.

The standby purge only asks Windows to discard reclaimable standby pages. It does not release weights pinned by PyTorch/CUDA; stage-local pinned-memory cleanup is a separate concern.
