# Configuration Reference

DDO is designed to work with a small public surface:

- call `enable_offload(module_or_pipeline, preset="auto")` for the default behavior;
- pass `DynamicOffloadSettings` only when application code wants explicit control;
- use `DDO_*` environment variables as optional overrides for experiments, launch scripts, or CI.

Python arguments should be preferred for reusable application code. Environment variables are best treated as runtime overrides: useful while benchmarking, but less explicit than code.

## Override Order

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
| `plan` | Optional named plan/policy marker. |
| `config` | The resolved `DynamicOffloadConfig`. |
| `requested_pin_cpu_memory` | The pinning choice before platform safety overrides. |
| `effective_pin_cpu_memory` | The final pinning choice used by DDO. |
| `disable_pin_on_wsl` | Whether WSL should disable pinned CPU memory automatically. |
| `running_on_wsl` | Whether the current process was detected as WSL. |
| `component_policies` | Per-component route overrides. |

## DynamicOffloadConfig

`DynamicOffloadConfig` controls how DDO patches or routes an individual module.

| Field | Default | Environment variable | Notes |
| --- | --- | --- | --- |
| `execution_device` | `"cuda:0"` | Python only | Device where active compute runs. |
| `offload_device` | `"cpu"` | Python only | Device where inactive weights are kept. |
| `target_module_classes` | built-in linear classes | Python only | Module classes eligible for dynamic linear runtime patching. |
| `skip_modules_pattern` | `()` | Python only | Regex patterns for modules DDO should ignore. |
| `always_resident_modules_pattern` | `()` | `DDO_ALWAYS_RESIDENT_MODULE_PATTERNS` | Comma-separated regex patterns kept resident. |
| `small_tensor_threshold_bytes` | `16 KiB` | `DDO_SMALL_TENSOR_THRESHOLD_KB` | Small tensors are moved directly instead of dynamically copied. |
| `execution_mode` | `"plan"` | `DDO_EXECUTION_MODE` | Internal execution mode. |
| `pin_cpu_memory` | `False` | `DDO_PIN_CPU_MEMORY` | Enables pinned CPU tensor copies when supported and safe. |
| `allow_pin_memory_fallback` | `True` | `DDO_ALLOW_PIN_MEMORY_FALLBACK` | Fall back cleanly if pinning fails. |
| `pin_cpu_workers` | `1` | `DDO_PIN_CPU_WORKERS` | Worker count for pinning preparation. |
| `cache_plan` | `True` | `DDO_CACHE_PLAN` | Cache structural planning in-process. |
| `cache_pinned_weights` | `False` | `DDO_CACHE_PINNED_WEIGHTS` | Reuse pinned tensors in the same process when possible. |
| `pinned_weight_cache_namespace` | `""` | `DDO_PINNED_WEIGHT_CACHE_NAMESPACE` | Separates pinned tensor caches between model/config groups. |
| `auto_budget_policy` | `"off"` | `DDO_AUTO_BUDGET_POLICY` | Supported values: `off`, `balanced`. |
| `max_resident_module_budget_gb` | `6.0` | `DDO_MAX_RESIDENT_MODULE_BUDGET_GB` | Upper cap used by auto budgeting; `0` means uncapped. |
| `auto_vram_headroom_gb` | `0.0` | `DDO_AUTO_VRAM_HEADROOM_GB` | Extra VRAM margin reserved by auto budgeting. |
| `max_pin_weight_budget_gb` | `0.0` | `DDO_MAX_PIN_WEIGHT_BUDGET_GB` | Upper cap for auto pinning; `0` means uncapped. |
| `available_system_ram_gb` | detected | `DDO_AVAILABLE_SYSTEM_RAM_GB` | Override detected available system RAM for testing. |
| `system_ram_headroom_gb` | `6.0` | `DDO_SYSTEM_RAM_HEADROOM_GB` | RAM held back from auto pinning decisions. |
| `pin_weight_budget_gb` | `0.0` | `DDO_PIN_WEIGHT_BUDGET_GB` | Explicit pinned weight budget. |
| `pin_weight_budget_ratio` | `0.0` | `DDO_PIN_WEIGHT_BUDGET_RATIO` | Fraction of candidate weights to pin. |
| `pin_weight_selection` | `"spread"` | `DDO_PIN_WEIGHT_SELECTION` | Supported values: `first`, `spread`, `largest`. |
| `resident_module_budget_gb` | `0.0` | `DDO_RESIDENT_MODULE_BUDGET_GB` | Explicit resident module budget. |
| `resident_module_patterns` | `()` | `DDO_RESIDENT_MODULE_PATTERNS` | Comma-separated regex patterns; `auto` lets DDO infer common block patterns. |
| `resident_module_selection` | `"spread"` | `DDO_RESIDENT_MODULE_SELECTION` | Supported values: `first`, `spread`, `largest`. |
| `load_safetensors_backend` | `""` | `DDO_LOAD_SAFETENSORS_BACKEND` | Optional safetensors loading backend override. |
| `verbose` | `False` | `DDO_VERBOSE` | Prints additional setup details. |
| `show_profile` | `True` | `DDO_SHOW_PROFILE` | Prints the dynamic offload profile summary. |

`DDO_SAFETENSORS_BACKEND` is also accepted as a compatibility alias for `DDO_LOAD_SAFETENSORS_BACKEND`.

## Global Environment Variables

These variables are recognized directly by DDO:

| Variable | Purpose |
| --- | --- |
| `DDO_PRESET` | Selects the preset when code calls `DynamicOffloadSettings.from_env()` or does not pass an explicit preset. |
| `DDO_PLAN` | Optional plan/policy marker. |
| `DDO_AVAILABLE_SYSTEM_RAM_GB` | Overrides detected available system RAM. Useful for simulating smaller RAM machines. |
| `DDO_DISABLE_PIN_ON_WSL` | Disables pinned CPU memory on WSL by default. Set to `0` to force pinning experiments. |
| `DDO_PURGE_WINDOWS_STANDBY_<PHASE>` | Enables Windows standby-list purge for a named phase. Non-Windows platforms no-op safely. |

Boolean variables accept `1`, `true`, `yes`, or `on` as true values.

## Component Overrides

For pipeline-level setup, DDO can route each component independently. Component names are uppercased in environment variables.

For a component named `transformer`, DDO recognizes:

| Variable pattern | Purpose |
| --- | --- |
| `DDO_TRANSFORMER_ROUTE` | Route for this component: dynamic offload, Diffusers group offload, direct device move, or off. |
| `DDO_TRANSFORMER_GROUP_OFFLOAD` | Enables the Diffusers group offload route for this component. |
| `DDO_TRANSFORMER_DYNAMIC_OFFLOAD` | Enables the DDO dynamic offload route for this component. |
| `DDO_TRANSFORMER_OFFLOAD_STREAM` | Diffusers group offload `use_stream` override. |
| `DDO_TRANSFORMER_OFFLOAD_RECORD_STREAM` | Diffusers group offload `record_stream` override. |
| `DDO_TRANSFORMER_OFFLOAD_TYPE` | Diffusers group offload type, such as `block_level` or `leaf_level`. |
| `DDO_TRANSFORMER_OFFLOAD_LOW_CPU_MEM_USAGE` | Diffusers group offload low-CPU-memory mode. |
| `DDO_TRANSFORMER_NUM_BLOCKS_PER_GROUP` | Diffusers block-level group size. |

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

The helper is safe to call on Linux, native Windows without purge support, and WSL. On unsupported platforms it returns no-op metadata.