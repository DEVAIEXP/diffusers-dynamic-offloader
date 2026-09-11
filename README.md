# Diffusers Dynamic Offloader

`diffusers-dynamic-offloader` (DDO) is a model-agnostic offload router for Diffusers-style inference on low-VRAM systems.

DDO can either apply its own dense-linear dynamic offload path or route a component to the official Diffusers group-offload implementation. The intent is to give runners a small API surface:

```python
from diffusers_dynamic_offloader import enable_offload

result = enable_offload(transformer, preset="auto", component="transformer")
transformer = result.module
```

The library resolves the preset, platform behavior, component policy, RAM/VRAM budget, quantized-backend compatibility, and optional Windows standby-list cleanup helpers. Host scripts should not need to decide manually whether a given component should use DDO dynamic offload, Diffusers group offload, or no offload.

## Install

From a host project:

```powershell
uv add git+https://github.com/DEVAIEXP/diffusers-dynamic-offloader.git
```

For local development:

```powershell
git clone https://github.com/DEVAIEXP/diffusers-dynamic-offloader.git
cd diffusers-dynamic-offloader
uv pip install -e .
```

Or, from another project environment, install the local checkout by path:

```powershell
uv pip install -e C:\path\to\diffusers-dynamic-offloader
```

## Minimal Full-Pipeline Usage

Use `enable_pipeline_offload(...)` when you have a complete Diffusers pipeline and want the shortest integration.

```python
import torch
from diffusers import DiffusionPipeline
from diffusers_dynamic_offloader import enable_pipeline_offload

pipe = DiffusionPipeline.from_pretrained(
    "your/model",
    torch_dtype=torch.bfloat16,
)

enable_pipeline_offload(pipe, preset="auto", execution_device="cuda:0")

image = pipe(prompt="a small robot holding a lantern", output_type="pil").images[0]
image.save("out.png")
```

This is the simplest path. It is also the least memory-aware path because the pipeline owns the full call and DDO cannot insert cleanup between prompt encoding, denoise, and VAE decode.

## Minimal Modular Usage

For Modular Diffusers, load the custom blocks and components, then apply DDO to the exposed modules:

```python
import torch
from diffusers import ModularPipeline
from diffusers_dynamic_offloader import enable_pipeline_offload

pipe = ModularPipeline.from_pretrained(
    "your/custom_blocks",
    trust_remote_code=True,
)
pipe.load_components(
    pretrained_model_name_or_path="your/model",
    names=["text_encoder", "tokenizer", "transformer", "vae", "scheduler"],
    torch_dtype=torch.bfloat16,
)

enable_pipeline_offload(pipe, preset="auto", execution_device="cuda:0")

image = pipe(
    prompt="a small robot holding a lantern",
    width=1024,
    height=1024,
    num_inference_steps=8,
    output="images",
)[0]
image.save("out.png")
```

When a component cannot be accelerated safely, DDO routes it through the compatibility path instead of replacing its forward.

## Usage Guides

- [Full pipeline usage](docs/full-pipeline.md): the smallest integration for existing Diffusers pipelines.
- [Phase-staged pipeline usage](docs/staged-pipeline.md): standard Diffusers pipelines split into prompt, denoise, and decode phases.
- [Modular pipeline usage](docs/modular-pipeline.md): Modular Diffusers full-pipeline and phase-staged examples.
- [Dynamic module loading](docs/dynamic-loading.md): load a component independently before attaching and routing it.
- [Presets and platforms](docs/presets-and-platforms.md): what each preset means and what `auto` currently chooses.
- [Configuration reference](docs/configuration.md): settings fields, recognized `DDO_*` variables, and override precedence.
- [Dynamic offload results](experiments/dynamic_offload_results.md): report-ready benchmark summary and comparison tables.

## Recommended Patterns

Use a full pipeline when:

- you want the smallest code change;
- the model already fits comfortably enough;
- you mainly want one preset API over Diffusers group offload and DDO dynamic offload.

Use phase-staged execution when:

- VRAM or system RAM is tight;
- prompt encoding, denoise, and VAE decode have very different memory shapes;
- you want Windows standby-list cleanup and CUDA cleanup between phases;
- you need apples-to-apples timing between full-pipeline and phase-by-phase execution.

Use official Diffusers group offload through DDO when:

- the model uses a quantized backend such as SDNQ, bitsandbytes, TorchAO, or Quanto;
- the backend has custom packed weights or custom forward logic;
- compatibility matters more than dense-linear streaming speed.

## Presets

| Preset | Main use | Transformer route | Notes |
| --- | --- | --- | --- |
| `auto` | Default entry point | Resolves to the recommended policy for the current DDO release | Current policy prefers `one_shot_fast`, with platform and backend safety checks. |
| `one_shot_fast` | Best Windows low-VRAM BF16 path found so far | DDO dynamic offload for dense transformer linears | Uses RAM-aware CPU pinning and a VRAM-headroom-aware resident-module budget. |
| `low_ram_safe` | Explicit constrained-memory fallback | DDO dynamic offload with lower resident pressure | Slower, but useful when peak accelerator memory matters more than latency. |
| `wsl_compat` | WSL stability fallback | DDO dynamic offload with WSL-safe settings | Avoids pinned CPU memory and stream combinations that were unstable in testing. |
| `warm_process` | Server/repeated generations | DDO dynamic offload with process-lifetime caches | Not a cold one-image latency preset. Useful for warm runners and services. |
| `diffusers_offload_compat` | Official block-level baseline | Diffusers group offload | Compatibility path and benchmark baseline. |
| `diffusers_leaf_offload_compat` | Official leaf-level baseline | Diffusers group offload | Often best for quantized backends and some native-Linux cold runs. |
| `off` | Disable offload routing | None | Caller must move modules/devices manually. |

## Platform Behavior

`auto` is intentionally model-agnostic. It does not hardcode LTX, FLUX, Z-Image, or any other model family. It inspects the component, available RAM, CUDA VRAM, platform, and preset policy.

| Platform | Recommended first try | Why |
| --- | --- | --- |
| Windows + low VRAM + enough system RAM | `auto` / `one_shot_fast` | Best measured DDO win: faster denoise than official Diffusers group offload while keeping memory bounded. |
| Windows + limited system RAM | `auto`, then `low_ram_safe` if needed | `auto` skips pinned CPU weights when RAM headroom is not enough; denoise is slower but safer. |
| WSL | `wsl_compat` | WSL showed unstable pin/stream behavior; the compatibility preset disables the risky parts. |
| Native Linux | compare `auto` and `diffusers_leaf_offload_compat` | DDO can improve denoise, but official Diffusers leaf offload may win cold one-shot totals when setup cost dominates. |
| Quantized backends | `diffusers_leaf_offload_compat` or `diffusers_offload_compat` | DDO preserves backend-specific forwards and does not replace packed quantized matmuls. |

## Quantized Models

DDO's acceleration path targets standard dense PyTorch `nn.Linear` modules with 2D weights. Quantized integrations frequently store packed weights and implement their own dequantization, layout transform, Hadamard transform, or matmul path.

For those modules, DDO preserves the backend forward and routes to the Diffusers compatibility path. This is deliberate: replacing a quantized forward may be faster in one local experiment but fragile across backend versions and hardware.

## Configuration

Application code can pass explicit settings. See the [configuration reference](docs/configuration.md) for every settings field and DDO-recognized environment variable:

```python
from diffusers_dynamic_offloader import DynamicOffloadSettings, enable_offload

settings = DynamicOffloadSettings.from_env(
    default_preset="auto",
    execution_device="cuda:0",
    offload_device="cpu",
)

result = enable_offload(
    transformer,
    settings=settings,
    component="transformer",
    max_resident_module_budget_gb=6.0,
)
```

Environment variables are optional overrides, useful for runners and benchmarks:

```powershell
$env:DDO_PRESET="one_shot_fast"
$env:DDO_MAX_RESIDENT_MODULE_BUDGET_GB="6"
$env:DDO_SHOW_PROFILE="1"
```

Runner-specific variables should use a separate prefix such as `DDO_RUNNER_*`; DDO itself only owns the `DDO_*` library settings.

## Results

The current report is in [experiments/dynamic_offload_results.md](experiments/dynamic_offload_results.md). It includes:

- Windows, WSL, and native-Ubuntu BF16 benchmark comparisons;
- DDO dynamic offload versus official Diffusers block/leaf group offload;
- full-pipeline versus phase-staged execution comparisons;
- quantized SDNQ compatibility notes;
- exploratory FLUX.2 Klein and Z-Image observations for auto-budget behavior.

## Project Scope

DDO intentionally avoids native low-level memory hooks. The priority is a portable PyTorch/Diffusers layer that is easy to integrate, easy to disable, and safe across different hardware and driver stacks.
