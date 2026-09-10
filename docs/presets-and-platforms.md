# Presets And Platforms

DDO presets are route policies, not model-specific hacks. The same preset can apply different routes to different component names.

## Presets

| Preset | Recommended for | Behavior |
| --- | --- | --- |
| `auto` | Default user-facing preset | Resolves to the current recommended policy. Today that is `one_shot_fast` plus platform and backend safety checks. |
| `one_shot_fast` | BF16/FP16 dense transformer weights on low-VRAM Windows or comparable systems | Uses DDO dynamic offload for standard dense transformer linears, selected resident modules, and RAM-aware pinned CPU weights. |
| `low_ram_safe` | Lower accelerator pressure | Uses a smaller resident-module budget and avoids expensive RAM assumptions. Slower but safer. |
| `wsl_compat` | WSL stacks where pinned memory or stream behavior is unstable | Disables pinned CPU memory and risky stream behavior. |
| `warm_process` | Server-style repeated generation | Keeps process-lifetime caches where possible. Compare separately from cold one-shot runs. |
| `diffusers_offload_compat` | Official compatibility baseline | Routes components to Diffusers block-level group offload. |
| `diffusers_leaf_offload_compat` | Quantized backends and fine-grained official offload | Routes components to Diffusers leaf-level group offload. |
| `off` | Manual control | DDO does not apply offload hooks. |

## Component Policy

Typical policy shape:

| Component | Dense BF16 default | Quantized/backend-specific default |
| --- | --- | --- |
| `transformer` / `unet` | DDO dynamic offload when beneficial | Diffusers group offload compatibility route |
| `text_encoder` | Diffusers group offload | Diffusers group offload |
| `vae` | Diffusers group offload or manual CUDA decode stage | Diffusers group offload or manual CUDA decode stage |
| small connectors / adapters | Diffusers group offload or caller-managed CUDA move | Diffusers group offload or caller-managed CUDA move |

## Platform Notes

Windows:

- `auto` / `one_shot_fast` is the main fast path from the current Windows BF16 benchmark results.
- Windows standby-list purge can help between phase-staged execution phases.
- The planner pins CPU weights only when available system RAM leaves configured headroom.

WSL:

- Prefer `wsl_compat` first.
- The measured WSL stack benefited from DDO dynamic offload with pinning disabled.
- Official Diffusers leaf group offload was unstable in one measured WSL run.

Native Linux:

- Compare `auto` with `diffusers_leaf_offload_compat`.
- DDO can improve denoise throughput, but official Diffusers leaf group offload may win cold one-shot totals when setup is much lower.
- Warm/server runs can change this tradeoff because DDO setup caches amortize across requests.

Quantized backends:

- Prefer `diffusers_leaf_offload_compat` or `diffusers_offload_compat`.
- DDO detects SDNQ-style packed modules and preserves their forward.
- Dense-linear DDO acceleration should only patch standard 2D `nn.Linear` weights.
