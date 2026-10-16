# Full Pipeline Usage

Full-pipeline integration is the smallest DDO API surface. Use it when the model already runs and you want DDO to choose per-component routes from a preset.

```python
import torch
from diffusers import DiffusionPipeline
from diffusers_dynamic_offloader import enable_pipeline_offload

pipe = DiffusionPipeline.from_pretrained(
    "your/model",
    torch_dtype=torch.bfloat16,
)

results = enable_pipeline_offload(
    pipe,
    preset="auto",
    execution_device="cuda:0",
    offload_device="cpu",
)


image = pipe(
    prompt="a small robot holding a lantern",
    width=1024,
    height=1024,
    num_inference_steps=8,
    output_type="pil",
).images[0]
image.save("out.png")
```

DDO looks for `pipe.components` first. If the pipeline does not expose it, it tries common Diffusers component names such as `text_encoder`, `text_encoder_2`, `transformer`, `unet`, and `vae`.

## Tradeoff

The full-pipeline path is convenient, but the pipeline controls the whole call. That means DDO cannot unload the text encoder before denoise, purge Windows standby memory between stages, or load the VAE only at decode time. For tight memory or cleaner benchmarks, use phase-staged execution.

