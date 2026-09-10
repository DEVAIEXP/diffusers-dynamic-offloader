# Phase-Staged Pipeline Usage

Phase-staged execution means running prompt encoding, denoise, and VAE decode as separate phases. This pattern is useful when each phase has a different memory shape and the host wants explicit cleanup boundaries.

The exact pipeline constructors are model-specific, but the lifecycle is generic.

## Standard Diffusers Pipeline

```python
import gc
import torch
from diffusers_dynamic_offloader import enable_offload, maybe_purge_windows_standby_cache

device = torch.device("cuda:0")
model_id = "your/model"

# Pass 0: prompt encoding
prompt_pipe = YourPipeline.from_pretrained(
    model_id,
    transformer=None,
    vae=None,
    torch_dtype=torch.bfloat16,
)
text_result = enable_offload(prompt_pipe.text_encoder, preset="auto", component="text_encoder")
prompt_embeds = prompt_pipe.encode_prompt(prompt="a small robot holding a lantern")

del text_result, prompt_pipe
gc.collect()
torch.cuda.empty_cache()
maybe_purge_windows_standby_cache("after_text_encoder")

# Pass 1: denoise
denoise_pipe = YourPipeline.from_pretrained(
    model_id,
    text_encoder=None,
    tokenizer=None,
    vae=None,
    torch_dtype=torch.bfloat16,
)
enable_offload(denoise_pipe.transformer, preset="auto", component="transformer")
maybe_purge_windows_standby_cache("before_transformer")

latents = denoise_pipe(
    prompt_embeds=prompt_embeds,
    width=1024,
    height=1024,
    num_inference_steps=8,
    output_type="latent",
).images

del denoise_pipe
gc.collect()
torch.cuda.empty_cache()

# Pass 2: decode
vae_pipe = YourPipeline.from_pretrained(
    model_id,
    text_encoder=None,
    tokenizer=None,
    transformer=None,
    torch_dtype=torch.bfloat16,
)
vae_pipe.vae.to(device)
image = vae_pipe.decode_latents(latents)
```

Some pipelines still require the VAE object during the denoise call even when returning latents. In that case, keep the VAE in the denoise-stage pipeline and route it through Diffusers group offload.

## Cleanup Rule

Release DDO result objects before cleanup if they hold references to module state:

```python
del offload_result
gc.collect()
torch.cuda.empty_cache()
```

This avoids retaining CPU/GPU tensors after the phase has ended.
