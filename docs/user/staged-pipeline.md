# Staged pipeline

Phase-staged execution means running prompt encoding, denoise, and VAE decode as separate phases. This pattern is useful when each phase has a different memory shape and the host wants explicit cleanup boundaries.

The exact constructors are model-specific. The lifecycle applies to both linear
and VMM Fast. Start with [integration](integration.md) for backend selection;
for VMM Fast, bootstrap before Torch/model imports and use its source-aware
configuration at transformer loading and attachment. The schematic example
below uses linear settings and application-provided pipeline/model classes.

## Common phase structure

```python
import gc
import torch
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_offload,
    from_pretrained_with_dynamic_offload,
    remove_dynamic_offload,
)

device = torch.device("cuda:0")
model_id = "your/model"
settings = DynamicOffloadSettings.from_env(
    execution_device=device,
    offload_device="cpu",
    default_preset="auto",
)

# Pass 0: prompt encoding
prompt_pipe = YourPipeline.from_pretrained(
    model_id,
    transformer=None,
    vae=None,
    torch_dtype=torch.bfloat16,
)
text_result = enable_offload(prompt_pipe.text_encoder, preset="auto", component="text_encoder")
prompt_embeds = prompt_pipe.encode_prompt(prompt="a small robot holding a lantern")
# Preserve only the inputs required by the next phase, on CPU where appropriate.
prompt_embeds = prompt_embeds.to("cpu")

del text_result, prompt_pipe
gc.collect()
torch.cuda.empty_cache()

# Pass 1: denoise. Load the memory-heavy component first so the host owns its
# loading and offload setup independently of pipeline construction.
transformer_load = from_pretrained_with_dynamic_offload(
    model_id,
    model_loader=YourTransformerClass,
    dynamic_offload_config=settings.config,
    apply_dynamic=False,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
denoise_pipe = YourPipeline.from_pretrained(
    model_id,
    transformer=transformer_load.module,
    text_encoder=None,
    tokenizer=None,
    vae=None,
    torch_dtype=torch.bfloat16,
)
transformer_result = enable_offload(
    denoise_pipe.transformer, settings=settings, component="transformer",
)

latents = denoise_pipe(
    prompt_embeds=prompt_embeds.to(device),
    width=1024,
    height=1024,
    num_inference_steps=8,
    output_type="latent",
).images.to("cpu")

report = transformer_result.metrics_snapshot(collect_completed=True)
remove_dynamic_offload(transformer_result.module)
del transformer_result, transformer_load, denoise_pipe
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
image = vae_pipe.decode_latents(latents.to(device))
```

Some pipelines still require the VAE object during the denoise call even when returning latents. In that case, keep the VAE in the denoise-stage pipeline and route it through Diffusers group offload.

For the helper's loading contract, when to use `apply_dynamic`, and measurement scope, see [Dynamic loading](dynamic-loading.md).

## Cleanup and reuse

On success, collect the completed report, retire DDO's hook with
`remove_dynamic_offload()`, then drop every pipeline, loader, result and state
reference that owns the component. Deleting only the pipeline leaves weights
alive when a loading/result object still refers to them. Official group hooks
remain owned by Diffusers.

Wrap attached inference in `result.cleanup_on_error()` as shown in
[integration](integration.md#run-report-and-retire). Cleanup errors must preserve
the original failure. Native unload can invalidate weights to meta placeholders;
reload before reuse. Pending or unsafe CUDA completion never authorizes force-free.

`torch.cuda.empty_cache()` can release unused PyTorch cache but does not prove
that native idle backing or external workspaces returned to the driver.
Explicit native idle-cache trim, PyTorch pinned-host cache release and Windows
standby purge are distinct operations. Routing does not enable them automatically;
see [component strategies](component-strategies.md#phase-boundaries-and-reporting)
and [configuration](configuration.md#windows-standby-purge).

Keep calibration separate from full generation. When building a profile, stop
after the measured denoising region and save its report without decoding.
