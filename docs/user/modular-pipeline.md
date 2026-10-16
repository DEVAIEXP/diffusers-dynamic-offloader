# Modular Pipeline Usage

Modular Diffusers makes phase-staged execution cleaner because the host can build and call only the blocks required for each phase.

## Full Modular Pipeline

```python
import torch
from diffusers import ModularPipeline
from diffusers_dynamic_offloader import enable_pipeline_offload

pipe = ModularPipeline.from_pretrained("your/custom_blocks", trust_remote_code=True)
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

## Phase-Staged Modular Pipeline

The phase-staged modular pattern keeps each memory-heavy phase explicit:

```python
import gc
import torch
from diffusers import ModularPipeline
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_offload,
    from_pretrained_with_dynamic_offload,
    maybe_purge_windows_standby_cache,
)

device = torch.device("cuda:0")
blocks_id = "your/custom_blocks"
model_id = "your/model"
settings = DynamicOffloadSettings.from_env(
    execution_device=device,
    offload_device="cpu",
    default_preset="auto",
)

# Pass 0: prompt only
prompt_pipe = ModularPipeline.from_pretrained(blocks_id, trust_remote_code=True)
prompt_pipe.load_components(
    pretrained_model_name_or_path=model_id,
    names=["text_encoder", "tokenizer"],
    torch_dtype=torch.bfloat16,
)
enable_offload(prompt_pipe.components["text_encoder"], preset="auto", component="text_encoder")
prompt_state = prompt_pipe(prompt="a small robot holding a lantern", output="state")
prompt_embeds = prompt_state.prompt_embeds.to("cpu")

del prompt_pipe, prompt_state
gc.collect()
torch.cuda.empty_cache()
maybe_purge_windows_standby_cache("after_text_encoder")

# Pass 1: denoise only. When the block API accepts a prebuilt component,
# load the transformer independently before attaching it to the denoise block.
transformer_load = from_pretrained_with_dynamic_offload(
    model_id,
    model_loader=YourTransformerClass,
    dynamic_offload_config=settings.config,
    apply_dynamic=False,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
denoise_pipe = ModularPipeline.from_pretrained(blocks_id, trust_remote_code=True)
denoise_pipe.load_components(
    pretrained_model_name_or_path=model_id,
    names=["scheduler"],
    torch_dtype=torch.bfloat16,
)
denoise_pipe.update_components(transformer=transformer_load.module)
enable_offload(denoise_pipe.components["transformer"], settings=settings, component="transformer")
maybe_purge_windows_standby_cache("before_transformer")

denoise_state = denoise_pipe(
    prompt_embeds=prompt_embeds.to(device),
    width=1024,
    height=1024,
    num_inference_steps=8,
    output="state",
)
latents = denoise_state.latents.to("cpu")

del denoise_pipe, denoise_state
gc.collect()
torch.cuda.empty_cache()

# Pass 2: decode only
decode_pipe = ModularPipeline.from_pretrained(blocks_id, trust_remote_code=True)
decode_pipe.load_components(
    pretrained_model_name_or_path=model_id,
    names=["vae"],
    torch_dtype=torch.bfloat16,
)
decode_pipe.components["vae"].to(device)
images = decode_pipe(latents=latents.to(device), output="images")
images[0].save("out.png")
```

Names such as `prompt_embeds`, `latents`, and `output="state"` depend on the custom block implementation. The DDO part is the same: call `enable_offload(...)` on the module for that phase and cleanup after the phase finishes.

`update_components(...)` is illustrative: attach the preloaded module using the API exposed by the chosen blocks. See [Dynamic module loading](dynamic-loading.md) for the helper contract and the benchmark rule to use one transformer loading path across comparable runners.
