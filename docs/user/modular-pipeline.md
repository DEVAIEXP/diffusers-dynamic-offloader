# Modular Diffusers

Modular Diffusers lets the host construct only the blocks needed for a phase.
The offload APIs and choice between linear and VMM Fast are the same as in
[integration](integration.md); custom blocks still own the model computation.

Review custom code before enabling `trust_remote_code`. Select a known revision
and record the resolved code and weight snapshots. Local custom-block checkouts
and Hub code revisions are different sources; one does not update the other.

## Full pipeline

A modular pipeline can load its components and delegate their routing to DDO:

```python
import torch
from diffusers import ModularPipeline
from diffusers_dynamic_offloader import enable_pipeline_offload

pipe = ModularPipeline.from_pretrained(
    "your/custom_blocks", trust_remote_code=True,
)
pipe.load_components(
    pretrained_model_name_or_path="your/model",
    names=["text_encoder", "tokenizer", "transformer", "vae", "scheduler"],
    torch_dtype=torch.bfloat16,
)
results = enable_pipeline_offload(
    pipe, preset="one_shot_fast", execution_device="cuda:0",
    use_environment=False,
)
```

This example selects linear policies. For VMM Fast, bootstrap first and load the
transformer with DDO's source-aware helper, then inject the prepared component
using the API exposed by the blocks. Configure remaining components separately;
do not attach offload twice to the transformer. See the
[full-pipeline pattern](full-pipeline.md#vmm-fast-setup).

## Staged pipeline

The host can instead build prompt-only, denoise-only and decode-only blocks:

| Phase | Components and output | Boundary |
| --- | --- | --- |
| Prompt encoding | Encoder/tokenizer produce conditioning embeddings | Keep required embeddings, retire the encoder and drop its state |
| Denoising | Scheduler and prepared transformer produce latents | Read back needed outputs, collect the report and retire transformer ownership |
| VAE decode | Decoder consumes latents on its execution device | Convert output and save after the isolated decode phase |

For the denoising block, associate checkpoint sources before attachment.
The attachment works with either family through the common configuration:

```python
# attach_component() is defined in the common integration example.
result = attach_component(local_transformer_snapshot, TransformerClass)
denoise_pipe.update_components(transformer=result.module)
with result.cleanup_on_error():
    with result.profile_run():
        state = denoise_pipe(
            prompt_embeds=prompt_embeds.to("cuda:0"), output="state",
        )
    latents = state.latents.to("cpu")
    report = result.metrics_snapshot(collect_completed=True)
```

Names such as `prompt_embeds`, `latents`, `output="state"` and
`update_components()` depend on the blocks. Adapt those calls to the selected
model. DDO owns offload configuration, not modular pipeline construction.

Use the [staged cleanup rules](staged-pipeline.md#cleanup-and-reuse) before
loading the next component. Drop the returned state as well as pipeline and
result references. A persistent [capacity profile](capacity-profiles.md) measures
only the chosen region; stop before decode when calibrating. Model-specific
chunking, attention and VAE streaming options belong to the blocks/host and
must be recorded separately from the offload preset.
