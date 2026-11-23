# Full pipeline

Full-pipeline integration lets the existing pipeline orchestrate prompt encoding,
denoising and decode. DDO selects offload per component. Both linear and VMM can
participate, with different preparation requirements.

## Common flow

Load the pipeline or its prepared components, attach each route once, then call
the pipeline normally. Model identifiers, constructors and inference arguments
below are placeholders; use the model's supported API.

## Linear setup

The smallest integration uses the linear default:

```python
import torch
from diffusers import DiffusionPipeline
from diffusers_dynamic_offloader import enable_pipeline_offload

pipe = DiffusionPipeline.from_pretrained(
    "your/model", torch_dtype=torch.bfloat16,
)
results = enable_pipeline_offload(
    pipe,
    preset="one_shot_fast",
    execution_device="cuda:0",
    offload_device="cpu",
    use_environment=False,
)

image = pipe(prompt="a small robot holding a lantern").images[0]
image.save("out.png")
```

DDO looks for `pipe.components` first, then common component names when needed.
`enable_pipeline_offload()` delegates route resolution to `enable_offload()`
for each module. See [component strategies](component-strategies.md) for VAE
defaults and explicit overrides.

## VMM Fast setup

Perform the common [integration](integration.md) bootstrap before importing
Torch or the pipeline. Load and attach the transformer through the source-aware
component helper, then inject the returned module into the pipeline:

```python
# attach_component() is defined in the common integration example.
result = attach_component(local_transformer_snapshot, TransformerClass)
pipe = DiffusionPipeline.from_pretrained(
    "your/model",
    transformer=result.module,
    torch_dtype=torch.bfloat16,
)
```

Configure only the remaining components, for example with an explicit official
Leaf encoder:

```python
from diffusers_dynamic_offloader import enable_offload

encoder_result = enable_offload(
    pipe.text_encoder,
    component="text_encoder",
    route="diffusers_group_offload",
    offload_type="leaf_level",
    use_stream=False,
    execution_device="cuda:0",
    use_environment=False,
)
vae_result = enable_offload(
    pipe.vae,
    component="vae",
    route="diffusers_group_offload",
    offload_type="leaf_level",
    use_stream=False,
    execution_device="cuda:0",
    use_environment=False,
)
```

Do not call `enable_pipeline_offload()` over the already attached transformer.
Use `components=` to restrict a pipeline-level call to the remaining modules,
or configure them individually as above. The pipeline must use the intended
execution device even while weights are stored on CPU.

Native encoder/VAE ownership is optional and requires its own supported sources
and capacity configuration. It is not enabled merely by selecting a native
transformer. Call the pipeline inside `result.cleanup_on_error()` and retire the
DDO component explicitly at the final boundary described in
[integration](integration.md#run-report-and-retire).

## When to use staged execution

A full pipeline can retain all its component objects and controls its internal
phase transitions. Attaching DDO does not defer VAE construction until decode or
automatically delete the encoder after encoding. If loading or intermediate
retention is too expensive, use [staged execution](staged-pipeline.md) or
[Modular Diffusers](modular-pipeline.md).

