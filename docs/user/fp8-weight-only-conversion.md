# Optional FP8 weight-only checkpoint conversion

Current support and the DDO/backend/host responsibility boundary are summarized
in [FP8 weight-only support](fp8-weight-only.md). The converter and native loader
are generic; model-specific performance comparisons belong to
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).

`scripts/convert_fp8_weight_only.py` converts one **local Diffusers component**
directory containing `config.json` and dense weights. It does not convert an
entire pipeline, download weights, load remote code or replace model math
forwards. The experimental recipe requires TorchAO 0.18.0 and the tested
Diffusers serialization support. TorchAO remains an optional dependency.

Example from the DDO repository, using an already prepared environment:

```powershell
python scripts/convert_fp8_weight_only.py `
  --source E:\Models\example-bf16\transformer `
  --output E:\Models\example-fp8\transformer `
  --max-shard-size 2GB --stream-shards
```

These paths are examples, not a command to convert the selected benchmark model.
The source must identify a supported built-in Diffusers `ModelMixin` through
`_class_name` in its configuration. Other component/remote-code loaders require
a separately reviewed integration; this utility does not handle Transformers
text encoders.

The standalone CLI disables GPU visibility before importing Torch. Both modes
convert `nn.Linear` weights using upstream `Float8WeightOnlyConfig`, leaving
other loaded tensors unchanged. They save `config.json`, the upstream
quantization configuration and safetensors data/scales. Original files are
never modified.
Source/output directories must be separate and non-nested. An existing output
directory, including a partial prior conversion, is refused. There is no force
overwrite option and no automatic cleanup of failed output.

With `--stream-shards`, only the architecture is constructed on meta. Two
tensor-at-a-time source passes collect complete upstream serialization metadata
and verify deterministic conversion/source hashes before writing output shards.
All output shards carry the complete logical metadata; no physical key suffixes
are invented. Source indices and complete architecture shapes/keys are checked.
The bounded route requires floating source tensors already in BF16 and refuses
shared parameter owners. Unconverted integer buffers are preserved.

RAM requirements are the selected output shard buffer plus the largest source
tensor and quantization/serialization working buffers, not the full transformer.
An individual logical tensor larger than the shard limit is written alone;
the limit is not a hard process-RSS ceiling. Validation loads one output shard
at a time. The report records the largest input tensor and peak retained output
buffer; neither field is a sampled peak RSS measurement.

Without `--stream-shards`, the original full-CPU-model mode remains unchanged:
allow RAM for the entire source component plus working tensors. It serializes
through `save_pretrained`; output sharding does not bound initial model loading.
Shared linear weight storage is refused pending a validated alias recipe.

`conversion_report.json` records dependency versions, recipe, component class,
converted linear count, per-tensor payload geometry/hashes and output shards.
Only `status: passed` with `serialized_reload_exact: true` means conversion
completed. A failed report and partial checkpoint are diagnostics, not usable
weights. Reconstruction must preserve every converted FP8 byte and FP32 scale,
as well as every unconverted tensor. The report does not establish BF16/FP8
quality equivalence, model-forward correctness after loading, lower VRAM or
faster inference.

## Explicit experimental DDO loader integration

The loader now accepts saved composite weights only when
`DynamicOffloadConfig.vmm_float8_weight_only=True`. It requires
`vmm_fast_runtime`, native transport, persistent source cache, full-component
ownership, full `.*` selection and no execution-auxiliary ownership. The default
is false. Linear and legacy NVMe paths do not accept this opt-in.

Use the existing native bootstrap before importing Torch/Diffusers, as
documented in [native integration](native-vmm.md). With an already configured
native fast component:

```python
from dataclasses import replace
from diffusers_dynamic_offloader import from_pretrained_with_dynamic_offload

fp8_config = replace(
    existing_native_config,
    vmm_float8_weight_only=True,
    vmm_component_tensor_ownership=True,
    vmm_module_pattern=".*",
    vmm_execution_auxiliary_ownership=False,
    profile_name=None,
    build_profile=False,
)
loaded = from_pretrained_with_dynamic_offload(
    converted_model_root,
    model_loader=component_class,
    subfolder="transformer",
    local_files_only=True,
    dynamic_offload_config=fp8_config,
    apply_dynamic=False,
)
```

The component class is supplied by the host. The DDO loader validates the
checkpoint's upstream recipe, local shard index and complete wrapper metadata,
constructs the architecture on meta, and reconstructs saved CPU weights through
Diffusers' TorchAO deserializer. It binds immutable sources to the final model
payload owners using upstream serialized names, not guessed suffixes. It does
not quantize on load or move the complete model to CUDA. Later attachment uses
the existing public offload lifecycle and explicit FP8 config; never call bulk
`module.to("cuda")` on this native component.

Device maps, arbitrary key mappings, remote component loading and other loader
options are deliberately refused for the initial route. Unknown recipes,
partial/ambiguous sources and incomplete reconstruction fail before attachment.
The dense quantized-payload admission guard remains unchanged.

The normal native persistent cache resolver uses 64 MiB segments, including
registration ceilings. Do not reuse the small direct-adapter probe's 2 MiB
cache segments as public runner settings. Existing accepted production budgets
are not automatically raised. Public metrics expose
`dynamic_offload_vmm_float8_weight_only`; native file-backed byte metrics describe
physical payloads, whereas the legacy logical tensor/planner sizes are not an
FP8 physical-memory measurement.

Bounded plain root parameters can be retained as component-lifetime state.
Root payload and buffers must fit `vmm_component_buffer_limit_mb`; their
page-rounded backing is subtracted from the existing weight ceiling. Root-only
aliases are preserved; cross-scope aliases and root subclasses are refused.
Publication retains immutable source descriptors, not loader CPU tensors.

Conversion success establishes serialization, not inference quality or speed.
Use [native validation](../developer/validation.md) and a matched model benchmark
to evaluate the converted checkpoint.
