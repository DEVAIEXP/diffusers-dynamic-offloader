# Optional FP8 weight-only checkpoint conversion

Current support and the DDO/backend/host responsibility boundary are summarized
in [FP8 weight-only support](fp8-weight-only.md). The converter and native loader
are generic; the model-specific checkpoints later on this page are historical
evidence, not runtime dependencies. Selected real-model Windows execution and the
latest group/native comparison have now completed; retain older pending labels
as dated checkpoints rather than current support statements.

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

Selected CPU validation passed 28 tests, including actual Diffusers component
save/load, single/split-shard reconstruction, original-file preservation,
overwrite/nested-path refusal and failed-output diagnostics, alongside existing
FP8 payload/source contracts. No CUDA or real LTX conversion ran in this block.
The earlier small native CUDA gate independently accepted the selected FP8
payload ownership/streaming mechanism; it is not full-model acceptance.

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

Selected CPU integration/regression validation passed 123 tests and six
subtests, including exact payload hashes through the explicit loader, single
and split shards, complete native geometry discovery, adapter dispatch doubles,
unsupported-route refusals and host parsing. The separately approved small
public CUDA gate passed BF16 and FP8 with exact outputs against each case's own
reference, streamed materialization, recovery after intentional RuntimeError
and KeyboardInterrupt, and complete cleanup. Evidence is saved under
`E:/LTX-2/outputs/fp8_public_gate_02`. This is not a real-model quality or
performance result. Real-model generation remains pending.

The selected LTX BF16 transformer occupies approximately 35.4 GiB. At the
resource check, free physical RAM was approximately 36.7 GiB, leaving too little
margin for the current full-component converter's additional working tensors.
Drive E: had approximately 4.8 GiB free, insufficient for the converted model;
drive F: had approximately 326 GiB free. The bounded conversion option is now
implemented and CPU-tested. The selected separate destination is
`F:/models/ltx23-distilled-fp8-weight-only/transformer`. The approved real
conversion completed on 2026-10-09: ten shards, 17.7235 GiB, 1,660 converted
linear weights, exact payload reconstruction and no CUDA initialization.
Original BF16 files remain unchanged. `conversion_report.json` has status
`passed`. This accepts serialization only; real-model DDO generation, quality
and performance have not yet been validated. See
[integration plan](fp8-weight-only-integration-plan.md).

The first real inference attempt stopped in CPU loading before denoising because
temporary upstream flatten-clone addresses were reused across shards. The
corrected loader validates those temporary clones per shard and retains global
source binding only for final model owners. Real CPU loading now passes with
4,186 logical tensors, 5,846 immutable sources and exact conversion-report
payload hashes. Corrected real CUDA generation is still pending.

FP8 native component discovery now supports bounded plain root parameters as
component-lifetime state, retaining the dense overlap guard on other routes.
Root payload plus component buffers must fit `vmm_component_buffer_limit_mb`.
Root-only aliases are preserved; cross-scope aliases and root subclasses are
refused. Immutable descriptors remain mandatory. One existing single native
buffer uploads this state once, retains it through all child execution, and
retires it after the final consumer event. Its actual page-rounded capacity is
subtracted from the unchanged weight ceiling; publication never retains CPU
loader tensors. Public residency totals include that native backing.

Selected CPU contracts passed 138 tests and six subtests. The real checkpoint's
CPU geometry passed with 24 KiB root state and 76 disjoint execution units.
The extension still requires a small approved CUDA gate before real-model
generation; no quality/performance acceptance is implied.

That subsequent approved small root-state CUDA gate passed on 2026-10-09:
BF16 control and FP8 candidate, exact references/payloads, once-only root
upload, ceiling-inclusive residency, interruption recovery and cleanup.
Reports live in `E:/LTX-2/outputs/fp8_root_public_gate_01`. Real-model
generation, quality and performance remain pending; the small gate does not
authorize or establish those results.

The separately approved real-model retry subsequently completed on 2026-10-09:
`E:/LTX-2/outputs/fp8_real_5s_4steps_gate_03`, 1280x704, 5 s, four steps with
image/prompt/audio conditioning, native BF16 encoder and saved FP8 transformer.
It completed cleanup without a reported error; native faults, temporary
fallbacks and cache admission denials were zero. CPU validation found finite
saved video/audio latents. This accepts the selected execution only: no decoded
quality review or matched BF16 speed/quality comparison has passed. Detailed
conditions and timings are in the integration plan and benchmark sidecar.
