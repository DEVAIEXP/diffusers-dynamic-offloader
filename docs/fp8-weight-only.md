# Experimental FP8 weight-only support

DDO's FP8 transport is model-agnostic, not an LTX backend. The host supplies a
Diffusers `ModelMixin` class and a supported saved component; DDO inspects its
parameters and immutable physical checkpoint sources, not its model name.
Model-agnostic implementation is not a promise that every model or quantizer is
validated. Real-model evidence currently covers selected Windows LTX workloads.

## Responsibility boundary

| Owner | Responsibility |
| --- | --- |
| Diffusers / TorchAO | Quantization format, wrapper reconstruction, tensor dispatch, dequantization and model computation |
| DDO / native allocator adapter | Physical data/scale descriptors and accounting, native publication, materialization, transfers, event ownership, retirement, cleanup and reports |
| Generic optional compute experiment | Explicit FP8 Linear compilation; not required by native FP8 loading/offload and not automatically enabled |
| Model host | Pipeline orchestration and model-specific optimizations; timestep compaction, residual/RoPE reuse and feed-forward chunk-conversion reuse belong outside DDO |

The core FP8 route preserves the model and backend math forwards. Native hooks
wrap execution for ownership/transport; they do not substitute dense computation
for TorchAO dispatch. The optional `float8_compile` helper installs compute
wrappers and must be described separately, not as the FP8 compatibility contract.
The completed official-group/native comparison disabled this helper and the
host's feed-forward conversion-reuse experiment on both sides.

## Supported experimental contract

- Saved TorchAO 0.18.0 `Float8WeightOnlyConfig` per-row E4M3 data, FP32 scales,
  logical BF16, ordinary supported Linear semantics; computation remains BF16.
- Explicit `vmm_fast_runtime`, native transport, persistent source cache, full
  component ownership and `vmm_float8_weight_only=True`; no model-name detection
  or automatic quantized opt-in. Use the weight-aware local component loader.
- Immutable local safetensors sources with complete physical payload metadata.
  Data, scales, ordinary parameters and bounded root state participate in native
  ownership, accounting and event-safe retirement.
- Existing linear, NVMe and dense-native defaults remain unchanged. Other
  quantized recipes retain compatibility routing or explicit refusal; this is
  not MXFP8/NVFP4, activation quantization, arbitrary unscaled FP8 or SDNQ support.
- Bootstrap precedes CUDA-owning imports. Do not bulk-move an attached component
  to CUDA, replace backend forwards, or retain a hidden full-model BF16 cache.

See [conversion and integration](fp8-weight-only-conversion.md) for the reusable
converter and generic loading snippet. Conversion is optional preparation, not
performed on every load. The converter does not handle Transformers text encoders.
Unsupported recipes, layouts, shared owners or loading options fail closed under
the documented contract; generic architecture discovery does not waive admission.

## Accepted evidence and limitations

The October 10 delivery review passed 118 selected CPU contracts in 14.19 s
with CUDA hidden and initialization forbidden by the applicable fixtures.
It covered conversion, optional compilation graph ownership, physical sources,
quantized owners/arena, dense owner/adapter and capacity-inspection contracts.
This CPU check is not new CUDA or Linux FP8 acceptance; the selected DLL was
not rebuilt or changed.

Small CPU/native CUDA contracts cover physical source reconstruction, ownership,
streamed materialization, root state, numerical agreement with each backend's own
reference, interruption and cleanup. Selected real-model Windows FP8 generations
completed at five and ten seconds. The latest five-second/eight-step comparison
recorded 235.2354 s native denoise versus 411.2244 s official streamless leaf group,
and 316.2860 s versus 443.4385 s tracked total, with exactly equal finite video/audio
latents. Decode/encoding was excluded. See [results](../experiments/dynamic_offload_results.md).

The selected official group stack failed streamed Float8Tensor pinning and failed
streamless transfers under `torch.inference_mode()`. The host control alone uses
`no_grad()` and no stream; DDO native retains its accepted execution context.
These are bounded installed-stack compatibility findings, not general Windows
limitations or a library-wide requirement to disable inference mode. The comparison
does not measure a functioning streamed FP8 group route or prove FP8 faster than
BF16. Memory/storage savings and lower transfer bytes need not improve compute time.

Linux's selected BF16 native acceptance does not cover these newer FP8 changes.
Other models/backends/platforms require user-driven validation; no automatic
benchmark matrix, default promotion or general quality-equivalence claim follows.
The [engineering record](fp8-weight-only-integration-plan.md) preserves earlier
checkpoint labels; its historical pending statements are not current status.
