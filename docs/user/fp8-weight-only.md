# Experimental FP8 weight-only support

DDO's FP8 transport is model-agnostic, not an LTX backend. The host supplies a
Diffusers `ModelMixin` class and a supported saved component; DDO inspects its
parameters and immutable physical checkpoint sources, not its model name.
Model-agnostic implementation is not a promise that every model or quantizer is
validated. Real-model evidence covers selected Windows and native-Linux LTX workloads.

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

## Limitations and validation

Small native contracts check payload reconstruction, root state, event-safe
materialization, interruption and cleanup against the backend's own reference.
Selected real-model Windows and native-Linux LTX workloads are covered; this
does not establish arbitrary-model, WSL FP8 or perceptual-equivalence support.

The selected official-group control required no-stream offload and
`torch.no_grad()` because that installed stack rejected streamed Float8Tensor
pinning and streamless transfers under inference mode. This is a bounded
compatibility finding, not a requirement to change every DDO execution context.

FP8 saves payload storage and transfer bytes, but dequantization can offset
those savings. It is not guaranteed faster than BF16. Model comparisons and
quality scope belong to the
[video benchmarks](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmarks/ltx23-video.md).
See [validation](../developer/validation.md) for native contract coverage.
