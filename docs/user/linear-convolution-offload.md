# Experimental eager convolution offload

`DynamicOffloadConfig(linear_convolution_offload=True)` explicitly extends
`linear_runtime` to dense `Conv1d`, `Conv2d` and `Conv3d` children. It is disabled
by default. Existing linear presets, native VMM and the NVMe baseline are unchanged.
No native allocator build is needed.

The public `enable_offload(..., component="vae", route="linear_convolution")`
constructs the isolated configuration inside DDO. Preset defaults select leaf,
not this experimental route; see [component strategies](component-strategies.md).

The wrapper temporarily onloads parameters using normal PyTorch allocation,
calls the original forward with its original arguments and restores source
storage in `finally`. It preserves parameter dtype and convolution math, including
subclass forwards. Copies run on the current compute stream; there is no new
prefetch stream, global synchronization, residency cache or convolution kernel.
PyTorch owns stream-ordered retirement of the temporary CUDA storage.

The initial contract requires eval mode, inference mode, CPU source weights,
explicit zero module residency, `auto_budget_policy="off"`, no calibration and
no persistent pinned-weight cache. Optional CPU pinning uses the existing DDO
pinning path. Bias and small non-convolution parameters retain the existing
linear runtime placement behavior. This does not offload activations.

Preflight refuses root convolutions, shared parameter/storage ownership,
quantized parameters, convolution child modules/buffers and existing execution
hooks. Custom forwards must not mutate parameters, retain their views or launch
work on another stream. Returned parameter views, reentry, training/grad mode,
active detach and CUDA graph capture are refused. Compile and native-allocator
combination are not accepted routes for this experiment.

Staged hosts must explicitly send decode inputs to `execution_device`; a module
with CPU source weights cannot advertise its execution device through `.device`.
Use the public result's `cleanup_on_error()`, `metrics_snapshot()` and normal
`remove_dynamic_offload()` lifecycle. `convolution_parameter` copy accounting
is emitted through the existing public report. Setup pinning counters retain
their historical linear-runtime labels, now including opted-in convolution weights.

## Validation

The public CUDA contract is `tests/test_linear_convolution_cuda.py`, enabled
with `DDO_RUN_LINEAR_CONV_CUDA_GATE=1`. It compares eager convolution outputs
with resident execution and checks source restoration after interruption,
using normal PyTorch allocation rather than the VMM DLL.

Small ownership/math tests do not establish full-model memory savings.
Selected Qwen VAE comparisons are maintained in
[DDO examples](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmarks/qwen-image-21.md).
Leaf and linear convolution have comparable measured peaks in that scenario;
do not describe convolution compatibility as a general memory reduction.
