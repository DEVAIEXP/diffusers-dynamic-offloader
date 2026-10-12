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

CPU ownership and preset tests are not CUDA or performance acceptance. The
small normal-allocator gate is `tests/test_linear_convolution_cuda.py`, disabled
unless `DDO_RUN_LINEAR_CONV_CUDA_GATE=1`. After approval, compare the same Qwen
transformer linear workload with VAE leaf versus this option: exact PNG equality,
absolute driver decode peak, decode duration and complete cleanup. Leaf remains
the selected capacity route until the candidate passes this comparison.

## Selected Windows CUDA gate (2026-10-11)

The explicitly approved normal-PyTorch-allocator gate completed with
`4 passed in 5.30s`. BF16 Conv1d/Conv2d/Conv3d plus GroupNorm matched resident
execution exactly (`rtol=0`, `atol=0`) across three repetitions per case.
A custom causal Conv2d forward also matched exactly; an injected KeyboardInterrupt
restored the CPU weight, followed by successful CUDA synchronization and detach.
The gate asserted allocator backend `native`; it did not use or rebuild the VMM DLL.

Reproduction, from the `ddo-examples` environment:

```powershell
$env:DDO_RUN_LINEAR_CONV_CUDA_GATE = "1"
try {
    .\.venv\Scripts\python.exe -B -m pytest E:\ProjetosIA\diffusers-dynamic-offloader\tests\test_linear_convolution_cuda.py -q -p no:cacheprovider
} finally {
    Remove-Item Env:\DDO_RUN_LINEAR_CONV_CUDA_GATE
}
```

This accepts only the small eager ownership/math contract. It does not measure
full-model VAE memory or establish an advantage over official leaf offload.
The matched Qwen VAE comparison subsequently completed: leaf decode measured
1.5958 s / 6.469936 GiB sampled absolute GPU peak; linear convolution measured
1.3642 s / 6.503990 GiB. Both 20-step PNGs matched SHA256
`1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45`.
This accepts selected model compatibility, not a memory reduction. The latest
preset-routing migration has CPU contract coverage but no new CUDA benchmark.
