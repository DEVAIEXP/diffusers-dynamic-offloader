# Native convolution component gate

Status: CPU contracts and the small CUDA convolution gate passed; real-VAE
acceptance pending.
No runtime defaults, protected paths or selected DLLs were changed.

The existing opt-in `vmm_component_tensor_ownership` adapter discovers all
immutable file-backed component parameters, including convolution and
normalization parameters. The default dense-Linear selector does not select
these convolutions. Reuse component ownership rather than adding a parallel
convolution selector or substituting convolution forward implementations.

CPU validation: 23 tests passed in `tests/test_vmm_component_adapter_cpu.py`,
including new Conv1d/Conv2d/Conv3d publication contracts. A component's explicit
`decode()` method does not call its root `forward()`. Parameters belong to
child execution units; publication preserves their native forward, drops CPU
loader references, and retirement replaces published parameters with meta
placeholders. These tests mock native views/events: they do not validate CUDA
kernels, transfer ownership, residency capacity or performance.

## Loaded-source preflight

Qwen Image 2.1 snapshot `d26bb61231c349cf6b7896fa83353113880e1ba3`
contains 238 VAE parameters stored in FP32. The selected runner loads them as
BF16. CPU-only inspection found zero immutable source bindings after that dtype
conversion; loading in checkpoint FP32 instead bound all 238 parameters.
CUDA remained uninitialized. A mismatched source must not be passed to native
transport or silently interpreted as BF16 bytes.

The FP32 component layout has 24 execution units, 1,350,961,616 unique parameter
bytes, no buffers, and a largest execution-slot payload of 345,116,160 bytes
(two slots: 690,232,320 bytes). This is source geometry, not native admission.
It includes the encoder even when only decode executes. The existing component
layout groups outer ModuleList children, not individual convolution leaves.
Consequently VMM must not be advertised as having the same memory geometry as
official leaf offload.

For a comparison preserving the selected BF16 math, prepare a separate BF16
checkpoint with the same values produced by the current loader's cast. Preserve
the original FP32 checkpoint. Do not switch decode to FP32, change kernel math,
or claim a quality/performance pass from this preflight. Inspect the actual
BF16 source layout before selecting a component-specific budget; do not reuse
or change the transformer's budget implicitly.

The separate saved BF16 copy was prepared on CPU at
`F:/models/qwen-image-21-vae-bf16`. All 238 saved tensors matched the original
FP32 tensors cast to BF16 exactly; source hash remained unchanged. Original
weights SHA256: `a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417`;
saved BF16 SHA256: `38fab976bd9bcd73fa0982736a2fb4f2549d1aee6a4a65b026f34fbc19d71f7b`.
The saved weights are 675,508,632 file bytes. CUDA remained uninitialized.
With local single-frame decode and residual aliasing configured, loaded-source
inspection bound all 238 descriptors, with zero rejected/missing groups, 24
execution units, zero buffers, 675,480,808 unique parameter bytes and two slots
totaling 345,122,816 payload bytes before page alignment. Native admission is
still unvalidated for this real component.

The host's VMM runner now exposes an explicit saved-source path and independent
VAE weight budget, mutually exclusive with leaf offload. It uses generic DDO
component ownership, public preflight/runtime reports and error/normal cleanup.
Load and attach are separate phases. Existing default/leaf and transformer
settings are preserved; no new DLL is required. CPU runner/conversion/block
contracts passed before proposing a real-model gate. Conversion belongs to the
host's `scripts/helpers/convert_safetensors_component_bf16.py`, not to a new
allocator transport or a model-specific DDO implementation.

## Selected real-VAE gate: functional pass, memory objective failed

User completed `outputs/qwen21_vae_native_vmm_20steps` in the examples host:
Qwen Image 2.1 base BF16, 1024x1024, 20 steps, seed 42, CFG 1, native KV cache,
untiled single-frame decode with residual aliasing, no direct padding. Native
transformer ceiling 4 GiB, staging 0.625 GiB, source cache 16 GiB, successful
Windows before-transformer standby purge and simple idle trim before VAE.
Saved BF16 VAE ceiling 0.375 GiB; same selected idle-trim DLL. No cuBLAS workspace
clear was requested. Python 3.12.11, Torch 2.12.1+cu132, Diffusers 0.40.0.dev0
from the local fork.

All 238 VAE parameters were file-backed; zero CPU source bindings, fault or
admission failures, temporary tensors, pool failures, source-read failures,
live source tickets or transfer-owned events. Completion collection retired
two remaining tickets. Decode used 12 execution-buffer uploads totaling
518,096,424 bytes. Two page-aligned slots occupied 348,127,232 bytes (332 MiB),
with zero arena-resident bytes. This exercises real execution-buffer transport,
unlike the tiny convolution gate. Normal cleanup and image save completed;
real-VAE interruption was not tested.

PNG SHA256 `1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45`
matches the accepted 20-step resident and leaf reference bytes exactly.
Decode was 1.4262 s with a 5.94677734375 GiB absolute sampled driver GPU peak,
versus simple-trim leaf's 1.8943 s and 3.92724609375 GiB. Native was faster in
this observation but used 2.01953125 GiB more peak GPU memory. Slot residency
alone does not explain that full difference; no activation/workspace or pool
causal attribution is established by these snapshots. Do not promote native
VAE as a memory-saving default. Preserve leaf as the selected capacity route.
Pipeline benchmark details belong to the examples host; no extra CUDA run or
budget change was performed while inspecting this result.

## Proposed small CUDA gate

The first approved attempt stopped during allocator installation with
`CUDA is already initialized`, before constructing or executing convolution
components. The probe imported DDO before installing the allocator; integration
imports in the selected host environment initialized CUDA. Move allocator
installation ahead of integration imports. This is a probe bootstrap correction,
not a convolution failure or a passing gate. Obtain approval for the corrected
attempt before proceeding; no real-VAE execution was performed.

The corrected, explicitly approved attempt passed on Windows using the
`ddo-examples/.venv` environment and
`build-idle-trim-cu128-20261011/Release/ddo_vmm.dll` (SHA256
`468A854EB550C6299C3DE785B65ECD4F4DF382C488B6F1C83AA6899983F12CFE`).
BF16 resident/native outputs matched exactly for both input sizes, each repeated
twice. All six parameter bindings were file-backed, with zero CPU source
bindings, fault failures, temporary tensors, cache admission denials or bypasses.
Completion collection retired six tickets with zero remaining. Intentional
decode cancellation and unload completed; all component parameters became meta
and execution hooks were removed. The DLL was not rebuilt or overwritten.

This small model fit in one native arena page and did not exercise pressure
eviction or execution-buffer substitution. It does not establish real-VAE memory
savings, performance, PNG equality, arbitrary convolution architecture support,
or cancellation inside an active convolution kernel. The optional TorchAO
import warning did not prevent this plain BF16 gate; it is not FP8 acceptance.

`ddo-vmm-allocator/scripts/vmm_convolution_component_probe.py` installs the
selected native allocator before CUDA allocation. A tiny BF16 Conv2d/GroupNorm
component executes through its `decode()` method with two input sizes and
repetition. Require exact equality against resident CUDA execution, six file
bindings, zero retained CPU source bindings, faults or temporary fallback,
completed transfer collection, and safe retirement after intentional decode
cancellation. No root-forward substitution or production VAE changes.

Obtain explicit approval before execution. Stop on the first numerical,
ownership or CUDA failure. A passing small gate is not a real-VAE memory or
quality pass. The eventual runner must keep native VAE attachment opt-in and
mutually exclusive with official VAE leaf offload, report executed controls,
and use the public DDO cleanup/report lifecycle. Preserve the accepted leaf
runner and reference PNG for comparison.
