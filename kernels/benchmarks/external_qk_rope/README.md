# Isolated external kernel benchmark adapter

This benchmark directory contains local build/registration glue only. It is not part of
DDO, its allocator, or the accepted independent execution kernels. No external
runtime behavior has been accepted yet. The pinned external source now exists locally
under ignored `vendor/`, with its original LICENSE retained.

The proposed external source is `sayakpaul/ltx2-simple-optims`, revision
`400ff39852d08df4572cea7fca4a800b0a99cc36`, declared Apache-2.0. A pinned
benchmark-only checkout must retain its original LICENSE and attribution;
record its revision in `SOURCE_REVISION`. Do not relabel external source as
independent work. The published Hub build advertises Linux/aarch64 only.

`bindings.cpp` replaces generated packaging registration, not the external
CUDA calculation. `build_windows.py` uses the original CUDA/header with
PyTorch's extension builder, NVCC/MSVC, SM 8.6, one compiler job and no fast-math
flag. Invoke from an x64 Visual Studio developer environment with the dedicated
video `.venv-vmm`. Build into a new directory, never the allocator DLL directory.
`--describe` prints the recipe without importing Torch or compiling.

Compilation loads only the extension registration and checks that CUDA was not
initialized. This does not validate kernels, stream/device ownership, numeric
accuracy, compilation integration or performance. Separate fake registrations
and a numerical/ownership CUDA gate are still required before the fourth
comparison route can run. No package dependency or runner default is changed.

The pinned loader and fake-output adapter now exist in `runtime.py`; these are
benchmark-only packaging glue, not changes to the external CUDA calculation.
The sibling host's `run_external_qk_rope_gate.ps1` prepares 12 small BF16
comparisons at widths 2048/4096, 32 heads, token pairs 17/17 and 17/5, seed 43,
using a non-default stream and the protected native allocator child bootstrap.
It compares original Diffusers RMSNorm/split rotary at `rtol=atol=1/128`, also
recording differing elements and maximum/RMS errors. This bounded operation
criterion is not final-latent parity or a benchmark. No model is loaded.

CPU-only configuration/regression contracts passed: 79 tests in 5.88 seconds,
with CUDA initialization forbidden and extension loading mocked. An initial
pytest setup failed to access the global temporary directory; the same checks
passed using a fresh host-local output directory. At this preparation checkpoint
the CUDA gate had only been previewed. Its later execution is recorded below.

Before building or running any CUDA gate, present the exact command and obtain
user approval. Do not automatically download dependencies, patch kernel math,
relax acceptance tolerances or substitute a different kernel on failure.

## Windows build checkpoint

The first build-only attempt in `build-windows-probe/` failed during CUDA
compilation: CUDA 13.2 CCCL rejects MSVC's traditional preprocessor and requires
`/Zc:preprocessor`. No kernel execution occurred and no accepted binary existed
at that first-build checkpoint.
The recipe now passes that flag to both C++ and NVCC's host compiler; original
external CUDA math remains unchanged. The failed directory is preserved.
Repeating into a fresh `build-windows-probe-v2/` requires approval. This flag
correction is not proof that the rest of compilation or linking will succeed.

The approved second build in `build-windows-probe-v2/` completed with exit 0
on October 7, 2026: NVCC 13.2, x64 MSVC 19.44, PyTorch `2.12.1+cu132`, SM 8.6.
The extension registration loaded successfully and the process verified that
CUDA was not initialized. No CUDA kernels or model benchmarks were executed.
Only local packaging/compiler flags changed; the pinned external CUDA source
was unchanged. Build manifest and full compiler log are preserved locally.

Binary: `external_qk_rope_bench.pyd`.
SHA256: `42a851774fc80f8bb83de5e802c8b9248af2b6aa79681a1b81ec44d6e62be020`.
External CUDA source SHA256:
`629a95a66757892e6103393b43628e420f201a1ab29a29fdf5653896c079b640`.
These identify build evidence, not numerical, ownership or performance acceptance.
Fake-operation registration and a separately approved small CUDA gate remain
prerequisites before the external-kernel model comparison.

## Approved small CUDA gate — numerical rejection

On October 7, 2026, the approved `E:\LTX-2\run_external_qk_rope_gate.ps1`
stopped at its first comparison, `single/2048/17,17/q`, outside the predefined
`rtol=atol=1/128` bound. Maximum absolute error 0.03125, RMS error
0.003262436715885997, 9872 differing elements. No CUDA execution error was
reported before the numerical failure. The remaining comparison outputs are
not accepted or assumed passing. No model benchmark, retry, source correction
or tolerance increase followed. The protected allocator hash stayed unchanged.

Artifact: `E:\LTX-2\outputs\external_qk_rope_gate_20261007_151437_171\report.json`.
Selected external binary hash remains the build checkpoint above.

Source inspection found a concrete arithmetic mismatch: local Diffusers
RMSNorm casts the normalized value to BF16 before multiplying the BF16 weight;
the external kernel multiplies normalization and weight in FP32 and then casts
to BF16. A CPU-only cast-order illustration using the same reciprocal for both
orders reproduced differences (9387 elements, max 0.0625, RMS
0.002789747668430209); the original-order expression matched the CPU norm
exactly. This illustration is not a replay of the CUDA input, reduction or
rotary arithmetic, and does not establish the only cause of divergence.

The external-with-kernel comparison route is therefore numerically unaccepted
under the selected gate, despite successful Windows compilation. Report this
status explicitly instead of inventing a performance number. Do not silently
patch third-party kernel arithmetic and label it an unchanged author benchmark.

## Q/K oracle correction preparation (not CUDA acceptance)

Subsequent read-only review found that the installed LTX attention Q/K uses
`torch.nn.RMSNorm`, whereas the historical small gate used Diffusers RMSNorm.
That failure remains preserved against its declared oracle; it does not prove
failure against the actual Q/K normalization. No claim is made that replacing
the oracle resolves reduction or rotary differences.

The host now provides an explicit, separately named candidate:
`E:\LTX-2\run_external_qk_rope_gate.ps1 -TorchRmsNorm`.
It calls the unmodified original PyTorch norm, retaining all 12 cases, seed,
non-default stream and `rtol=atol=1/128`. Both binary hashes remain pinned;
reports use a fresh `external_qk_rope_torch_rms_gate_*` directory. The historical
Diffusers oracle remains the default for reproducibility. Seven CPU-only
adapter/oracle tests passed in 3.86 seconds with CUDA initialization forbidden.
The command was previewed only; separate user approval is required to execute
CUDA. Passing this small gate would not authorize a model benchmark or establish
final-latent parity. Production runtime and defaults remain unchanged.

## Approved corrected PyTorch Q/K gate

The separately approved `-TorchRmsNorm` gate completed with exit 0 on October 7,
2026. All 12 single/paired comparisons at widths 2048/4096 and token pairs
17/17, 17/5 were finite and bit-exact: zero differing elements, maximum and RMS
errors zero. Original tolerances were not relaxed. No CUDA execution error was
reported; both selected allocator and external-kernel hashes were unchanged.
Artifact: `E:\LTX-2\outputs\external_qk_rope_torch_rms_gate_20261007_204831_328\report.json`.

The historical Diffusers-oracle rejection remains preserved. It must not be
generalized to the actual PyTorch Q/K operation: the corrected selected cases
passed without changing third-party CUDA math. This does not establish model
parity, long-sequence performance or native weight-ownership correctness for a
model integration. No model run, default promotion or automatic retry followed.

## Optional host model comparison preparation

The sibling host now has `external_qk_rope_execution.py`, opt-in through
`--external-qk-rope-library`. It uses the local Diffusers fork's backend-neutral
pre-normalization callback; this adapter is not installed or imported by DDO.
Live Q/K norm weights are explicit opaque-operation inputs. Same-table attention
uses the dual op and cross attention keeps separate tables/single ops. No native
residency, kernel math, attention backend or default changes were introduced.

CPU integration/export and existing regression contracts passed (112 tests,
six subtests, 15.94 seconds) without CUDA initialization. The prepared host
`run_corrected_modulation_ab.ps1 -ExternalQkRopeAb` compares compiled execution
without/with this external kernel, with unchanged selected two-step workload
and safety/final-latent parity gates. Only command preview has run; full-model
CUDA execution, performance and ownership remain unvalidated. User approval
is required before that A/B. Installed opaque norm counts are not invocation
counts; no Python submission counters run inside the compiled callback.

## Approved host A/B — model parity rejection

The subsequent approved two-step compiled-control versus compiled-plus-external
Q/K sequence completed both model runs, but exited 1 at final-latent parity.
Artifact: `E:\LTX-2\outputs\regional_compile_external_qk_ab_20261007_211322_386`.
Selected native safety, source collection and callback retirement passed; both
pinned binary hashes stayed unchanged. Final video/audio were finite but not
bit-exact (relative RMS 3.0383% / 2.4354%). The earlier 12-case small-operation
acceptance remains valid only for its selected synthetic inputs, not the model.

Second CUDA intervals were 39.6475 s control and 39.4613 s external (about 0.47%
difference); total runner 123.6231 s versus 130.7741 s. One ordered pair does not
establish a reliable performance benefit, and first intervals include lazy
compilation. No production/default promotion, retry or relaxed numerical gate.
This does not invalidate the author's separate benchmark or identify the sole
operation causing the final divergence. Full details remain in the sibling
host's `model_execution_optimizations.md` and `execution_comparison_summary.md`.

Keep the pinned external checkout under the ignored `vendor/` directory and
build evidence under ignored `build-windows-*/` directories. The recipe is not
installed by DDO's `src`-only package discovery. LTX runners, model-specific
integration and performance artifacts remain in the sibling host repository.
