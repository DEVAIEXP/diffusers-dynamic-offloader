# Validation

## RC5 status

RC5 has selected source, installed-candidate and model validation. The final
consolidated wheel pair still needs assembly and validation of its exact
packaged bytes. Candidate suffixes below identify the artifacts tested; they
do not certify a later rebuild or a package with a different payload.

| Area | Recorded RC5 validation | Remaining limit |
| --- | --- | --- |
| Model integration | Selected LTX Image, LTX Video and Qwen runtime, attention, precision and feature comparisons on Windows, native Linux and WSL | Coverage differs by model, route and platform; use the host's measured conditions |
| Windows native allocation errors | Installed checked bridge and model-weight OOM handling under external GPU-memory pressure | Only the matching Windows Torch/CPython build has checked-bridge acceptance |
| Recovery notices | Selected Windows runs verified opt-in summaries and final error reporting | A recovery message alone is not proof of completed inference |
| Saved linear budgets | CPU contracts and selected Windows video runs apply saved budgets over inherited preset values | Latest correction has source validation; final wheel validation remains separate |
| FP8 component capacity | Selected Linux and Windows installed-candidate calibration and video runs | WSL FP8 is not covered by these checks |

Full-model results and reproduction belong to
[DDO examples benchmarks](https://devaiexp.github.io/ddo-examples/docs/benchmarks/).
This page records library and package validation, rather than duplicating
pipeline performance tables. See [test hardware and environments](hardware.md).

## Validation layers

| Layer | What it establishes |
| --- | --- |
| CPU contracts | Routing, configuration, reports, discovery and mocked ownership |
| Native CPU contracts | File/storage behavior and Linux admission observations |
| Small CUDA contracts | Payload correctness, registration reuse, event ownership, close refusal, interruption and idle trim |
| Model validation | Correct output and public reports for a fixed model/workload |
| Installed wheel | Discovery, linkage and separately tested native behavior of the actual packaged bytes |

Compilation and source-checkout tests do not establish installed-wheel acceptance.
A successful model does not certify all routes, quantizers or platforms.

## Installed-package check without CUDA

From the checkout, run `scripts/probe_installed_native_cpu.py` with the installed
environment's Python and `-I -B`. It checks the native library hash and automatic
discovery, then simulates the child-process bootstrap without loading the DLL/.so
or running CUDA. This is a maintainer release check, separate from GPU tests and
complete model runs.

## Windows native allocation errors and recovery notices

The selected installed environment used DDO `0.1.0rc5+recovery.win2` and
allocator `0.1.0rc5+bridge.win2`, with CPython 3.12, Torch 2.12.1+cu132 and
NVIDIA driver 617.42. These are local candidate identities, not final-release
wheel acceptance.

| Native payload | SHA256 |
| --- | --- |
| Windows allocator DLL | `7d4d7e55d8245d5f1af94af5a3e18af389ca9a69aa9f993ba5fdf421a4b5a0d3` |
| Torch-linked checked bridge | `35f3cec2c3f18346e134c54361aeb787418dbfbe631244142539dbab0d8402c9` |

CPU selection and installed-discovery checks passed. A small CUDA gate checked
the installed boundary and exact payloads. Selected eight-second LTX Video
execution completed without external pressure. Under external pressure,
activation-allocation refusal raised `torch.OutOfMemoryError` instead of
publishing invalid tensor storage; cleanup completed without a subsequent
illegal-address error. A separate selected model-weight execution-slot refusal
raised `NativeVmmOutOfMemoryError`, also a `torch.OutOfMemoryError`, with cleanup
completed. Eleven focused CPU weight-error tests passed.

The bridge is selected only when the recorded Torch version/build, CPython,
platform and module hash match. Linux, WSL and other Torch builds retain the
plain registration path unless a separately validated matching bridge is
provided. These checks do not establish recovery from an already corrupted CUDA
context or safe behavior for every allocation failure.

Successful-recovery summaries are quiet by default. Selected Windows checks
verified `DDO_SHOW_RECOVERY_WARNS=1` enables them; final errors remain visible.
Windows standby purge is also opt-in. The selected LTX Image LoRA check
completed with the updated package. See
[troubleshooting](../user/troubleshooting.md) for error messages and boundary
inspection.

## Selected linear-profile budget correction

Saved linear profiles replace inherited preset resident-module budgets, including
a recommendation of zero. Explicit positive caller budgets retain precedence;
without a saved recommendation the preset default remains in effect.

Twelve named-profile CPU tests passed. Selected Windows LTX Video executions
with working-tree DDO applied their existing zero-budget profiles without the
host's budget override and produced byte-identical videos. One Shot Fast retained
effective CPU pinning and matched its historical performance (333.07 s denoise
versus 335.02 s). Two Low RAM Safe runs completed correctly but took 499.58–535.41 s
versus 419.30 s historically; their performance variation remains unexplained.
This is selected functional acceptance, not a general performance guarantee or
installed-wheel acceptance. Native binaries and published benchmark rows were
unchanged.

## Selected component-profile correction

The Linux DDO candidate `0.1.0rc5+fp8floor.2` has selected installed-wheel CPU
and LTX 2.3 saved FP8 pipeline acceptance. The candidate changes two Python
modules to derive the structural minimum from component execution geometry
and preserve persistent root capacity in the final profile recommendation.
Its 58 CPU contracts include unchanged Linear BF16 sizing, explicit-budget
preservation, component sizing and large-page root reservations.

One approved calibration and an eight-step, eight-second FP8 measurement
completed using the installed candidate. The final profile was applied with
matching shapes; native final-failure counters and quarantined pool bytes were
zero. Source/dispatcher and media checks passed. Visual quality is unconfirmed.
Pipeline timings and reproduction belong to the
[host benchmark](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmarks/ltx23-video-details.md#fp8-linux).

The allocator remains RC5 with native SHA256
`8d69af5056ee8c28da6c6e0b0adf1e7e5e23c67b1d66b641c01cfc949d6af459`.
No native arena, backing segmentation, eviction or fence code changed. This
selected Linux result alone does not establish general quantization support or
Windows/WSL acceptance of the correction. The Linux candidate is not a final or
published RC5 distribution; final assembly and other pending fixes require
their own review.

The correction also completed a selected installed-package Windows LTX 2.3
FP8 workload: one three-step calibration finalized at 806 MiB and the
eight-step, eight-second measurement applied its matching profile. Source
dispatch was active, reported native final failures and quarantined bytes were
zero, media checks passed and the user accepted visual quality. The tested
Python candidate was `0.1.0rc5+fp8floor.win1`; it retained the previously
installed Windows library-discovery correction. The native DLL remained
`60f73271cd1552769daf1e02de50702b46ab17a4b99e239b804a26c129ebf815`.
See the [host measurement](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmarks/ltx23-video-details.md#fp8-windows).

A local RC5 wheel consolidates those exact 24 tested Python files, changing
distribution metadata from the candidate version to `0.1.0rc5`. Its installed
bytes and RECORD were verified, and 14 floor/residency CPU contracts passed
with CUDA initialization blocked. The selected model run belongs to the
candidate with identical Python payload, not a second GPU run of the renamed
wheel. WSL FP8 and unrelated pending source/native changes are not covered.
At that stage, an additional linear-profile contract failed against both the
candidate and original Windows RC5 package. The later
[linear-budget correction](#selected-linear-profile-budget-correction) has its own
selected validation; it does not imply that the full test suite is green.
Final distribution follows the separate [package delivery](releases.md#validate-and-publish) process.

## Earlier validation

RC2/RC3 artifact identities and selected checks are retained in
[Validation history](validation-history.md). They are not RC5 test results.

## Source-level coverage

Public tests cover policy/precedence, loading/source association, reports,
dense convolution ownership, source-cache/event lifetimes and bounded cleanup.
Saved TorchAO FP8 has separate payload/root-state/interruption contracts;
BF16 success does not imply FP8 success. Refer to
[FP8 support](../user/fp8-weight-only.md) for the supported recipe.

Native Linux admission must preserve overcommit mode, physical/process bounds
and finite cgroup limits. Unknown observations refuse growth. WSL additionally
requires bounded host-registration validation; an observed capacity is not a
universal preset.

Full model comparisons and their exact reproduction conditions are maintained
in [DDO examples](https://github.com/DEVAIEXP/ddo-examples), not duplicated here.

### Residual CPU source ownership

The explicit `detach_auxiliary_cpu_sources` contract has CPU coverage for byte
preservation, aliases, source provenance, pre-copy budget refusal, transactional
rollback and protection of retained native CPU payloads. A mapped-storage
regression checks that tiny registered residual aliases release their large
original storage owner after detachment, without treating storage extent as a
measurement of recovered RAM or private commit.

Selected Windows two-step model validation also checked complete finite,
bitwise-identical video/audio latents and safe closure after bounded detachment.
Its commit observations and limits are recorded in the
[model measurement details](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmarks/ltx23-video-details.md#residual-cpu-source-detachment).
This does not establish full inference/decode, other platforms/models, wheel
acceptance, automatic enablement or a speed improvement.

## Acceptance rules

Use [manual validation](manual-validation.md) to record routes, versions, native
identity, output and measurement scope. Missing fields are unavailable, not zero.
Host callbacks, CUDA events and whole-pipeline wall time remain distinct.

Successful inference does not prove recovery from a poisoned CUDA context,
all-platform support, long-run stability or final release of driver memory.
Retain failed-run diagnostics separately from accepted performance results.
