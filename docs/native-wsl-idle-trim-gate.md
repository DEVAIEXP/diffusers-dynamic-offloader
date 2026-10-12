# WSL isolated idle-trim gate (2026-10-12)

Status: user-executed small CUDA pool ownership probe PASS. This is not
persistent-source registration, model generation, performance or general WSL
support acceptance. Existing Windows and native Linux binaries are preserved.

## Environment and build

- WSL GPU-enabled Ubuntu 22.04.3 session, user `master`, `/dev/dxg` present.
- RTX 3060 Ti, 8192 MiB; NVIDIA-SMI 615.78.05, KMD 617.42, CUDA UMD 13.4.
- Example host environment `.venv-wsl-gpu`, Python 3.12.12, uv 0.9.11.
- Torch 2.12.1+cu132, TorchAudio 2.11.0+cpu, TorchCodec 0.15.0+cu132,
  official Diffusers 0.41.0, editable sibling DDO/allocator.
- Imports succeeded without CUDA initialization. The selected eight-file CPU
  gate passed 53 tests in 44.68 seconds in this session. The earlier 53-test
  result belongs to a different Ubuntu session and is not this WSL gate.
- GCC 11.4.0, environment-local CMake 4.4.4, CUDA toolkit 13.2.78.
  Toolkit components came from `cuda-toolkit[nvcc,cccl]==13.2.1`; no Linux
  NVIDIA driver was installed. FFmpeg shared libraries were installed separately
  in this distribution before the successful TorchCodec import.

The preserved Linux binary could not load here: required GLIBC_2.38,
GLIBCXX_3.4.31 and CXXABI_1.3.15 were unavailable. No system runtime replacement
was attempted. A separate candidate was built from the allocator checkout:

- Path: `ddo-vmm-allocator/build-wsl-idle-trim-20261012/libddo_vmm.so`.
- SHA256: `ffc33a4887e51c89aa6e3a01b7660ee8e4f5d8742bb79d990ca749d97494506d`.
- Release; span pool ON; joined spans, source dispatcher, source test hooks,
  allocation diagnostics, probes and execution ops OFF.
- Driver link selected `/usr/lib/wsl/lib/libcuda.so`.
- Configuration/build completed; unused alternative allocator helper warnings
  remained. `ldd` reported no missing dependencies or ABI version errors.

## User-executed bounded CUDA probe

From the examples host root:

```bash
.venv-wsl-gpu/bin/python -B \
  ../diffusers-dynamic-offloader/ddo-vmm-allocator/scripts/vmm_idle_trim_probe.py \
  --library ../diffusers-dynamic-offloader/ddo-vmm-allocator/build-wsl-idle-trim-20261012/libddo_vmm.so
```

The reviewed probe uses a 32 MiB temporary allocation, a 1024-byte active payload,
and a 512 MiB cache setting. These are not a guaranteed total driver-memory peak.
It installs the native allocator in a fresh process before CUDA initialization.

| Observation | Before trim | After trim | Final |
| --- | ---: | ---: | ---: |
| Mapped bytes | 35,651,584 | 2,097,152 | 0 |
| Requested live bytes | 1,024 | 1,024 | 0 |
| Pending bytes | 33,554,432 | 0 | 0 |
| Idle backing bytes | 33,554,432 | 0 | 0 |
| Live allocations | 1 | 1 | 0 |
| Quarantined bytes | 0 | 0 | 0 |
| Failures | 0 | 0 | 0 |

Returned `status: PASS`; exact active payload, repeated trim and allocation
after trim checks succeeded. Cumulative eviction wait was 0.0002522 seconds
after the first trim and 0.0002684 seconds at final cleanup; these are native
wait counters, not inference timings. No model or persistent source cache ran.

Next inspect WSL host/cgroup capacity and propose a separately approved native
source/capability gate before model execution. Do not inherit native Linux
model acceptance, increase ceilings automatically, promote this binary, or
repeat the closed Windows/Linux benchmark matrix.

## Subsequent source gate and corrected candidate

The WSL preflight observed MemAvailable 40,362,832 kB, CommitLimit 54,091,820 kB,
Committed_AS 368,236 kB, overcommit mode 0 and cgroup membership `0::/`.
The readable cgroup2 mount was `/sys/fs/cgroup`, with the memory controller
present and no root memory.max/current files. Root absence is expected and
does not authorize ignoring finite host/process admission.

The first source-cache contract stopped before CUDA host registration with
`source cache host/cgroup headroom observation unavailable` (one failure in
11.63 seconds). No later contract or model ran in that sequence. Investigation
found an empty relative-path join retaining a trailing separator, making the
mount root compare unequal to its canonical base. The native fix preserves
the base for empty relative paths, including exact subtree mount-root matches;
unknown hierarchy, malformed limits, escape and memory admission guards remain.

Two root-identity regression cases were added to `native/tests/linux_host_test.cpp`.
The user compiled and ran that CPU-only executable successfully, including
existing ancestor/strict-mode/escape contracts and actual WSL host observation.
The candidate was rebuilt in the same isolated WSL directory:

- Corrected SHA256:
  `5915cca767c89ab7606c04f2001c9eddffa762386c852cc9f517708a2d03584c`.
- The earlier idle-trim probe remains evidence for the original hash above,
  not automatically for this rebuilt binary. Accepted Linux/Windows artifacts
  were not overwritten.
- The same three user-executed CUDA source-cache contracts subsequently passed:
  **3 passed in 11.62 seconds**, without skips or parameter changes.

Selected contracts in `tests/test_source_cache.py`:

1. `test_persistent_storage_survives_registration_recycling_without_reread`
2. `test_immutable_batch_retains_sources_recycles_events_and_guards_close`
3. `test_live_tickets_protect_cache_close_and_clear`

These exercise small retained segments, registration recycling, asynchronous
copies with exact payload assertions, immutable batch lifetime/event reuse,
and refusal to close active cache resources. Small registration success does
not prove full-model pinning capacity or performance. The model gate remains
pending at that checkpoint and required a reviewed workload, actual
cache/custom-block paths, fixed ceilings and separate user confirmation.

## Subsequent registration capacity and staged workload acceptance

All observations below use the corrected candidate hash above. The initial
staged workload failed during native text-encoder host-ring construction with
`cuMemHostRegister failed: CUDA_ERROR_OUT_OF_MEMORY`, before encoder forward.
The requested staging ring was 24 windows of 64 MiB (1.5 GiB total).

Standalone normal-PyTorch-allocator controls passed 128 MiB, six 64 MiB windows,
512 MiB both as one block and eight windows, and sixteen 64 MiB windows.
Registration above the observed 64 MiB RLIMIT_MEMLOCK succeeded; that limit
alone does not explain the failure. No OS limit or driver setting was changed.

The user then executed the bounded automated diagnostic documented in
[Host registration capacity diagnostic](host-registration-probe.md), with a
1536 MiB ceiling and 64 MiB windows. The JSON at
`../ddo-examples/outputs/wsl_host_registration_probe.json` records:

- 1152 MiB passed in two fresh processes.
- 1216, 1344 and 1536 MiB returned host-registration CUDA OOM.
- No timeout or unexpected error occurred.
- This is an observed standalone boundary at 64 MiB resolution, not a universal
  WSL limit, guaranteed available capacity under model load, or a DMA gate.

Two separately approved staged LTX image workloads subsequently completed eight
steps and saved a PNG, using real prompt encoding and native encoder/transformer
routes. Both retained the 4 GiB accelerator weight ceiling and 24.625 GiB source
cache capacity, with a 384 MiB staging ring. Only the source registration ceiling
changed: 128 MiB for the first case, 512 MiB for the second. Evidence directories:

- `../ddo-examples/outputs/ltx_image_modular_vmm/wsl_bounded_registration_20261012`
- `../ddo-examples/outputs/ltx_image_modular_vmm/wsl_registered512mib_20261012`

Both transformer reports recorded zero fault failures, temporary fallback
tensors, pool failures, quarantined bytes, source admission denials, ring fallback
bytes and host copy bytes. Source-cache registration peaks respected their
configured ceilings. The second workload reduced registration churn, while
source read bytes and hit/miss counts remained unchanged. Churn remains a known
cost; a small registration ceiling does not reproduce fully registered-cache
performance. Pipeline timings and hardware recommendations belong to the
examples-host benchmark documentation, not to this technical gate.

This accepts the selected workload on the tested WSL environment, not arbitrary
models or general WSL support. Saved reports do not prove every final resource
counter returns to zero after process exit; the original idle-trim probe remains
tied to its own binary. Windows/native-Linux defaults and accepted binaries were
preserved. No automatic runtime tuning or ceiling increase was introduced.

## Finite WSL sequence closure

The user subsequently completed the approved five-second BF16 LTX video with
image/audio conditioning, eight steps, 121 frames at 24 fps and 1280x704.
Native encoder/transformer ceilings were 4/1.5 GiB respectively. GPU streaming
video decode and staged audio decode completed, with finite stereo 48 kHz output
and a saved MP4. Encoder/transformer snapshots had zero selected native failures,
temporary tensors, quarantine, live source tickets and transfer-owned events.

The approved Qwen Image 2.1 base BF16 workload also completed twenty steps at
512x512, seed 42, saving a PNG. The transformer used native VMM with a 4 GiB
weight ceiling and 16 GiB retained source capacity. VAE decode executed official
no-stream leaf, without tiling or local VAE compute experiments. Its transformer
snapshot had the same zero selected failure/ownership counters.

Both workloads kept 384 MiB staging and a separate 512 MiB source-registration
ceiling. Artifacts are in the sibling examples host:

- `outputs/ltx_video_wsl_audio_5s_20261012/metrics/modular_run.json`
- `outputs/qwen21_wsl_vmm_leaf_512_20261012/metrics.json`

This closes the requested finite WSL sequence. Do not automatically repeat
benchmarks or extend to SDNQ/FP8, larger models or alternate presets. WSL
acceptance is selected-workload compatibility, not parity with Windows/native
Linux or a matched performance comparison. Pipeline results and the developer
installation recipe belong to `ddo-examples`; packaging automation and reviewed
local checkpoint commits remain delivery work.
