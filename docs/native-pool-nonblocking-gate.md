# Native pool opportunistic retirement candidate

Status (2026-10-05): compiled, CPU-validated, six small CUDA contracts passed,
and the approved matched video case completed with an observed improvement.
This candidate is not the selected Windows binary and does not change host defaults.

## Change and ownership

The native span pool previously waited for idle backing's consumer events while
trimming its cache from `free`. The candidate queries completion and only selects
fully releasable backing for opportunistic trimming. Pending backing remains
mapped, accounted, and reusable through the existing stream dependencies.
Allocation-pressure eviction retains its completion waits and checked release.
Quarantine, admission guards, cache limits, and weight/staging budgets are unchanged.
The cache threshold is not a hard bound on outstanding consumer-owned bytes;
it never authorizes increasing the physical allocation admission limits.

No forward implementation, model-specific path, linear preset or protected NVMe
runtime was changed. This is an independent native ownership-policy correction.
Driver unmapping may itself incur overhead; absence of explicit event waits in
opportunistic retirement is not a promise that all driver calls are nonblocking.

## Build and CPU evidence

Built with MSVC 19.44, CUDA toolkit 12.8.93, Release, native span pool enabled,
source dispatcher and lifecycle fault injection disabled.

- Candidate: `ddo-vmm-allocator/build-pool-nonblocking-20261005/Release/ddo_vmm.dll`
- SHA256: `E478BB872F593C0BCD272DFDBC9389DC430EA04951D4B95F2FF3197F08B218DE`
- Accepted binary remains in `build-lifecycle-off-20261002/Release/` with SHA256
  `755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729`.
- All seven isolated CPU CTest contracts passed, including pending/partially
  completed backing retirement checks added to the span ledger contract.
- `tests/test_dynamic_offload_presets.py`: 32 tests and two subtests passed.
  Pytest reported an inaccessible cache-directory warning; tests completed.

CPU evidence does not establish CUDA safety, baseline preservation or speedup.
The prior video's 59.075447 seconds of pool eviction wait is cumulative through
the snapshot, not an isolated denoise measurement or guaranteed removable time.

## Approved sequence and commands

Run from the DDO repository using its `.venv-vmm`. Scope the candidate DLL path
to the validation process and restore previous environment values afterward.

```powershell
$env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-pool-nonblocking-20261005\Release\ddo_vmm.dll'
.\.venv-vmm\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_torch_cache_stream_cuda.py -q
```

Six small cases cover same-stream, cross-stream and legacy-stream reuse with
4 MiB and zero cache thresholds. Hypothesis: an in-flight writer survives `free`
without an explicit completion wait, and its successor observes the required
stream ordering. Require six passes, actual pointer reuse, correct payload and
no CUDA errors. A completion that already finished at the observation boundary
fails this diagnostic; investigate before attributing it to synchronization.
Stop at the first failure or memory anomaly.

Only after this gate passes, run one matched no-input-audio video case from
`E:\LTX-2`, with scoped `LTX_USE_AUDIO_CONDITION=0`, `CUDA_LAUNCH_BLOCKING`
unset, and previous environment restored afterward:

```powershell
.\run_ddo_vmm_video.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-pool-nonblocking-20261005\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 5 --steps 8 --ddo-vmm-resident-weight-budget-gb 2 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024
```

Keep the prior BF16, 121-frame, 24 fps, seed 43 conditions and existing 1.5 GiB
staging / 24.625 GiB source-cache and registration settings. Hypothesis: deferred
opportunistic retirement reduces host waiting without destabilizing allocation.
Require completed generation, cleanup and save; zero collected quarantine,
pool failures and temporary fallbacks; no OOM or memory anomaly. Compare progress,
generation and total times against 27.86 s/step, 223.6 s and 297.2001 s respectively,
using matching measurement scopes. Also compare mapped/pending bytes, eviction
wait, materializations and uploads. A single faster run is observational, not a
repeat-validated speedup. Do not promote on a slower or unsafe result, and do not
automatically adjust budgets or extend the test sequence.

## Executed evidence

The six CUDA cases passed in 2.34 seconds using DDO's `.venv-vmm` and the
candidate hash above. The conditional video case then exited successfully,
with the console confirming `use_audio_condition=False`. Scoped environment
values were restored. The accepted DLL and runner defaults remain unchanged.

| Measurement | Accepted-binary control | Candidate |
| --- | ---: | ---: |
| Progress average, s/step | 27.86 | 26.09 |
| Generation phase, console seconds | 223.6 | 209.4 |
| Full run, JSON seconds | 297.2001 | 282.7551 |
| Pool eviction wait, cumulative seconds | 59.075447 | 0.0001051 |
| Pool eviction attempts | 499 | 526 |
| Native batch submit, cumulative seconds | 27.2338389 | 49.0033472 |
| Pool mapped bytes at snapshot | 2248146944 | 2017460224 |
| Pool pending bytes at snapshot | 1099190272 | 1909000704 |

Progress improved by approximately 6.4%, generation by 14.2 seconds and total
by 14.4450 seconds (4.9%). These are single-run observations, not repeat-validated
causal estimates. Removing opportunistic waits changed scheduling: native batch
submission now records more host time. Submit/enqueue counters include waits and
overlap other work; they are not GPU DMA elapsed time and must not be added to
pool waits or subtracted from total runtime as independent costs.

Both snapshots recorded 192 materializations, 512 execution-buffer uploads,
zero execution-plan mismatches, pool failures, quarantine bytes, source-read
failures and temporary tensors. Source tickets were collected from 2 to 0.
The increased pending-byte snapshot preserves unobserved completion ownership;
it does not by itself establish still-running GPU work or a leak.

Candidate driver usage after pipeline return was 4.95 GiB absolute. Tracker
observations reached 29.0081 GiB RSS, versus 28.7480 GiB in the control.
Generation monitoring stayed disabled, so neither recorded driver deltas nor
RSS observations establish continuous denoise peaks. No OOM or CUDA error was
reported. Cleanup, transformer unload, video/audio decode and save completed.

Artifacts: sibling host `outputs/ddo_native_video_20261005_203826_052/`, with
the MP4 and JSON under `metrics/`. CPU media inspection decoded all 121 frames
at 1280x704, confirmed one audio stream and a nonconstant first frame. Visual
quality, audio quality and full GPU numerical equivalence were not assessed.

The candidate passes this selected video gate, but has not been promoted to the
runner default. The accepted image route was not rerun, and these results do
not certify arbitrary models, shapes or platforms. No additional CUDA run or
budget adjustment was performed beyond the approved sequence.
