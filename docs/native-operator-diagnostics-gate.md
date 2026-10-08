# Sampled forward operator diagnostic (2026-10-06)

## Evidence and scope

The approved allocation/retry diagnostics did not explain the slow initial
video step: the latest run measured zero model-arena retry synchronization and
0.0981724 seconds in enclosing faults. Execution windows dominate but include
host launch gaps and GPU dependencies. See
[recorded diagnostic evidence](native-allocation-diagnostics-gate.md).

Prepare one operator-level diagnosis, not a performance fix or budget search.
DDO owns a generic selected-module profiler and public report. The video host
opts in via `--diagnose-operators` with `--diagnose-denoise`; its selected path
defaults to `transformer_blocks.0`, invocations 0 and 2. No Diffusers source or
native code changed in this block. The ordinary host branch remains disabled.
Capture only two forwards rather than an entire denoise operator trace.

Profiler start/stop may synchronize and alter submission cadence. No tensor
shape/stack/memory tracing. Direct-child labels distinguish attention,
normalization and feed-forward modules; operator and runtime events allow
host-versus-device attribution when CUDA tracing is actually available. Host
wall time and profiler-stop wait are recorded separately. This is not a
benchmark, proof of WDDM paging, or justification to change attention backends.
Zero/missing CUDA activity refuses capture; no silent legacy-event fallback.

## Completed CPU preparation

- 47 selected DDO CPU contracts passed, including real CPU profiler/export,
  unchanged outputs/forwards, hook removal, prior-hook preservation, exception
  handling, bounded selection, ownership, overwrite refusal and unavailable
  CUDA tracing rejection without CUDA initialization.
- 34 host modular CPU contracts passed, including default-off/native diagnostic
  prerequisites and minimum step count. Expected negative-fixture errors remain
  intentional test output.
- No CUDA run followed these changes. An initial test invocation lacked the
  submodule source path; another encountered denied default pytest temporary
  storage. Final validation used the explicit submodule PYTHONPATH and a fresh
  ignored `.internal/` temporary directory; no user directory was deleted.

Preserved DLL SHA256:
`335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.
Preserved Diffusers transformer forward SHA256:
`F37835B2935C01ED1DC23C67BAF702A9B6CB0D399611512A2E627FC23A8BAC71`.
No promotion or image GPU non-regression claim.

## Proposed single execution, pending user confirmation

```powershell
Set-Location E:\LTX-2
.\run_ddo_vmm_video_modular.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-allocation-retry-diagnostics-20261006\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 8 --steps 4 --seed 43 --no-use-audio-condition --output-type latent --diagnose-denoise --diagnose-operators --diagnose-operator-module transformer_blocks.0 --ddo-vmm-resident-weight-budget-gb 2 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources
```

Hypothesis: selected first/third forwards expose dominant attention/linear/
normalization/chunked operators, blocking CUDA launches or device-activity gaps,
which the broad native execution windows cannot distinguish. The first block
alone cannot rule out costs elsewhere. Unsupported tracing is a failed
diagnostic gate, not permission for an automatic alternative or full repetition.

Criteria: exactly two requested captures with actual CUDA activity and readable
traces; public operator status complete; four CPU/GPU phase boundaries and native
timeline complete/no drops; packed video/audio finite; original native safety
counters zero and tickets retired. Stop on profiling failure, OOM or memory
anomaly. No policy, budget, chunk-size, backend, decode or step-count change.
Do not interpret profiler-perturbed timings as an improvement over prior runs.

## Interrupted dispatch (2026-10-06)

The user approved the proposed command. Its execution tool returned `aborted`
without a session identifier, output or process exit code. A new directory
`E:/LTX-2/outputs/ddo_modular_video_20261006_150618_511/` was observed, but
recursive inspection found no files; the following process snapshot showed
no active high-memory inference Python process. A CIM command-line query was
denied. No trace, metrics, latents, completion or CUDA safety result can be
claimed, and the reason for interruption is unknown. The wrapper does not
persist stdout by default. Preserve this directory; do not treat the interrupted
dispatch as a diagnostic PASS or a profiler failure. No automatic repeat was
started. Ask for confirmation to relaunch the same command with unchanged scope.

## Approved relaunch result (2026-10-06)

User explicitly requested a relaunch of the same command. Added only console
capture via PowerShell `Tee-Object`, without changing workload arguments.
Artifact: `E:/LTX-2/outputs/ddo_modular_video_20261006_150751_571/`.
Console: `E:/LTX-2/outputs/operator_diagnostic_console_20261006_150751_508.log`.
Exit 0, completed; total elapsed 579.3895 s, compute-stream phase 539.3473 s.
Step intervals: 160.6900/233.8860/97.3991/47.2788 s. Profiling perturbs cadence;
these times are not matched performance evidence or a demonstrated regression.

Public operator report complete, exactly two requested captures (invocations
0 and 2); each reports 1019 device activity events. Native timeline complete,
1187 samples, no drops/failures/pending and events released. Original encoder/
transformer failure, denial, fallback, mismatch, quarantine, failed observation/
reclaim and cache-pressure-entry counters zero. Tickets encoder 7 to 0,
transformer 0 to 0. Allocation diagnostics separately recorded one recovered
access OUT_OF_MEMORY return; this is not an unrecovered pipeline OOM and must
not be hidden behind the zero final safety counters. Model-arena retry sync
remained zero; 98 faults totaled 0.1068239 host seconds.

CPU artifact inspection with CUDA initialization forbidden confirmed finite
float32 packed video `[1, 22000, 128]` and audio `[1, 201, 128]`. Peak RSS
29.6500 GiB, GPU peak delta 6.9727 GiB over baseline 1.0268555 GiB. Selected
diagnostic collection/safety criteria PASS; no image GPU non-regression claim.

### Selected-forward observations

| Observation | Invocation 0 | Invocation 2 |
| --- | ---: | ---: |
| Host forward seconds | 1.751903 | 2.661906 |
| Profiler-stop host seconds | 0.577725 | 0.736242 |
| Raw kernel events / summed durations | 897 / 2.081667 s | 897 / 2.261280 s |
| Raw stream-sync calls / summed host durations | 6 / 1.195807 s | 6 / 2.499648 s |
| Raw pageable H2D copies / summed device durations | 6 / 0.000014 s | 6 / 0.000016 s |
| Raw pinned H2D copies / summed device durations | 79 / 0.089128 s | 79 / 0.057216 s |

Raw trace sums can overlap and are not independent phase partitions. Public
aggregate durations differ slightly from raw events; do not silently substitute
one measurement for another. Device activity counts also include non-kernel
activities. Top device aggregates mix operator/range/kernel levels and must
not be added together. BF16 GEMM and the current SDPA memory-efficient kernel
appear; no attention-backend correction follows from this observation.

The expensive stream synchronizations are nested in `aten::to` / `_to_copy` /
`copy_`: invocation 0 has 0.817959 and 0.377722 s waits; invocation 2 has
1.027477, 1.187102 and 0.284892 s waits. The device-wide synchronizations
(0.539879/0.690522 s) are outside operator/child annotations, consistent with
profiler finalization; do not treat them as a proven normal-forward defect.

Six pageable copy sizes are 73728, 36864, 32768, 8192, 16384 and 4096 bytes.
They match the forward's four local modulation tables (two cross-modal tables
are sliced into scale/shift and gate copies): 172032 bytes per block,
8257536 bytes across 48 blocks. That matches the observed residual CPU storage.
`get_mod_params()` moves these slices with `.to(temb.device)`; DDO's existing
small-local placement uses a 16 KiB threshold, below each complete table's
size. CPU-source detachment clones their CPU storage but explicitly does not
change GPU placement. This identifies a concrete uncovered auxiliary route,
not a proof that its waits explain all long-step cost or would vanish as savings:
they can drain already queued GPU work.

Next correction proposal: cover residual auxiliary parameters within native
execution-unit source/view ownership, model-agnostically and charged to the
unchanged native weight ceiling. Preserve source aliases, dtype, event lifetime
and rollback/cleanup. Do not merely raise the shared small-tensor threshold,
add an unaccounted Python residency cache, remove synchronization globally or
modify protected linear/NVMe routes. Native/CPU contracts must precede a
separately approved CUDA correctness/performance gate. No correction or further
CUDA execution was performed in this diagnostic turn.
