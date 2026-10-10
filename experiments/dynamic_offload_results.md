# DDO native VMM benchmark results

Selected development acceptance, updated 2026-10-04. DDO is model-agnostic;
LTX is the benchmark workload, not a library dependency or architecture policy.
Intermediate validation runs are retained locally in the internal implementation
report, separately from this public summary. Recorded measurements and raw
benchmark artifacts are unchanged.

## Windows image host migration gate — 2026-10-10

One approved BF16 image generation completed in the independent `ddo-examples`
uv-managed `.venv`: real prompt, 1280x704, seed43, eight steps, no PAG,
native encoder then native transformer. Preserved active4GiB, total staging1.5GiB,
source/registration24.625GiB and candidate DLL SHA256
`755fcdd8dd1a72b371402750c071ca4b2146c0ac6bfe6b94c9b78f50f1ebb729`.
No native build, library promotion, budget or runtime-default change.

| Measurement | Historical staged Windows gate | New host, single run |
| --- | ---: | ---: |
| CUDA warm steps2–8 | 1.395602 s/it | 1.524931 s/it |
| CUDA transformer phase with dependencies | 16.498256 s | 18.657049 s |
| Encoder phase | 12.6624 s | 20.6153 s |
| RunTracker total | 38.5135 s | 50.7519 s |
| Sampled peak RSS | 27.3479 GiB | 27.4590 GiB |

Functional migration **PASS**: exit0, saved plausible 1280x704 image; selected
fault/admission/read/plan-mismatch/pool-quarantine counters zero, temporary
fallbacks zero, encoder tickets31→0 and transformer736→0, transfer-owned
events zero at final snapshots and all phase monitor errors null.
Before denoise, encoder source process committed/registered bytes are zero.
Current sampled driver GPU absolute peak7.716034GiB is distinct from
baseline-relative delta6.6892GiB (baseline1.026855GiB). This is not WDDM shared
memory accounting and is higher than the historical approximately6.97GiB peak.

Do not accept performance parity from this run: warm timing is approximately9.27%
higher, total approximately31.78% higher. The historical gate is a reference,
not a contemporaneous matched control; the timing difference's cause is not
established. No pixel-equality comparison or repeated benchmark was performed.
Artifacts remain ignored under `ddo-examples/outputs/migration_image_gate/`;
PNG SHA256 `877876a4a8e1143a59e51a737ef3eb7f05a27980a6a44d20fed80d2ab581cced`.
No further CUDA run is authorized by this result.

## Video compact timestep / residual CPU checkpoint — 2026-10-07

The sibling video host and editable local Diffusers fork now expose separate,
default-OFF compact timestep and inference-owned residual-reuse options.
Consolidated CPU regression sequence: **147 passed, 11 subtests passed**.
Compact modulation expansion and residual reuse pass exact small FP32/BF16
checks; changed projection dimensions have a separate tiny CPU arithmetic bound,
not approved real-model CUDA equivalence. Admission uses prepared CPU masks,
retaining dense execution for spatial variation and reference-layout mismatches.
No CUDA run, allocator build, budget/default change or performance acceptance
followed. Native normalization/modulation and paired RoPE fusion remain pending.
Host implementation and limitations: `E:\LTX-2\model_execution_optimizations.md`.

## Video feed-forward chunk A/B — 2026-10-07

Approved two-step diagnostic: BF16, 1280x704, 8 s at 24 fps (193 frames),
seed 43, no audio conditioning, latent output, without Nsight. Transformer
ceiling 1.5 GiB, encoder ceiling 4 GiB, source cache/registration 24.625 GiB,
staging 1.5 GiB, phase-local modulation enabled and normalization chunk 1024.
Only the feed-forward chunk changed; DLL and other requested controls matched.

| Observation | FF chunk 1024 | FF chunk 2048 |
| --- | ---: | ---: |
| CUDA step 1 | 52.8863 s | 50.2793 s |
| CUDA step 2 | 44.6857 s | 42.6468 s |
| CUDA denoise phase | 97.5960 s | 92.9493 s |
| Total, latent output | 136.0103 s | 130.9887 s |
| Sampled absolute GPU peak | 7.9995 GiB | 7.9995 GiB |

Second-step improvement was **4.5629%**, below the predeclared 5% threshold.
CPU comparison found finite but nonidentical video/audio latents; relative RMS
differences were **4.8191% / 2.5589%**. No numerical tolerance was approved;
do not claim equivalence or rendered-quality regression from these differences.
Both runs exited 0, with selected native failure/quarantine/fallback counters
zero and diagnostic events released. **Not accepted for promotion**: preserve
chunk 1024, existing binaries, defaults and frozen routes. This single ordered
pair does not establish stable performance or eight-step/video-output parity.
Artifact paths and numerical details: [auxiliary gate](../docs/native-execution-auxiliary-gate.md).

## Video phase-local modulation diagnostic — 2026-10-06

Approved bounded Windows comparison: BF16, 1280x704, 8 s at 24 fps, seed 43,
two denoise steps, no audio conditioning, latent output. Same joined-span DLL,
2 GiB transformer/4 GiB encoder ceilings and 1024 FF/norm chunks; only the
local Diffusers `--phase-modulation` option changed. This is not the image
benchmark below or an eight-step/full-decode video benchmark.

| Observation | Original modulation | Phase-local modulation |
| --- | ---: | ---: |
| Denoise CUDA phase | 300.5835 s | 98.6875 s |
| CUDA step 1 / 2 | 104.2100 / 196.2883 s | 49.8097 / 48.8539 s |
| Total, latent output | 339.5551 s | 138.3953 s |
| Cumulative observed native activation-pool peak | 6.1523 GiB | 5.1445 GiB |

Completed with exit 0, no selected failures/quarantine/temporary fallbacks,
retired source tickets and released diagnostic events. Saved video/audio
latents are finite and bit-exact against the control, with identical artifact
SHA256. Sampled GPU baseline-relative peak remains 6.9727 GiB: pool reduction
is not a claim of reduced whole-GPU peak. Defaults and image/linear/NVMe paths
remain unchanged; option remains explicit. Two steps do not establish settled
steady-state or eight-step performance. Full hashes, paths, scope and evidence:
`docs/native-execution-auxiliary-gate.md`.

The separately approved **eight-step** continuation completed with exit 0:
denoise **454.5071 s**, latent-output total **494.8370 s**, cumulative native
pool peak **5.1445 GiB**. CUDA steps: **47.28 / 47.42 / 50.87 / 43.62 / 44.17 /
46.55 / 129.36 / 45.19 s**. Steps 2–8 mean **58.17 s**, median **46.55 s**;
the seventh-step outlier prevents approximately 49 s/step stability acceptance.
All saved latents are finite, final selected safety counters are zero and
retirement completed. This is not decoded-video quality, eight-step bit-exact
parity or a matched comparison against historical MP4 totals. No new test or
default promotion followed; evidence is in the same gate document.

A later separately approved video-output attempt was interrupted 120 s after
its eighth host callback, without an MP4 or final timing/cleanup report. Its
command inadvertently selected CPU/non-streaming decode instead of the prior
validated streaming CUDA decode; the finalization timeout is not a matched
regression. An independent observer recorded approximately 26.5–26.8 GiB of
worker shared GPU memory, sustained graphics/memory clocks and lower power
during slow intervals. This does not prove WDDM paging causality. No default
or allocator change was made; full-video acceptance remains open. Evidence
and the command-level correction are recorded in the same gate document.

The separately approved corrected streaming-CUDA MP4 repetition completed
with exit 0: **697.7858 s total**, **612.0763 s CUDA denoise**, **45.0749 s
decode/save**. CUDA steps: **56.35 / 46.45 / 65.25 / 122.35 / 64.32 / 49.93 /
91.32 / 116.02 s**, steps 2–8 mean **79.38 s**. The runner validated 193 frames
at 1280x704 and finite audio; selected final safety counters were zero,
tickets ended at zero and diagnostic events were released. Native pool peak
remained **5.1445 GiB**. Output/retirement completed, but denoise stability and
performance acceptance remain open; no default promotion or further execution.
Full command, artifact hash and limitations are in the same gate document.

A separately approved single comparison changed only the transformer resident
weight ceiling from **2 to 1.5 GiB**: CUDA denoise **436.0448 s**, decode/save
**42.7006 s**, total **519.4920 s**, exit 0. CUDA steps **48.28 / 44.54 / 57.53 /
112.73 / 46.47 / 41.95 / 42.16 / 42.33 s**; steps 2–8 mean **55.39 s** includes
the fourth-step outlier. Effective resident weights were approximately
**1.4766 GiB**. The MP4 hash matches the preceding 2 GiB run; selected safety
counters were zero and retirement completed. Observed denoise improved
approximately **28.8%**, but the slow step remains and one run does not prove
causality/stability. Native pool and sampled whole-run GPU peaks were unchanged.
No default promotion or further benchmark; details are in the same gate document.

User-run Nsight CUDA/WDDM diagnostic on 2026-10-07 at the same **1.5 GiB**
ceiling completed with CUDA denoise **346.2786 s**, total **430.2082 s** and
steps **41.63–45.16 s**, without reproducing the slow interval or recorded
standby purge. Saved MP4 hash matches the preceding two runs; selected safety
checks and retirement completed. This instrumented run is not a matched
benchmark or stability acceptance. CUDA/WDDM data is present, but the profiler
warns that driver 13.4 is unsupported and uses 13.2 tracing libraries.
Limitations and trace evidence are recorded in the same gate document.

User-run standby purge before the transformer was confirmed (2.9247 s) in
the subsequent instrumented run. Without/with purge: CUDA denoise
**346.2786 / 345.5177 s**, mean steps 2–8 **43.0134 / 43.0688 s**, full total
**430.2082 / 430.7291 s**. Saved MP4 hashes match, selected safety counters
were zero and cleanup completed. This pair shows no meaningful purge benefit
for this workload; neither run reproduced the slow window. Keep `vmm_fast`
standby purge disabled by default and the separate PyTorch host-cache release
intact. No further repeat or code/default change; full scope is in the gate document.

## Workload and scope

- LTX 2.3 distilled image model, BF16 transformer and text encoder.
- Real prompt, 1280×704, seed 43, eight steps, no PAG, one generation.
- Phase-staged native encoder → unload → native transformer → VAE → save.
- GPU: NVIDIA RTX 3060 Ti, 8 GiB. Active weight budget 4 GiB; native source
  storage and registration ceilings 24.625 GiB each; total staging 1.5 GiB;
  native PyTorch allocator cache 512 MiB. RAM must also fit loader/runtime
  demands and safety margins; this is not a low-system-RAM benchmark.
- Normal linear defaults and the protected NVMe route are unchanged. Native
  component ownership remains explicit opt-in, not automatic encoder support.

## Final staged observations

| Scope | Windows, 2026-10-03 | Native Linux, 2026-10-04 |
| --- | ---: | ---: |
| Encoder load / native attachment | 0.7951 / 2.5477 s | 0.6240 / 1.2323 s |
| Entire prompt phase | 12.6624 s | 11.7280 s |
| Transformer load / prepare | 0.3965 / 0.6261 s | 0.3953 / 0.7808 s |
| First transformer CUDA step | 6.729011 s | 5.788787 s |
| Steady state, CUDA mean of steps 2–8 | 1.395602 s/step | 1.050970 s/step |
| Transformer CUDA phase, including dependencies | 16.498256 s | 13.145578 s |
| Entire Generate phase, including connectors/setup | 24.2141 s | 19.0590 s |
| VAE phase / image save | 1.5570 / 0.0760 s | 1.2131 / 0.2129 s |
| RunTracker total | 38.5135 s | 32.2142 s |
| Sampled peak process RSS | 27.3479 GiB | 27.3482 GiB |
| Sampled absolute driver GPU usage | approximately 6.97 GiB¹ | 6.499573 GiB |

¹ Windows absolute use is reconstructed from delta 5.9403 GiB plus rounded
console baseline 1.03 GiB; an exact baseline was not persisted in that historical
JSON. Linux persisted baseline 0.722290 GiB and delta 5.7773 GiB separately.
Missing historical fields are unavailable, not zero.

RunTracker includes encoder, generation, VAE and saving, but excludes interpreter,
import and bootstrap startup. CUDA events measure the compute stream with
dependencies; host callbacks are not CUDA timing. First step and steady state
are separate. Nested setup timings must not be summed.

These are single staged runs on different OS/driver environments, not a statistical
cross-platform comparison or process-cold benchmark. File/driver cache state was
not equalized; no causal explanation of the timing difference is established.
Separate Windows transformer reuse validation observed 1.401990 / 1.397915 /
1.394595 s/step over three generations; it does not establish Linux repetition
or whole-pipeline repeated-generation support.

## Correctness and bounded safety

Both selected staged runs completed with real prompt encoding, native denoising,
a visually checked 1280×704 image and saved reports. No temporary tensor fallback,
source admission denial/bypass, execution-plan mismatch, allocator quarantine or
memory-monitor error was reported. Encoder source counters were zero before
transformer denoise; tickets were collected to zero at completion boundaries.

Linux transformer cache committed/registered 24.25 GiB, with 1.5 GiB staging
separate. Aggregate source reads 25,846,148,360 bytes versus 187,588,136,576 DMA
bytes show reuse, not a per-step read trace. Memory was sampled every 100 ms;
peaks are not instantaneous enforced caps or WDDM shared-memory measurements.
A plausible image does not prove cross-platform numerical/pixel equivalence.
The selected Windows encoder additionally passed its embedding/mask reference check.

Critical source, registration, event, ownership and cancellation contracts passed
in their recorded platform scopes. This is not arbitrary-model, multimodal,
multi-GPU, compiled-forward or driver-loss recovery acceptance. Quantized forwards
retain compatibility routes. No native VMM support is promised for WSL or macOS.

## Evaluation against Diffusers

No new matched official Diffusers block/leaf benchmark was run with this final
native staged encoder configuration. Earlier linear/compatibility measurements
are historical and must not be used to calculate a speedup for this version.
The table reports DDO performance, not superiority over official Diffusers.

For a matched evaluation, hold model revision, precision, prompt, seed, shape,
steps, guidance, phase staging, attention implementation and timing boundaries
constant. Record dependency/driver versions, executed component routes, memory
definitions and cache conditions. Report setup, first step, steady state,
complete denoising and full pipeline separately. Native allocator installation
requires a fresh process; label warm generations separately from cold invocations.

## Reproduction

See the [benchmark manifest](../docs/benchmark-reproduction.md) for environment,
external host fixture, model revision, commands and native binary identity.
[Runtime support](../docs/native-support-status.md) and the
[metric catalog](../docs/metrics.md) define integration and reporting semantics.

Python wheels do not bundle native binaries. Reproduction currently requires
the allocator submodule/platform build and sibling host fixture, not a
self-contained installed-wheel benchmark. No additional benchmark or publication
is implied by this report.

## Selected Windows video host gate (2026-10-05)

The sibling `E:\LTX-2` multimodal host completed a selected native transformer
case:1280x704, five seconds/121frames at24fps, seed43, BF16, real prompt/image/
audio conditioning and eight distilled steps. Native weight residency was2GiB;
total staging1.5GiB and retained source storage/registration24.625GiB remained
unchanged. The host explicitly enabled1024-token standard Diffusers feed-forward
and RMSNorm chunking to bound activation temporaries. These are optional host
experiments, not DDO residency heuristics, new DDO presets or image defaults.
The selected allocator binary and native implementation were unchanged.

The preceding approved four-step synchronous diagnostic completed generation,
unload, decode and save in202.3265s total (128.1s generation, diagnostic only).
The conditional eight-step run, without launch blocking, completed those same
phases in308.7771s total (236.4s generation; console timing rounded). Progress
averaged29.46s/step including startup. No speedup claim or matched image/Diffusers
comparison is made for this larger video workload.

The eight-step report recorded192materializations and zero fault failures,
temporary tensors and collected span-pool quarantine bytes. Tickets were
collected from2 to0; transformer hooks were removed before decode. The encoder
and auxiliary components retained their existing group-offload/staged routes.
Observed absolute driver usage after generation was5.18GiB (4.15GiB above the
baseline); tracker RSS observations reached29.10GiB. Generation monitoring was
disabled: these are not continuous denoise peaks or enforced caps.

MP4/JSON artifacts are in the host's
`outputs/ddo_native_video_20261005_115722_179/`; the MP4 has1280x704/121frames and
an audio stream. CPU decode of all frames passed; this does not establish visual
quality or full-model GPU numerical equivalence. Reproduction details and the
preceding failed candidates remain in the host's `ddo-native-video.md`. The
selected gate is not a promise for arbitrary video duration, shapes or models,
and the accepted image route was not modified or rebenchmarked.

### Video input-audio conditioning control (2026-10-05)

The approved eight-step control changed only `LTX_USE_AUDIO_CONDITION=0`;
the console confirmed `use_audio_condition=False`. All selected shape, seed,
native budgets, activation chunks and allocator binary settings above were
preserved. Audio-video transformer execution and generated audio remained
enabled. This removes input-audio conditioning, not the audio model branch.

| Case | Progress average | Generation phase (console) | Full run (JSON) |
| --- | ---: | ---: | ---: |
| Previous image + input-audio conditioning | 29.46 s/step | 236.4 s | 308.7771 s |
| Image conditioning, no input-audio conditioning | 27.86 s/step | 223.6 s | 297.2001 s |

The observed progress average decreased by approximately 5.4%; the generation
phase decreased by 12.8 s. These are single-run observations, not an isolated
or repeat-validated estimate of the conditioning cost. All eight steps,
hook removal, transformer unload, video/audio decode and save completed without
a reported failure. The console recorded 192 materializations, zero temporary
fallbacks, zero collected span-pool quarantine bytes and tickets collected from
2 to 0. Absolute driver usage after generation was 5.17 GiB; tracker RSS
observations reached 28.748 GiB. Generation monitoring remained disabled;
these observations do not establish continuous denoise peaks.

Artifacts: sibling host `outputs/ddo_native_video_20261005_201936_215/`, with
MP4 and JSON under `metrics/`. CPU decoding verified all 121 frames at
1280x704, a nonconstant first frame and the presence of an audio stream;
visual/audio quality and GPU numerical equivalence were not assessed.
The saved JSON does not expose the input-audio switch as a top-level field;
the executed command and console configuration establish this control's route.
No library, native allocator, preset or image runner was changed.

### Video native pool opportunistic retirement (2026-10-05)

The approved candidate changed only native activation-cache trimming: `free`
queries consumer completion and retires only fully completed backing, while
allocation-pressure retirement retains its event waits. Admission and cache
ceilings, host chunks, BF16, 1280x704, 121 frames, 24 fps, seed 43, eight steps
and the no-input-audio condition were preserved. The accepted binary and host
defaults were not replaced.

| Matched no-input-audio case | Progress average | Generation phase (console) | Full run (JSON) |
| --- | ---: | ---: | ---: |
| Accepted allocator control | 27.86 s/step | 223.6 s | 297.2001 s |
| Nonblocking opportunistic-retirement candidate | 26.09 s/step | 209.4 s | 282.7551 s |

Observed progress improved by approximately 6.4%, generation by 14.2 s and
total by 14.4450 s (4.9%). These are single-run observations, not repeat-validated
speedups. Pool eviction wait fell from 59.075447 s to 0.0001051 s cumulatively
through the report snapshot, while native batch-submit time rose from
27.2338389 s to 49.0033472 s. This changed scheduling/overlap; neither counter is
an independent denoise cost or GPU DMA elapsed time. Do not sum these counters
or claim the entire removed pool wait as elapsed-time savings.

Seven CPU contracts and six small CUDA stream/cache contracts passed before
the model run. Generation, cleanup, unload, decode and save completed with
zero reported pool failures, quarantine bytes, temporary tensors, execution-plan
mismatches and source-read failures. Materializations remained 192 and execution
buffer uploads remained 512. Source tickets were collected from 2 to 0.
Post-generation driver usage was 4.95 GiB absolute; tracker RSS observations
reached 29.0081 GiB. Generation monitoring was disabled; these are not continuous
denoise peaks. Pending pool ownership remains accounted until completion is
observed, even if the GPU already completed the work.

Candidate SHA256:
`E478BB872F593C0BCD272DFDBC9389DC430EA04951D4B95F2FF3197F08B218DE`.
Artifacts: sibling host `outputs/ddo_native_video_20261005_203826_052/`.
CPU inspection decoded 121 frames at 1280x704, one audio stream and a
nonconstant first frame. Quality and GPU numerical equivalence were not assessed.
Commands, ownership details and limitations are recorded in
`docs/native-pool-nonblocking-gate.md`. The candidate remains explicitly selected
for this gate, not promoted to host defaults; the image workload was not rerun.

### Video Diffusers activation-lifetime candidate (2026-10-05)

The source-only candidate releases consumed local activation references in
`LTX2VideoTransformerBlock.forward`, without arithmetic or attention changes.
The video environment used the editable local fork, branch `ltx-image`, HEAD
`0424acb0728f5421d20852728ae9817482a1ef20`, plus the recorded lifetime patch.
DDO dependency pins and the sibling image environment were not updated.
The transformer source before the patch matched the pinned fork; the intervening
commit changes a separate image pipeline, not this video transformer.

The approved tiny real-block CUDA gate passed exact FP32/BF16 output comparisons
with both cross-modal directions enabled, using the normal PyTorch allocator.
Its PyTorch allocated peak was 8596992 bytes. The conditional full video used
the same native pool candidate and all conditions/budgets from the preceding
no-input-audio case. This separates GPU arithmetic checks from native lifetime
integration; neither establishes general image/model non-regression.

| Case | Progress average | Generation phase (console) | Full run (JSON) |
| --- | ---: | ---: | ---: |
| Native pool candidate, retaining forward | 26.09 s/step | 209.4 s | 282.7551 s |
| Native pool candidate, early-reference-release forward | 29.44 s/step | 236.1 s | 310.9901 s |

The observed denoise average was 12.8% slower and total increased by 28.2350 s.
This candidate did not meet the selected performance criterion and is not
recommended for promotion on this evidence. It completed generation, cleanup,
decode and save with zero reported pool failures, quarantine bytes, temporary
tensors, source-read failures and execution-plan mismatches. Materializations
and execution-buffer uploads remained 192 and 512; source tickets reached zero.

Effective cache retention differed despite identical configured limits:
retained source bytes fell from 22604243016 to 16912135472, peak committed bytes
from 23085449216 to 18387828736, and source-cache pressure evictions rose from
371 to 480. Native bypass/source-read bytes rose from 102139290488 to
139096044040. These are bytes serviced through the file-backed bypass route,
not proof of physical-disk reads. Batch-submit wall time rose from 49.0033472 s
to 84.2293399 s; pool eviction wait remained negligible at 0.0001577 s.
The native cache respects dynamic RAM/commit headroom, not just its configured
ceiling. These observations do not isolate the source patch's causal effect or
identify the reason for the headroom difference. Do not compensate by raising
limits or disabling host-memory admission guards.

Driver usage after generation was 4.95 GiB absolute; tracker global RSS
observations reached 25.7256 GiB, including the text-encoder phase. Generation
monitoring remained disabled; these are not continuous denoise peaks.
CPU inspection confirmed 121 frames at 1280x704, one audio stream and a
nonconstant first frame. Quality/full-model numerical equivalence was not assessed.

Artifacts: sibling host `outputs/ddo_native_video_20261005_210813_488/`.
Source SHA256: `581703C58AF60D1D124C087265465E13C585D2B8CFB486C510A1A5B082F265E8`.
The local patch and editable installation remain diagnostic candidates, not
published dependency updates. No rerun, budget change or modular custom-block
implementation followed this gate. Preserve the prior 26.09 s/step reference;
resolve effective host-cache comparability before drawing a speedup conclusion.

#### Read-only source-cache pressure audit

The persistent transport leaves `host_headroom_bytes` at its native API default
of 1 GiB and resolves 64 MiB source slots for this model-window workload.
Before every cache request, `poll_pressure()` samples Windows
`GlobalMemoryStatusEx`: physical RAM available and available commit capacity
(`ullAvailPageFile`, not free space on the pagefile disk). It uses the smaller
observation. Below 1.0625 GiB it suspends growth; while suspended and below
1.125 GiB it discards completed tail storage in 64 MiB units. Growth resumes
once the upper threshold is reached. An unavailable observation also suspends
growth. Referenced/filling slots are not discarded to force room.

For the persistent-buffer route, explicit registration trimming preserves
storage; the pressure path explicitly requests storage discard. Thus the
371/480 pressure-eviction counts reflect this native headroom response, not a
reduced configured cache ceiling. Zero `admission_denials` does not mean no
pressure: misses while growth is suspended bypass before slot admission, and
are counted separately as 10702/14601 pressure-blocked requests. A final
`pressure_suspended=0` does not erase earlier pressure episodes.

Both reported process limits were 28051505152 bytes (26.125 GiB, including
1.5 GiB registered host staging), and neither cache approached the configured
24.625 GiB ceiling at its recorded commitment peak. The zero DXGI system
headroom option in the host JSON is a different control; it does not disable
the source cache's 1 GiB host reserve. Do not conflate GPU weight budget,
registered host staging, source commitment and driver shared-memory usage.

The existing snapshot cannot identify whether RAM, commit, failed observation,
or outside-process consumption initiated each pressure episode. Current machine
observations cannot reconstruct that history. A bounded future diagnostic should
capture native observations at admission/pressure transitions, their RAM/commit
values, thresholds, committed/registered bytes, reclaim outcome and reason, via
DDO's public report rather than runner-internal metric accesses. Preserve all
limits and ownership rules; do not add per-tensor logging or an automatic budget
increase. No code, binary, environment or CUDA execution changed in this audit.

### Native pressure observations candidate (2026-10-05)

Following approval, added fixed-size native pressure observations and an optional
snapshot export, propagated through DDO's public report and readable profile.
No admission threshold, cache ceiling, budget, ownership policy or additional OS
poll changed. Candidate SHA256:
`15D349814F4E6D19C3F055D91BB91F018125E2306ACE8FB60E484FCD00976323`.
Eight C++ CPU contracts and 59 Python tests/two subtests passed. The old selected
DLL remains usable: six Python pressure/report tests passed, optional export test
skipped. No CUDA execution or performance claim for this binary. The two small
CUDA contracts followed conditionally by one unchanged video case are proposed
in [the gate](../docs/native-pressure-observations-gate.md), pending confirmation.

### Approved pressure-observation video result (2026-10-05)

Both selected small CUDA contracts passed after repairing an inaccessible pytest
temporary directory with a fresh workspace base directory. The approved matched
BF16 video then completed: 1280x704, five seconds, 121 frames at 24 fps, seed 43,
eight steps, no input audio, unchanged 2 GiB GPU weight budget / 1.5 GiB staging /
24.625 GiB source and registration ceilings. Local lifetime-patched Diffusers
remained installed. Progress final average was 28.60 s/step, console generation
229.4 seconds and JSON total 303.0063 seconds: still slower than the 26.09 s/step
pool candidate, not a demonstrated performance improvement.

The new public report recorded 241 pressure entries, all commit-limited, and 241
recoveries; no RAM-limited entries or failed observations. Minimum sampled RAM
available was 17.1824 GiB versus only 0.7825 GiB available commit. Pressure evicted
476 slots and released 31943819264 bytes cumulatively. Final retained bytes were
16979244336 (15.8132 GiB), bypass route bytes 140737816120. These do not establish
physical disk traffic or identify the consuming allocation. Post-run read-only
CIM showed a fixed 32 GiB pagefile, not automatically managed; no OS setting changed.

Generation/cleanup/decode/save completed, tickets reached zero, with zero reported
pool failures/quarantine/temporary tensors/source-read failures/plan mismatches.
CPU media inspection decoded 121 frames at the requested shape/rate and one audio
stream. Quality and full-model numerical equivalence remain unassessed. Driver
after-generation usage was 4.95 GiB absolute; RAM observations reached 25.7253 GiB
including encoding, not continuously sampled denoise peaks. Encoder used group
offload, not the native ownership route. No default promotion or extra run.

Artifacts: sibling host `outputs/ddo_native_video_20261005_221330_003/`.
Detailed observations and conditions: [pressure gate](../docs/native-pressure-observations-gate.md).

### Pre-transformer PyTorch host-cache boundary candidate (CPU only)

Added an explicit opt-in DDO helper to release unused PyTorch pinned host blocks
once after prior phases complete, before native transformer setup. The video
runner's new flag is disabled by default and scoped to native VMM; image, linear,
NVMe, native DLL, Diffusers forward and memory ceilings remain unchanged.
The public report separates raw inconsistent active counters from validated
values and records allocator/CPU observations before and after release.
60 DDO CPU tests plus two subtests and 13 video-host CPU tests plus 12 subtests
passed. No CUDA execution, measured recovered bytes or performance claim for
this candidate. Proposed one small CUDA gate and one matched video are documented
in [the boundary gate](../docs/pytorch-host-cache-boundary.md), pending approval.

### Approved early host-cache release video result (2026-10-05)

The small CUDA live-pinned-block/copy gate passed, then the approved matched
video with `--release-pytorch-host-cache-before-transformer` completed. Same
BF16, 1280x704, five seconds, 121 frames/24 fps, seed 43, eight steps, no input
audio, DLL/local Diffusers forward, activation chunk sizes and budgets.

| Measurement | Without early release | With opt-in release |
| --- | ---: | ---: |
| Final progress average, s/step | 28.60 | 26.23 |
| Generation console seconds | 229.4 | 210.4 |
| Full run JSON seconds | 303.0063 | 283.1354 |
| Retained native cache bytes | 16979244336 | 20322541640 |
| Commit-pressure episodes | 241 | 206 |
| Native cache bypass-route bytes | 140737816120 | 117728656640 |

The boundary relinquished 3487499715 PyTorch allocator bytes (3.248 GiB) before
transformer setup; adjacent snapshots showed similar-scale private commitment
reduction and available-commit increase. New validated active-byte fields stayed
null because the raw Torch active counter exceeded allocated bytes. Generation
improved by 19.0 seconds and total by 19.8709 seconds in this single sequential
comparison. No repeat-validated causality or new overall record: the earlier pool
case remains 26.09 s/step / 209.4 s generation / 282.7551 s total.

Pressure remains: all 206 episodes were commit-limited. No guards/ceilings or
pagefile policy changed. Generation/cleanup/decode/save and CPU frame inspection
completed, with zero reported temporary tensors, mismatches, source-read failures,
pool failures or quarantine and tickets reaching zero. Full-model numerical
equivalence/quality unassessed; no continuous denoise memory peak measurement.
Artifacts: sibling host `outputs/ddo_native_video_20261005_223421_903/`.
Keep the flag opt-in, defaults/frozen image routes unchanged. Detailed report:
[boundary gate](../docs/pytorch-host-cache-boundary.md).

### Approved residual CPU source detachment video (2026-10-06)

The single approved video added only
`--detach-transformer-auxiliary-cpu-sources` to the preceding phase diagnostic:
BF16, 1280x704, five seconds, 121 frames/24 fps, seed 43, eight steps, no input
audio, unchanged pressure-observation DLL/local Diffusers forward, 2 GiB resident
weight ceiling, 1024-token feed-forward/normalization chunks and early host-cache
release. Artifact: sibling host
`outputs/ddo_native_video_20261006_091222_765/`.

| Measurement | Previous phase diagnostic | Residual detachment |
| --- | ---: | ---: |
| Final progress average, s/step | 26.35 | 24.87 |
| Generation console seconds | 211.4 | 199.7 |
| Full run JSON seconds | 283.9525 | 276.9484 |
| Attached process private commitment, GiB | 44.0165 | 8.5854 |
| Retained native cache bytes | 19999143216 | 26362339400 |
| Commit-pressure entries / recoveries | 213 / 213 | 0 / 0 |
| Native cache bypass-route bytes | 118439786536 | 92894603200 |
| Native cache batch-submit seconds | 59.3932947 | 20.4131390 |

DDO copied 8257536 bytes once across 192 CPU owners/extents from four shards,
with zero unbound CPU groups and no GPU placement change. The boundary took
0.0710587 seconds. Adjacent private-byte samples fell from 47266713600 to
9218457600: 38048256000 bytes (35.4352 GiB) recovered in this workload. Available
commit increased by 38049525760 bytes. This real-model observation supersedes
the unresolved recovery hypothesis for this selected case, not the preserved
synthetic CPU probe's different result. The generic operation's
`commit_recovery_proven=false` remains intentionally conservative; OS samples,
not copied payload size, establish the measured recovery.

Native cache retained 24.5518 GiB within the unchanged 24.625 GiB ceiling. All
30841 pressure observations were valid; minimum sampled available RAM/commit
were 10427731968/31595470848 bytes. Required fault/guard/CUDA/capacity/source-read/
pool failures, temporary fallbacks, plan mismatches, quarantine and failed
reclaims were zero. Tickets were 0 to 0; materializations/buffer uploads stayed
192/512. All five phase reports completed without inspection errors.

Generation improved by 11.7 seconds and full runtime by 7.0041 seconds in this
single sequential comparison; no repeat-tested causality or general model/image
regression claim. More useful cache residency raised observed global tracker
RSS to 29.1745 GiB, versus 25.7260 GiB previously; this is not a continuous denoise
peak. After-generation private bytes were 38267658240, below the prior
69840949248. Driver usage after pipeline return remained about 4.95 GiB absolute
/ 3.93 GiB over baseline. Encoder execution remained Diffusers group-offload,
not native encoder execution. Cleanup, decode and save passed. CPU media
inspection decoded all 121 frames, 1280x704/24 fps, with one generated audio
stream and first-frame pixel standard deviation 51.4805808; quality/full-model
equivalence unassessed. Keep the correction opt-in; no further CUDA run,
parameter expansion, default promotion or compatibility removal is authorized
by this completed sequence.

### Approved read-only video phase diagnostic (2026-10-05)

Same early-release workload/DLL/local forward/budgets, adding only
`--inspect-host-memory-phases`. Artifact:
sibling host `outputs/ddo_native_video_20261005_225817_983/`.
Final progress 26.35 s/step, generation 211.4 seconds, JSON total 283.9525 seconds;
near the prior 26.23 / 210.4 / 283.1354, not a new speed record.

All five public phase reports completed without inspection errors. Loading
increased private commitment from 8.1216 to 43.5684 GiB with little RSS growth.
After early cache release, private bytes were 42.4982 GiB; after native attach,
44.0165 GiB. Registered CPU tensor storage fell from 37977676288 to 8257536 bytes
(4186 to 192 extents), while private commitment did not fall similarly. Residual
scale/shift parameters resolve to shards; their small visible storage lengths
do not enumerate hidden parent mappings. A bounded CPU synthetic bulk-loader
check confirmed a 64 KiB tensor could retain its file mapping, but did not show
immediate private-commit recovery after cloning/dropping the mapped view.
This is an ownership-correction candidate, not proven recovered video memory.

Native cache retained 19999143216 bytes (18.6257 GiB); 213/213 pressure entries/
recoveries remained commit-limited. Generation, cleanup, save and decoding all
121 video frames passed. Required failure/fallback/quarantine/mismatch counters
were zero; tickets collected 2 to 0. No new continuous memory peak or quality/
equivalence claim, GPU-budget increase or default change. Detailed phase table,
CPU limitations and next correction scope are in the
[boundary gate](../docs/pytorch-host-cache-boundary.md).

### Approved modular video smoke gate (2026-10-06)

The sibling host's separate modular runner completed the approved BF16 video
case: 1280x704, five seconds, 121 frames/24 fps, seed 43, two distilled steps,
no input-audio lock and packed latent output (no decode). The pressure-observation
DLL, local Diffusers transformer forward, 2 GiB weight ceiling, 1024-token FF/
RMSNorm chunks, 24.625 GiB source/registration ceiling, early host-cache release
and bounded auxiliary CPU source detachment were unchanged.

Artifact: sibling host `outputs/ddo_modular_video_20261006_102749_566/`.
Denoise phase: 55.2658 seconds; final progress average: 27.54 s/step; full
latent-output run: 92.7974 seconds. This two-step startup-containing smoke gate
is not a matched comparison with the accepted eight-step 24.87 s/step control.
No performance promotion or decode/visual-quality claim.

CPU-only inspection confirmed entirely finite float32 packed video/audio tensors
of shapes [1, 14080, 128] and [1, 126, 128]. Native plan passes: two;
materializations/signature hits/execution-buffer uploads: 48/24/128. Required
fault/guard/CUDA/capacity/source-read/pool failures, temporary fallbacks, plan
mismatches, quarantined bytes and failed reclaims were zero. Cache pressure
entries/recoveries: 0/0; failed observations: zero of 7735 observations. Source
tickets were 0 to 0 at collection. The process retired the native component,
saved latents and exited successfully; the snapshot was taken before retirement,
not a post-retirement ownership census.

Sampled absolute driver peak: 7.6819 GiB, or 6.6551 GiB over the 1.0269 GiB
baseline. Sampled process RSS peak: 28.2050 GiB. Cache retained 26362339400 bytes
within its unchanged ceiling. Minimum available RAM/commit:
2728017920/20441145344 bytes; do not automatically expand memory ceilings.
The earlier failed tokenizer/declared-schedule handoffs and correction evidence
are preserved in the sibling host's `ddo-modular-video.md`.
Existing reference/image/frozen-runtime routes were not changed or rerun.
Real-model numerical parity, decode and an explicitly approved matched benchmark
remain pending; this completed smoke gate authorizes no additional CUDA run.

## Approved modular video eight-step gate (2026-10-06)

Sibling artifact: `outputs/ddo_modular_video_20261006_112207_195/`,
`metrics/modular_run.json`. Selected BF16 workload: 1280 x 704, 5 seconds,
24 fps/121 frames, seed 43, real prompt, image conditioning, eight steps,
no input-audio lock, packed latent output (no decode). Native budgets remain
2 GiB resident weights, 1.5 GiB staging, 24.625 GiB source/registration;
FF/RMSNorm chunks remain 1024. Pressure-observation DLL SHA256 remains
`15D349814F4E6D19C3F055D91BB91F018125E2306ACE8FB60E484FCD00976323`.

Candidate host changes: positive-only distilled connectors, inference-mode
phase execution and generation-local precomputed RoPE. Local Diffusers forward
SHA256: `F37835B2935C01ED1DC23C67BAF702A9B6CB0D399611512A2E627FC23A8BAC71`;
the default non-precomputed path is preserved. 24 CPU contracts previously
passed, including exact small-transformer RoPE parity.

Eight-step progress average: **24.39 s/step**, versus the recorded reference
runner's **24.87 s/step**. Denoise phase including final readback/synchronization:
**195.3200 seconds**. Total latent-output run: **234.9401 seconds**. The approximately
1.9% progress-average difference is one selected observation, not an isolated
optimization effect, repeatability claim or full-pipeline/decode comparison.
The earlier two-step 27.54 s/step result remains startup-containing smoke evidence.

CPU-only readback confirmed finite float32 packed video/audio tensors,
[1, 14080, 128] and [1, 126, 128]. Sampled absolute GPU peak: 7.6155 GiB;
baseline-relative delta: 6.5886 GiB over 1.02685546875 GiB. Sampled RSS peak:
28.4167 GiB. Historical reference memory monitoring was different; no matched
memory-reduction claim. Native minimum available RAM/commit:
3668602880/20231127040 bytes.

Native plan passes/materializations/signature hits/execution-buffer uploads:
8/192/168/512. Required fault/guard/CUDA/capacity/source-read/pool failures,
admission denials, temporary fallback tensors/bytes, plan mismatches,
quarantine and failed reclaims were zero. Cache pressure entries/recoveries:
0/0; failed observations: zero. Retained source bytes: 26362339400; total
resident weights: 2143289344 bytes, within the fixed ceiling. Source tickets:
0 to 0 at nonblocking collection. Snapshot precedes successful retirement,
not a post-retirement ownership census.

Status: selected eight-step safety/performance criterion met; output saved and
run completed. No decoded video, real-checkpoint numerical/visual parity,
native encoder integration or CUDA image-regression claim. No further CUDA
execution authorized by this completed gate.

## Approved staged native encoder/video gate (2026-10-06)

Sibling artifact: `outputs/ddo_modular_video_20261006_113526_724/`,
`metrics/modular_run.json`. Same eight-step BF16/1280 x 704/5-second/24-fps/
121-frame/seed-43 real-prompt image-conditioned workload as the preceding
modular gate; no input-audio lock or decode. Explicit encoder opt-in uses
4 GiB weight ceiling, component ownership, 2 MiB pages and 1 MiB bounded
buffers; transformer retains its approved 2 GiB ceiling. Staging/source/
registration, chunks, DLL and local forward hashes remain unchanged.

Encoder phase: **13.5180 s**, versus **17.7256 s** in the preceding group-offload
run. Denoise phase: **190.3552 s**; final progress average **23.77 s/step**,
versus **24.39 s/step**. Total latent-output run: **228.9680 s**, versus
234.9401 s. Transformer attach increased from 5.1255 to 7.9564 s; do not
attribute all total/denoise differences to the encoder or claim repeatability.

CPU-only comparison of the two saved packed float32 outputs found bit-exact
video [1, 14080, 128] and audio [1, 126, 128]: finite, torch.equal true,
maximum and mean absolute difference zero. This is selected final-latent parity
with the preceding modular group-offload run, not pixel/audio decode parity,
raw encoder-embedding equality or arbitrary-checkpoint support.

Preflight bound 1065/1065 source groups with zero missing/rejected/ineligible
groups or invalid/ambiguous headers; 82 execution units, six bounded buffers,
34,818 buffer bytes. Executed public reports confirm native encoder ownership
with 4 GiB, then native transformer with 2 GiB. Resident-weight snapshots:
4030726144 and 2143289344 bytes. Nonblocking encoder ticket collection: 7 to 0;
transformer: 0 to 0. Successful retirement and process exit 0, no post-retirement
allocation census.

Both reports have zero required fault/guard/CUDA/capacity/source-read/pool
failures, admission denials, temporary tensors/bytes, plan mismatches,
quarantine, failed observations/reclaims and cache pressure entries/recoveries.
Encoder retained/process-registered/committed: 23555211264 bytes; transformer
retained: 26362339400, process-registered/committed: 26440892416 bytes, within
the unchanged source ceiling. These observed snapshots do not constitute
a continuous no-overlap census.

Sampled absolute GPU peaks: encoder 5.0132 GiB, denoise 7.6969 GiB;
denoise baseline-relative 6.6700 GiB over 1.02685546875 GiB. Sampled RSS peak:
28.4537 GiB. Native minimum available RAM/commit: encoder
8268324864/25797361664, transformer 4287311872/19938123776 bytes.

Status: selected staged native encoder/video safety and no-regression gate PASS.
Encoder remains explicit opt-in; streaming decode, decoded-media quality and
CUDA image regression were not tested or promoted. No further CUDA case is
authorized by this completed sequence.

## Approved modular full-video gate (2026-10-06)

Sibling artifact: `E:/LTX-2/outputs/ddo_modular_video_20261006_115322_220/`,
report `metrics/modular_run.json`, media `ltx23_modular_1280x704_5.0s_seed43.mp4`.
The approved eight-step BF16 workload retains 1280 x 704, five seconds,
24 fps, 121 causal-grid frames, seed 43 and no input-audio conditioning.
Native encoder/transformer ceilings remain 4/2 GiB; source/registration
24.625 GiB, staging 1.5 GiB and FF/RMSNorm chunks 1024 are unchanged.
DLL and local Diffusers forward hashes are unchanged from the preceding gate.

Executed from `E:/LTX-2`:

```powershell
.\run_ddo_vmm_video_modular.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-pressure-observations-20261005\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 5 --steps 8 --seed 43 --no-use-audio-condition --output-type video --decode-device cuda:0 --stream-video-decode --ddo-vmm-resident-weight-budget-gb 2 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources
```

Hypothesis: bounded temporal GPU decode completes after transformer retirement
without keeping the entire decoded video on GPU. Acceptance: no native failure
or OOM, expected frame dimensions/count, finite audio and a readable MP4.

Phase seconds: encoder 14.1422, connectors 6.7997, video encoder 3.9754,
empty audio encoder 0.1470, transformer load/prepare 2.1805, attach 8.0256,
denoise 197.0318, decode/save 30.1365. Total: **266.5138 s**;
final denoise progress average: **24.60 s/step**. The preceding latent-only
23.77 s/step was not reproduced; decode follows denoise and is not an
explanation for this variation. The recorded original full-MP4 control total
was 276.9484 s. Neither difference establishes an isolated or repeatable gain.

Sampled absolute driver GPU peaks: encoder 5.1772 GiB, denoise 7.6247 GiB,
decode/save 5.2983 GiB; baseline-relative deltas 4.1504/6.5978/4.2715 GiB
over 1.02685546875 GiB. Peak sampled process RSS: 28.3814 GiB;
decode/save RSS peak 7.4905 GiB. Sampling is not an allocator census or
a matched memory comparison with the historical control.

Both native reports have zero required fault/guard/CUDA/capacity/source-read/
pool failures, admission denials, temporary fallbacks, plan mismatches,
quarantine, failed observations/reclaims and cache pressure entries/recoveries.
Encoder tickets collected 7 to 0; transformer 0 to 0. Resident-weight
snapshots: 4030726144/2143289344 bytes. Retirement succeeded; snapshots
are not post-retirement ownership measurements. Process exit: 0.

CPU-only full PyAV readback confirmed H.264, 121 frames at 1280 x 704,
24 fps, container duration 5.041667 s; AAC stereo at 48000 Hz, all decoded
audio finite. Generated audio has 240480 samples/channel; decoded AAC has
240640 due to codec padding. First-frame RGB standard deviation 51.6658
confirms nonblank variation, not visual quality. No packed-latent comparison
was performed for this MP4 run.

Status: selected full-pipeline execution/media-structure gate PASS. User
visual/audio review remains pending. No general model support, image CUDA
regression, input-audio conditioning or decoded-media parity claim; no
additional CUDA execution is authorized by this completed gate.

## Approved modular duration extension: eight seconds (2026-10-06)

Artifact: `E:/LTX-2/outputs/ddo_modular_video_20261006_120540_047/`;
report `metrics/modular_run.json`; media
`ltx23_modular_1280x704_8.0s_seed43.mp4`. User approved an eight-second
case followed conditionally by ten seconds after checking safety and media.
Exact command is the preceding full-video gate with only `--seconds 8`
substituted for `--seconds 5`. Resolution 1280 x 704, 24 fps, BF16, eight
steps, seed 43, no input-audio conditioning, all ceilings, chunks and
DLL/forward hashes remain unchanged. No implementation or preset change.

Hypothesis: native execution and bounded temporal decode permit longer
duration without raising memory ceilings. Criteria: no OOM/native failure
or memory anomaly, finite audio, complete readable MP4 and 193 frames.

Process exit 0, status completed. Total **593.8051 s**; denoise
**512.8294 s**, final progress average **64.05 s/step**; decode/save
**42.5483 s**. Console cumulative timestamps show first step 179.51 s,
second approximately 86 s, then approximately 41 s per step. The latter
are rounded console differences, not separately instrumented timings;
the full average must not be replaced by the faster late-step estimate.
The duration extension passed capacity/execution, not a performance target.

Sampled absolute GPU denoise peak **7.9995 GiB**, delta **6.9727 GiB**
over baseline 1.02685546875 GiB; sampled RSS peak **29.3297 GiB**.
Decode/save absolute GPU/RSS: 5.3198/11.3892 GiB. Dedicated memory is
near the physical ceiling; no shared-memory or paging measurement was
collected to explain the slow startup. No causal cache/driver claim.

Both native reports have zero fault/guard/CUDA/capacity/source-read/pool
failures, admission denials, temporary tensors/bytes, plan mismatches,
quarantine, failed observations/reclaims and pressure entries/recoveries.
Encoder tickets 7 to 0; transformer 0 to 0, nonblocking collection.
Native minimum available RAM/commit bytes: encoder 12280246272/28619956224,
transformer 6498349056/20888707072. Retirement succeeded.

CPU-only full PyAV readback confirmed 193 H.264 frames at 1280 x 704,
24 fps, duration 8.041667 s; AAC stereo at 48000 Hz, all audio finite.
Generated audio samples/channel: 384480; decoded AAC: 385024 (padding).
Selected eight-second gate PASS, allowing the already approved ten-second
case. Visual/audio quality and numerical parity remain user-review/unverified
items; no image route was modified or rebenchmarked.

### Conditional ten-second result (same approved sequence)

Artifact: `E:/LTX-2/outputs/ddo_modular_video_20261006_121621_526/`,
`metrics/modular_run.json`, `ltx23_modular_1280x704_10.0s_seed43.mp4`.
Executed the preceding full-video command with only `--seconds 10`
substituted, after the eight-second media/native gate passed. No budget,
implementation, preset, DLL or forward change. Expected frame count: 241.

Process exit 0, completed. Total **899.4344 s**; denoise **808.3324 s**,
final progress average **100.97 s/step**; decode/save **53.2628 s**.
Encoder/connectors/video encode/empty audio encode/load+prepare/attach:
13.5706/6.7985/3.6856/0.1415/2.0754/7.5912 s. Console first step:
272.92 s; rounded cumulative differences place second near 177 s and
steps three through eight near 59-60 s each. These are startup-containing
single runs, not a long-duration performance baseline or causal diagnosis.

Sampled absolute GPU denoise peak **7.9995 GiB**, delta **6.9727 GiB**
over the same baseline; peak RSS **31.4914 GiB**. Decode/save absolute
GPU/RSS peaks: 5.2964/16.1433 GiB. No automatic ceiling growth or parameter
calibration. Dedicated GPU memory remains near full, but shared-memory
paging and the cause of the first-two-step cost were not measured.

Both native reports again have zero required failures/denials, temporary
fallbacks, mismatches, quarantine, failed observations/reclaims and cache
pressure entries/recoveries. Tickets: encoder 7 to 0, transformer 0 to 0;
retirement succeeded. Native minimum available RAM/commit bytes: encoder
12664147968/28591906816, transformer 5264330752/19394445312.

CPU-only full media decode: H.264, 241 frames at 1280 x 704, 24 fps,
10.041667 s duration; AAC stereo 48000 Hz, all audio finite. Generated
samples/channel: 480480; decoded AAC: 481280 (padding).

Selected ten-second capacity/media gate PASS: this workload exceeds the
user-reported earlier eight-second duration limit without an OOM. This is
not a claim of faster generation, general maximum supported duration,
matched external-reference performance or image CUDA non-regression.
User visual/audio acceptance remains pending. The approved sequence is
complete; no further CUDA run, tuning or implementation change is authorized.

### Startup allocation diagnosis prepared (non-CUDA only)

Optional driver allocation/pool counters and existing GPU/native timeline
integration are ready in a separate diagnostic build. Normal execution and
the accepted DLL are preserved. Pure-CPU counter contract, 41 Python contracts
with candidate exports and 33 host CPU contracts passed; no CUDA run followed.
See [scope, hash and proposed four-step eight-second gate](../docs/native-allocation-diagnostics-gate.md).
This preparation does not establish the cause of the slow first two steps.

### Eight-second startup diagnostic collected (2026-10-06)

Approved four-step latent-only diagnostic completed, exit 0; packed video
and audio finite. Compute-stream intervals: 204.8341/89.8687/106.2195/93.0882 s;
phase 494.1370 s, total 532.6823 s. Native timeline complete (1187 samples,
zero drops), original safety counters zero and encoder/transformer tickets
retired. Diagnostic-only, not matched performance or image regression evidence.

Measured allocation driver-call host time totals 6.7117 s, insufficient to
explain denoise directly. Transformer-block windows dominate, but are not
kernel-only time. Two first-step access calls returned non-success; existing
access-retry context synchronization is outside the measured wrappers, so
allocation-path waiting remains unresolved. Do not remove that safety wait
or tune ceilings on this evidence. Accepted DLL/policy unchanged. Details and
artifact: [native allocation diagnosis](../docs/native-allocation-diagnostics-gate.md).

### Access-retry timing prepared (non-CUDA)

Separate diagnostic candidate adds existing access-retry synchronization and
enclosing fault timing plus access return categories, without changing policy.
CPU C++ contract and 44 Python contracts passed; previous binary hashes
unchanged. No CUDA execution. Follow-up command/hypothesis/criteria await
confirmation in [native allocation diagnosis](../docs/native-allocation-diagnostics-gate.md).

### Access-retry diagnostic collected (2026-10-06)

Approved same eight-second/four-step latent-only diagnostic completed, exit 0.
Native schema 2 and timeline complete (1187 samples, zero drops), packed
video/audio finite, original native safety counters zero and tickets retired.
Phase 406.3883 s, total 443.9171 s; event intervals
244.9366/61.4196/49.0461/50.8503 s. Not a matched benchmark or a claimed speedup.

Existing model-arena retry synchronization: zero calls/time; enclosing 98 faults:
0.0981724 s. One first-step access returned NOT_READY outside that measured retry;
final inference succeeded. Driver-call categories totaled 7.3684 host seconds.
This rules out the suspected retry wait for this run, not all driver/forward
waiting or WDDM paging. Preserve its safety wait and accepted memory policy.
Unresolved execution-window cost needs operator/host-versus-GPU attribution
before a correction. Artifact and details:
[native allocation diagnosis](../docs/native-allocation-diagnostics-gate.md).

### Sampled operator tracing prepared (non-CUDA)

Generic DDO selected-forward profiling/public reporting and optional host
consumption are prepared, default off. Two selected invocations only; profiling
may synchronize and is not benchmark evidence. 47 DDO CPU and 34 host CPU
contracts passed. Native binary/forward hashes preserved, no CUDA execution.
The next single command and acceptance criteria await user confirmation in
[sampled operator gate](../docs/native-operator-diagnostics-gate.md).

The approved operator diagnostic dispatch was interrupted: execution tool
returned `aborted`, without session/output/exit code. Observed directory
`E:/LTX-2/outputs/ddo_modular_video_20261006_150618_511/` contains no files;
subsequent process inspection showed no active high-memory inference process.
No diagnostic or safety result is available; interruption cause unknown.
No automatic repeat. Relaunch requires confirmation.

### Sampled operator diagnostic completed after approved relaunch

Artifact `E:/LTX-2/outputs/ddo_modular_video_20261006_150751_571/`, exit 0.
Exactly two valid CUDA operator captures; native timeline complete/no drops,
packed video/audio finite and final native safety counters zero; tickets retired.
One recovered access OUT_OF_MEMORY return is recorded separately. Phase
539.3473 s, total 579.3895 s; event intervals
160.6900/233.8860/97.3991/47.2788 s. Diagnostic-only, not matched performance.

Both sampled forwards contain six pageable CPU-to-GPU copies matching the
modulation tables: 172032 bytes/block, 8257536 bytes across 48 blocks. These
tables exceed the existing 16 KiB small-local threshold and remain CPU after
source detachment. Stream waits within `to/copy_` sum approximately 1.20/2.50 s
in the samples, whereas pageable copy device time is approximately 14/16 us.
Waits may drain prior GPU work; do not count them as guaranteed savings or
extrapolate one block to all step cost. Preserve attention/backend/budgets.
Propose accounted native auxiliary ownership rather than a Python residency
cache or shared-threshold change. Full trace interpretation and next gate scope:
[sampled operator diagnostic](../docs/native-operator-diagnostics-gate.md).

## 2026-10-06 — execution-unit auxiliary ownership preparation

Implemented a default-off, model-agnostic native auxiliary ownership candidate
after the bounded operator diagnostic. Auxiliary parameters now join existing
immutable source batches, arena views and event-fenced execution slots without
raising the GPU ceiling or the shared small-tensor threshold. Existing dense
prefetch is preserved; component-wide encoder ownership remains separate.

CPU evidence: DDO selected contracts **110 passed, 2 subtests passed**; host CPU
contracts using `E:/LTX-2/.venv-vmm` **53 passed, 31 subtests passed**. An initial
host collection attempt in DDO's environment lacked `torchaudio`; no CUDA model
failure or dependency replacement occurred. No CUDA gate or new performance
measurement was executed. Earlier accepted results remain unchanged.

The new flag is `--ddo-vmm-execution-auxiliary-ownership`. The unchanged diagnostic
DLL hash is `335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.
See [implementation and proposed small CUDA gate](../docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — auxiliary ownership small CUDA gate

The user approved the two-case native auxiliary lifecycle gate. Actual result:
**2 passed in 5.26 s**, exit 0, using DDO's `.venv-vmm` and the unchanged
diagnostic DLL SHA256
`335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

Both ordinary-error and cooperative-interruption cases passed BF16 numerical
comparison, alias identity, immutable source reuse, zero retained CPU source
bindings, zero asserted fault/fallback/admission/bypass/pending-event counters
and final alias/source/hook cleanup. The small test used its explicit 0.07 GiB
weight ceiling; it did not alter real-model ceilings or the default route.

This is small CUDA contract evidence, not a video performance result, full
allocator-bootstrap gate or image regression acceptance. A real-video bounded
diagnostic with the new flag is proposed, not executed. Its full command,
hypothesis and acceptance criteria are in
[the auxiliary ownership gate](../docs/native-execution-auxiliary-gate.md).

### Auxiliary candidate geometry audit (CPU/meta only)

Read-only local config/header inspection reported CUDA uninitialized. It found
864 added auxiliary groups: 672 previously-small groups (4.125 MiB) and 192
residual groups (7.875 MiB). First-block spans grew 68 → 86, but source windows
stayed 13 and **no unit touched additional windows**. Largest aligned payload
grew 773131264 → 773393408 bytes, while rounded two-slot capacity remained
1547698176 bytes. Increased slot capacity or new file windows therefore do not
explain this candidate's regression. Runtime pressure/cache causes remain
unproven without the missing final counters. Proposed next correction preserves
the existing small-parameter route and adds durable public per-step diagnostics;
no corrective patch or new CUDA execution was performed during this audit.

### Narrowed auxiliary candidate and durable partial metrics (CPU only)

The user approved implementation without CUDA execution. The revised selector
preserves the existing small-parameter threshold (inclusive boundary) and joins
only residual parameters to native batches. Expected first-block spans: 72
instead of the broad candidate's 86; aligned payload 773303296 bytes; unchanged
two-slot page capacity 1547698176 bytes. It includes 192 residual groups while
preserving 672 previously-small groups. CPU/meta inspection reported CUDA
uninitialized; these are not runtime performance measurements.

DDO selected contracts: **120 passed, 2 subtests passed**. Host CPU contracts:
**53 passed, 31 subtests passed**. A new public default-off metrics journal
persists before-denoise and per-step partial snapshots without completion
collection or CUDA waits, and the host enables it only for denoise diagnostics.
The snapshot explicitly does not prove GPU completion or replace final gates.

DLL unchanged; no CUDA or new model execution occurred. The earlier broad
candidate's regression and interrupted artifacts remain preserved. The revised
small CUDA fixture uses the production threshold and checks that a small
parameter retains its existing route; execution still requires confirmation.
See [the updated gate](../docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — real-video auxiliary candidate interrupted

Approved bounded 8-second/4-step latent diagnostic with auxiliary ownership:
`E:/LTX-2/outputs/ddo_modular_video_20261006_155602_695`.
The user requested interruption after a timing regression. Wrapper child exit
-1; enclosing command exit 1. First progress step 473.24 s versus the previous
diagnostic's 160.689953125 s CUDA-event step; second step approximately 198 s
from rounded console elapsed. No four-step average, final metrics, packed
latents or normal cleanup acceptance is available.

Only invocation 0 was exported: pageable H2D copies 6 → 0 and stream-sync
events 6 → 0, with 897 kernel events in both captures. Post-attach CPU source
storage observation was 0 bytes. However, guarded forward publication moves
the native upload preamble inside the profiler scope, so the new capture's
larger pinned byte total is not a matched whole-step transfer comparison.

The candidate is **not accepted for promotion**. Preserve defaults and previous
accepted results; no budgets or attention backend were changed. Static review
identifies a broader change to small-parameter ownership and native batch/view
geometry beyond the six residual table copies, but final native counters are
unavailable and the slowdown's cause is not yet established. Full partial
evidence and the next investigation boundary are in
[the auxiliary ownership gate](../docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — revised residual-only small CUDA gate

The user approved only the two revised small CUDA cases, not another video run.
Actual result: **2 passed in 5.20 s**, process exit 0, DDO `.venv-vmm`, unchanged
DLL SHA256 `335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

With the production 16 KiB threshold, small Parameter identity and CUDA placement
were preserved; native residual aliases, BF16 parity, immutable source reuse,
asserted fault/fallback/admission/bypass/pending-event counters and unloading
checks passed for ordinary error and cooperative interruption. This is acceptance
of the revised small lifecycle contract only, not real-video speed, image
regression or journal publication under CUDA. Defaults and all earlier measured
results remain unchanged. No long or real-model execution occurred in this gate.

## 2026-10-06 — residual-only two-step video diagnostic, interrupted

Approved case: 8 s, 1280x704, seed 43, two steps, latent output, no audio
conditioning, 2 GiB transformer / 4 GiB encoder ceilings, residual-only native
ownership and partial metrics journal; no operator profiler. DLL SHA256 remains
`335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

Transformer attach **8.3725 s**. Partial ownership report confirms 192 groups /
7.875 MiB with the existing 16 KiB threshold. First callback host elapsed
**177.9379351 s**; no second callback arrived before the user-requested stop,
approximately **768 s** later. These are host observations, not final GPU event
timings or a matched benchmark. Forced child exit -1 is not a CUDA exception.

Output `E:/LTX-2/outputs/ddo_modular_video_20261006_165752_982` preserves
`before_denoise.json` and `step_0000.json`; console is
`E:/LTX-2/outputs/residual_auxiliary_console_20261006_165752_896.log`.
No final metrics, packed latents, numerical validation or normal cleanup result
exists. First partial failure/quarantine counters are zero but completion is
unproven. Candidate **not accepted for promotion**; narrowing small-parameter
ownership did not resolve the slowdown. Preserve accepted defaults and budgets.
See [the auxiliary gate](../docs/native-execution-auxiliary-gate.md) for conditions
and limitations. No follow-up CUDA run was started.

## 2026-10-06 — pending-span fragmentation CPU reproduction

### Subsequent joined-span bounded video completion

User-approved repeated 8 s / 1280x704 / seed 43 / two-step latent diagnostic,
same residual ownership and budgets, changing only the DLL to SHA256
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Both 240 s callback limits and the 120 s finalization limit passed; exit 0.
Attach **7.8578 s**, total **339.5551 s**, CUDA phase **300.58353125 s**,
steps **104.2100390625 / 196.28828125 s**, tail **0.08518723 s**.
Two steps do not establish settled steady state.

Output `E:/LTX-2/outputs/ddo_modular_video_20261006_175732_123` contains final
metrics, all three partial snapshots and finite float32 packed video
`[1,22000,128]` / audio `[1,201,128]`, verified independently on CPU. Selected
failure/quarantine/fallback/ownership counters passed; native retirement and
unload completed normally. Activation pool observed peak **6.15234375 GiB**;
GPU peak **6.97265625 GiB is baseline-relative**, RSS peak **29.54099274 GiB**.
Snapshot growth reduction, WDDM paging and sole causal attribution are not
proven. Accepted for this bounded selected-workload completion only, not
eight-step performance, image regression, parity or default promotion.
See [the gate](../docs/native-execution-auxiliary-gate.md) for raw paths,
limitations and the static historical validation label in the JSON.

CPU-only MSVC x64 reproduction against the unchanged production ledger:
**61 checks passed**, exit 0. Four adjacent idle 1024-byte spans with different
pending fences cannot satisfy a 2048-byte request despite 4096 idle bytes.
Completion observations restore the fit; shared-fence pending reuse and
live/quarantine/rollback/byte-conservation controls pass. No CUDA library is
linked, no GPU operation is executed and no production allocator code or DLL
changes. This confirms a conservative ledger limitation, not the cause of the
interrupted video's slowdown. Reproduction and next implementation boundary:
[auxiliary gate](../docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — joined-span small CUDA gate passed

User-approved eight-case sequence: **8 passed in 6.09 s**, exit 0, DDO
`.venv-vmm`. Candidate SHA256 verified:
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
The two new same/cross-stream joined reservations reused the original 2 MiB
address without increased mapped bytes/backing count, retained both unfinished
predecessor events, produced correct consumer values and recorded zero
failure/quarantine counters. Six existing allocator ordering cases passed.
Acceptance is limited to this small native contract: no video performance,
stall-causality or image regression claim. Defaults and old binaries remain
unchanged; no model run followed. Details:
[auxiliary gate](../docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — opt-in joined-span implementation, CPU only

Separate dependency-preserving native multi-span operation implemented behind
the default-OFF `DDO_VMM_NATIVE_JOINED_SPANS` build option. CPU: **176 new checks
passed**, including seven injected metadata failures; **1292 original ledger
checks passed**; stream dependency contract passed. Candidate DLL compiled with
MSVC x64 / CUDA 12.8.93, SHA256
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Prior binaries/defaults preserved. No CUDA execution or video benchmark occurred;
the proposed eight-case small gate still requires confirmation. No performance
or real-model causal claim. Full build and gate boundary:
[auxiliary gate](../docs/native-execution-auxiliary-gate.md).
# Video Triton fusion candidate — 2026-10-07

Experimental opt-in RMSNorm/modulation and paired Q/K rotary callbacks are
implemented in the local editable Diffusers/video host, with generic Triton
kernels in the allocator adapter. Both flags remain off by default. Existing
image runners, allocator binaries, chunks and residency budgets are unchanged.
CPU routing/regression: **164 passed, 11 subtests passed in 10.40 s**.
Kernel CUDA/native-VMM numerical validation is pending user approval; no
performance gain or real-model parity is claimed. Prepared gate and constraints:
the sibling video host's `model_execution_optimizations.md` and
`probe_fused_execution_cuda.py`. Shared cross-modal norms remain unchanged.

The first approved CUDA attempt stopped in the split-RoPE reference because
its reshape used frequency batch 1 for activation batch 2. The gate did not
pass. A reference-only zero-stride batch expansion corrected the synthetic
oracle; candidate kernels, inputs and tolerances remain unchanged. Subsequent
CPU regression: **168 passed, 11 subtests passed in 8.37 s**.

Approved corrected CUDA/native-VMM retry passed with exit code 0: 36 numerical
comparisons, including non-default streams and tiny-model path restoration.
RMS FP32 max absolute error `1.9073486328125e-6`; BF16, rotary, and tiny-model
outputs were exact in the selected sample. 28 RMS/modulation and 16 paired rotary
submissions; allocator reported `pluggable` with PyTorch `2.12.1+cu132`.
The accepted DLL, image host, defaults and budgets remain unchanged. This is a
small-contract gate, not a real-video performance or quality acceptance.

## Video operation Inductor baseline — 2026-10-07

Approved BF16 eager/Inductor/manual-Triton microbenchmark stopped at the
Inductor RMS/modulation precision gate, before timed rounds. Windows CUDA
compilation/execution with the pluggable allocator succeeded, but 9378/4194304
elements exceeded the approved `rtol=atol=1/128` eager-parity bound; max absolute
difference 0.125. Generated code fused reduction/modulation while omitting
intermediate BF16 rounding. The default-policy artifact is preserved at
`E:\LTX-2\outputs\execution_backends_20261007_01`. There are no accepted
steady timings, manual-kernel comparison or offloading conclusions from this
attempt. A precision-cast-preserving repeat requires separate approval.

The approved repeat with `emulate_precision_casts=True` completed, preserving
the same workload, DLL and tolerances; all selected BF16 comparisons were exact.
Warm CUDA-event interval medians in milliseconds (eager / Inductor / manual
Triton): RMS/modulation `0.648862 / 0.158912 / 0.092768`; interleaved paired
rotary `0.976298 / 0.195134 / 0.166758`; split paired rotary
`0.597914 / 0.162181 / 0.137062`. First-call compile/cache time is separate.
Artifact: `E:\LTX-2\outputs\execution_backends_20261007_02\benchmark.json`.
One synthetic sequence, with some overlapping per-round ranges: not a
real-video speedup, general statistical conclusion or group-offloading result.
Original default-precision failure remains preserved; no default promotion.

## Real-video fusion A/B — 2026-10-07

Approved matched fresh-process pair: BF16, 1280x704, 8 seconds at 24 fps
(193 frames), seed 43, two steps, real prompt, image conditioning, no input
audio conditioning, joint audio generation, latent output. Compact timesteps
and in-place residuals off. Identical protected DLL, chunks and budgets;
only the candidate enabled RMS/modulation and paired-rotary fusion.

| Measurement | Baseline | Both fusions |
| --- | ---: | ---: |
| Total run, seconds | 135.6195 | 119.9721 |
| Denoise CUDA phase, seconds | 94.038078 | 81.589656 |
| Second step CUDA interval, seconds | 47.842949 | 40.142801 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.213604 |
| Sampled absolute GPU peak, GiB | 7.9995 | 7.2405 |
| Sampled process RSS peak, GiB | 28.701458 | 28.486149 |

Both completed with exit code 0, zero selected fault/admission/transport/cache
failures, quarantine and temporary fallbacks, and completed timing/cleanup.
Candidate submissions: 576 RMS/modulation and 384 paired rotary. Second-step
interval reduction approximately 16.1%; one ordered pair, not statistical
or final eight-step acceptance. First-step JIT cost was not isolated.

**Numerical acceptance failed:** CPU comparison found finite, nonidentical
latents. Video relative RMS difference `0.02875492`, max absolute difference
`0.87924790`; audio relative RMS difference `0.02205039`, max absolute
difference `0.10100091`. Recorded generation metadata and shapes matched.
The agreed no-divergence criterion was not met; do not promote or claim
quality parity. Small-contract exactness did not generalize to this real model.
Cause remains unestablished. Both flags stay off by default; no additional
CUDA test or relaxed tolerance was authorized from this result.

Artifacts: `E:\LTX-2\outputs\ddo_modular_video_20261007_121841_203` (baseline)
and `E:\LTX-2\outputs\ddo_modular_video_20261007_122144_968` (candidate).
Full conditions and comparison are in the sibling host's
`model_execution_optimizations.md`. Image routes and allocator defaults remain
unchanged; this is not an image regression gate or group-offloading comparison.

## Same-input real-video fusion diagnosis — 2026-10-07

Approved one-step diagnostic retained the preceding A/B's model, BF16,
1280x704, 8 seconds / 24 fps / 193 frames, seed 43, chunks, budgets and DLL.
It compared the first six RMS/modulation calls and four rotary pairs on the
same actual inputs, always returning the original eager results downstream.
Completed with exit code 0; all 14 comparisons finite, normal cleanup, zero
selected native failure/quarantine counters, ring fallback bytes and live source
tickets. Protected DLL/hash unchanged. This is not a performance benchmark.

Video RMS/modulation calls 1/3/5 differed in respectively 181/413/237 of
90112000 elements, with maximum absolute differences
`0.0078125 / 0.03125 / 0.00390625` and relative RMS differences
`0.000004788072 / 0.000011778770 / 0.000007793214`.
All three audio RMS/modulation calls and all eight sampled split-rotary outputs
were exact. The result localizes an operation-level numerical difference to
video RMS/modulation in this sample; it does not prove the arithmetic substage
responsible or explain all accumulated final-latent divergence. It is not
all-layer rotary acceptance. Both fusion defaults remain off.

Artifact:
`E:\LTX-2\outputs\fusion_same_input_20261007_130030_567\same_input_report.json`;
normal runner metrics and eager-return latents are preserved in its `run`
subdirectory. Host report: `model_execution_optimizations.md`. No additional
CUDA sequence or numerical-tolerance relaxation followed.

## Conservative modulation correction — 2026-10-07, CPU only

The opt-in callback now preserves the original norm forward and its installed
chunking, then fuses only modulation. Separate modulation dtype rounding is
retained; paired rotary and all protected runtime defaults/budgets/DLLs are
unchanged. Submission snapshots identify the original norm backend explicitly.
No historical artifact was rewritten; earlier fused-reduction timing and peak
results do not apply to this revision.

CPU-only validation: **72 passed in 6.24 s**, including eight new mocked-launch
handoff cases for FP32/BF16, affine/bias and chunked/unmodified norm forwards.
These are controller/regression checks, not kernel numerical acceptance.
The proposed small CUDA contracts followed by the same-input one-step gate
are consolidated in the sibling host's `validate_modulation_correction.ps1`.
No CUDA execution, real-model parity or speedup is claimed for this correction;
the sequence still requires user approval.

## Corrected modulation operation gates — 2026-10-07

User-approved consolidated sequence completed with exit code 0: all 36 small
CUDA comparisons were exact, followed by all 14 finite and bit-exact first-block
real-video same-input comparisons. The three video RMS/modulation calls that
previously differed now had zero differences. PyTorch `2.12.1+cu132` / CUDA
`13.2` / pluggable allocator; original norm forward plus Triton modulation,
with unchanged paired rotary, protected DLL/hash, budgets and defaults.

Encoder and transformer completed source-ticket collection with zero remaining
tickets, selected native failure/quarantine counters and ring fallback bytes.
Artifact:
`E:\LTX-2\outputs\fusion_same_input_20261007_133053_249\same_input_report.json`;
normal metrics and eager-return latent output remain in `run/`. CPU-only
comparison against the preceding diagnostic's saved video/audio latents was
bit-exact and finite, without CUDA initialization. Both diagnostics returned
original eager outputs downstream, so this is control consistency and sampled
operation validation, not full candidate-output parity or a speedup result.

No additional CUDA run or default promotion followed. Matched two-step final
latent/performance acceptance requires separate approval. The rejected earlier
fused-reduction candidate's 16.1% timing reduction does not apply to this revision.

## Corrected end-to-end modulation/rotary A/B — 2026-10-07

Approved matched fresh-process pair: BF16, 1280x704, eight seconds at 24 fps
(193 frames), two steps, seed 43, real prompt/image, no input audio conditioning,
joint audio generation and latent output. Same DLL, budgets and chunks as the
preceding A/B. Only two fusion flags and output directory differed in recorded
requested controls. Actual corrected candidate outputs fed the model.

| Measurement | Baseline | Corrected candidate |
| --- | ---: | ---: |
| Run total, seconds | 131.1571 | 124.7787 |
| Denoise CUDA phase, seconds | 91.723539 | 85.009664 |
| Second step CUDA interval, seconds | 44.555465 | 41.541945 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.504395 |
| Sampled absolute GPU peak, GiB | 7.9995 | 7.5312 |
| Sampled process RSS peak, GiB | 28.561218 | 28.452538 |

**Selected gate passed:** second-step reduction **6.7635%** (target at least 5%);
run total reduction **4.8632%**. CPU comparison found finite, bit-exact final
video/audio latents, with zero differing elements and zero maximum/RMS errors;
recorded generation metadata matched. No CUDA initialization in this comparison.
Both completed safely with zero selected native failure/quarantine/ring-fallback
counters and remaining source tickets, completed CUDA timing/timelines and
released events. Candidate submitted 576 composite norm/modulation and 384
paired rotary operations. DLL/hash, budgets and default routes unchanged.

One ordered two-step pair, not statistical proof, eight-step acceptance or
decoded-video quality/timing comparison. Flags remain off by default; no
further CUDA run or promotion followed. Artifact root:
`E:\LTX-2\outputs\corrected_modulation_ab_20261007_134003_382`.
Host details: `model_execution_optimizations.md`.

Memory interpretation: `global_peak_vram_gb` is a baseline-relative delta,
not absolute usage. Baseline here was `1.02685546875 GiB`; sampled absolute
phase peaks come from `peak_vram_absolute_gb`. The preceding October 7 fusion
table's label is corrected and its absolute row added; raw artifacts and
recorded delta values remain unchanged.

## Corrected eight-step video candidate closure — 2026-10-07

Approved single candidate: BF16, 1280x704, eight seconds at 24 fps (193 frames),
eight steps, seed 43, real prompt/image, no input audio conditioning, joint
audio generation and latent output without decode. Only step count and artifact
directory differ from the accepted corrected two-step candidate. Same protected
DLL, budgets, chunks and explicit modulation/rotary callbacks; defaults unchanged.

| Measurement | Corrected candidate |
| --- | ---: |
| Run total, seconds | 375.6927 |
| Denoise CUDA phase, seconds | 335.218125 |
| First step CUDA interval, seconds | 43.324703 |
| Mean steps 2-8 CUDA interval, seconds | 41.690544 |
| Steps 2-8 minimum / maximum, seconds | 40.442004 / 43.348531 |
| Sampled baseline-relative GPU peak, GiB | 6.655048 |
| Sampled absolute GPU peak, GiB | 7.6819 |
| Sampled process RSS peak, GiB | 28.466244 |

Selected closure passed with exit code 0 and finite saved video/audio latents
checked on CPU without CUDA initialization. Eight complete CUDA step intervals;
native timeline 2412 samples, zero dropped/failure/pending samples, released
events. Selected native failure/quarantine/ring-fallback counters and remaining
source tickets were zero. Transformer materializations 64, temporary fallbacks
zero; composite norm/modulation submissions 2304 and rotary-pair submissions
1536. Original norm forward and protected DLL/hash remained unchanged.

This is bounded safety and observed timing, not a matched eight-step speedup or
final numerical-parity comparison. Bit-exact final-output parity and measured
6.7635% second-step reduction belong to the accepted two-step A/B only. No
decoded-video timing/quality, general long-run stability, image-route validation
or default promotion is claimed. No additional CUDA run followed.

Artifact root:
`E:\LTX-2\outputs\corrected_modulation_8steps_20261007_140009_668\candidate`.
Host details: `model_execution_optimizations.md`. Memory deltas are relative
to the initial 1.02685546875 GiB baseline; absolute peaks are reported separately.

## Regional compilation integration gate — 2026-10-07

Approved fresh-process two-step control versus regional Inductor compilation,
using the same eight-second BF16 1280x704/24-fps workload, seed 43, prompt/image,
budgets, chunks, SDPA selection and protected allocator. Both optional execution
kernel callbacks were off. No external kernel was loaded.

Control completed and passed native safety/ticket collection and finite-latent
checks: total 128.0026 s, CUDA denoise phase 87.234930 s, step intervals
43.363313 / 43.832199 s. Sampled baseline-relative GPU peak 6.972656 GiB,
absolute GPU peak 7.9995 GiB, process RSS peak 28.673908 GiB.

Candidate **rejected at tracing** before completing a step: Dynamo attempted
to trace the native component forward's host materialization preamble and
encountered an unsupported ctypes `_FuncPtr.__call__` in the arena-info query.
48 repeated blocks were configured, but no completed compiled denoise or final
candidate output exists. Candidate timing/parity/completed cleanup counters are
unavailable; do not interpret configuration success as execution acceptance.
This is a native-hook/compilation boundary issue, not proof that Windows cannot
run Inductor. No performance comparison or speedup is claimed for compilation.

Artifacts: `E:\LTX-2\outputs\regional_compile_ab_20261007_152225_714`.
Stopped on failure without retry, tracing relaxation, hook removal or budget
change. Protected allocator hash and accepted independent candidate unchanged.

## Compute-only compilation/native small gate — 2026-10-07

After the module-call tracing rejection, the diagnostic host compiles only
declared compute forwards before native attachment. Existing native wrappers
remain outside that graph; no allocator code/binary or runtime defaults change.
This is an adapted regional experiment, not the unmodified Diffusers
`compile_repeated_blocks` module-call boundary.

Approved small gate passed: two generic BF16 units, width 128, batch 2, seed 43,
native auxiliary ownership, 0.07 GiB native ceiling, 2 MiB staging, 64 MiB source/
registration limits. Four finite, bit-exact outputs against eager CUDA (two
default-stream and two non-default-stream invocations), zero selected native
failure/quarantine/ring-fallback counters, zero remaining source tickets and
completed native retirement/compute-forward restoration. Source read bytes
remained 132776, with no reread after the first invocation.

PyTorch `2.12.1+cu132`, CUDA 13.2; Inductor fullgraph/static, precision-cast
emulation enabled, CUDA Graphs disabled. Protected DLL/hash unchanged.
Artifact: `E:\LTX-2\outputs\regional_vmm_gate_20261007_154748_190\report.json`.
This is bounded integration acceptance, not LTX parity/performance or general
model support. A corrected real-model A/B remains separately approval-gated.

## Compute-only regional LTX A/B — 2026-10-07, parity rejected

Approved corrected two-step pair with the same eight-second BF16 workload,
budgets, chunks and protected allocator. Only compilation flag/output directory
differed. Native materialization stayed outside the compiled compute forward;
the earlier ctypes tracing failure did not recur. No extra kernels were enabled.

| Measurement | Eager control | Compute-only Inductor |
| --- | ---: | ---: |
| Run total, seconds | 142.3055 | 160.0945 |
| CUDA denoise phase, seconds | 101.783156 | 120.570742 |
| First CUDA step interval, seconds | 54.160285 | 81.907914 |
| Second CUDA step interval, seconds | 47.598875 | 38.637332 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.515648 |
| Sampled absolute GPU peak, GiB | 7.9995 | 7.5425 |
| Sampled process RSS peak, GiB | 28.591766 | 28.966026 |

Observed second-step reduction 18.8272%, with larger total due to the initial
interval including compilation. One ordered pair with a slower control than
previous rounds; not statistical proof or a direct ranking against those rounds.
Both completed with finite outputs and passed selected native safety/ticket/
timing checks, but final **bit-exact parity failed** and the sequence exited 1.
CPU-only comparison: video max error 0.964721680, RMS 0.031353850, relative RMS
3.4264475%; audio max 0.093261838, RMS 0.013786504, relative RMS 2.2103660%.
No CUDA initialization during comparison. Numerical cause is not localized yet.

Artifact root: `E:\LTX-2\outputs\regional_compile_ab_20261007_160559_683`.
No tolerance relaxation, retry or promotion; protected DLL/budgets/defaults
unchanged. Accepted independent kernels keep their earlier bit-exact evidence.

## Isolated compiled RMSNorm diagnostic — 2026-10-07

Approved bounded synthetic comparison, no LTX checkpoint/native allocator or
external kernel: BF16 non-affine RMSNorm, width 4096, 1024 tokens, seed 43.
PyTorch 2.12.1+cu132 / CUDA 13.2, fullgraph/static Inductor, precision-cast
emulation enabled and CUDA Graphs disabled. Both outputs finite, but 34 of
4194304 elements differed; maximum absolute error 0.015625, RMS error
0.000019602589408010927. Exit 1 at the first mismatch; remaining cases not run.

This localizes a reproducible isolated numerical difference to compiled
RMSNorm, not the complete real-model divergence. No performance measurement,
parity relaxation, allocator change, rerun or default promotion is implied.
Artifact: `E:\LTX-2\outputs\compiled_rms_gate_01\report.json`.

## Isolated opaque eager RMSNorm CUDA gate — 2026-10-07

Approved correction diagnostic passed with exit 0: four BF16 synthetic cases,
4096/2048 channels, non-affine/affine, 1024 tokens, seed 43. All outputs finite
and bit-exact, zero differing elements and maximum/RMS errors. Fullgraph/static
Inductor calls the installed eager RMSNorm behind an opaque operator;
precision-cast emulation on and CUDA Graphs off, PyTorch 2.12.1+cu132 / CUDA 13.2.
No LTX checkpoint, native allocator or external kernel involved.

This accepts the isolated numerical boundary, not full-model parity,
native integration or performance. Defaults and protected allocator unchanged.
Artifact: `E:\LTX-2\outputs\compiled_rms_opaque_gate_01\report.json`.

## Regional compilation with opaque eager RMSNorm — 2026-10-07, parity rejected

Approved matched two-step/eight-second BF16 pair, original budgets and protected
allocator; candidate 48 compute blocks and 384 opaque installed eager RMSNorm
boundaries, chunk size 1024, extra kernels off. Both generations completed with
finite outputs and passed selected native safety/event/ticket checks, but the
sequence exited 1 at the zero-tolerance final video comparison.

| Measurement | Eager | Compiled with opaque RMSNorm |
| --- | ---: | ---: |
| Total runner, seconds | 220.9844 | 156.4098 |
| CUDA phase, seconds | 179.687781 | 115.338961 |
| First CUDA step, seconds | 130.394578 | 67.002844 |
| Second CUDA step, seconds | 49.266117 | 48.312063 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.811882 |
| Sampled process RSS peak, GiB | 28.824039 | 28.898205 |

Only approximately 1.94% second-step reduction in this ordered pair. The unusually
slow eager first interval prevents interpreting the total difference as a
reliable compilation speedup. CPU-only final comparison: video maximum error
0.906602740, RMS 0.029376684, relative RMS 3.2103767%; audio maximum error
0.093359470, RMS 0.014644632, relative RMS 2.3479481%. Both tensors finite.
The isolated RMSNorm fix is insufficient for full-model parity. No repeat,
threshold relaxation or default promotion; accepted independent kernels intact.

Artifact: `E:\LTX-2\outputs\regional_compile_eager_rms_ab_20261007_171446_372`.

## Bounded compiled-operation real-input diagnosis — 2026-10-07

Approved first selected block audit, original workload geometry/native budgets:
six RMS/modulation calls plus four rotary pairs, 14 compared outputs, all finite
and bit-exact on identical real inputs. Operation-level fullgraph/static
Inductor, opaque installed eager RMSNorm, 1024 chunks, precision-cast emulation,
CUDA Graphs off; independent Triton kernels not executed. Protected DLL unchanged.

Expected cutoff after selected coverage, no completed model generation.
Diagnostic reports `diagnosed`; partial runner reports `failed` with the expected
`AuditStop`. No full transformer safety/retirement/timing gate is implied by the
wrapper exit 0. Full-block numerical divergence remains unlocalized; selected
independent operations do not prove equivalent cross-operation graph lowering.
No rerun or default promotion.

Artifact: `E:\LTX-2\outputs\compiled_ops_first_block_20261007_173501_968\same_input_report.json`.

Coverage clarification after read-only source/generated-code review: candidate
C's opaque boundaries cover Diffusers RMSNorm only. Q/K modules are
`torch.nn.RMSNorm` and remained compiler-visible. The selected 14-output audit
starts rotary comparisons after eager Q/K normalization; it does not validate
that normalization. This uncovered path is not yet a proven numerical cause.
No additional CUDA test or accepted-performance/default change followed.

## Isolated PyTorch Q/K RMSNorm diagnosis — 2026-10-07, parity rejected

Approved bounded test of installed `torch.nn.RMSNorm`: BF16, non-affine,
4096 channels, 1024 tokens, seed 43, fullgraph/static Inductor with precision
casts preserved and CUDA Graphs off. Both outputs finite, but 34 of 4194304
elements differed; maximum absolute error 0.015625, RMS error
0.000019602589408010927. Exit 1 at first mismatch; remaining cases not run.
PyTorch 2.12.1+cu132 / CUDA 13.2, no LTX/native allocator/external kernel.

This demonstrates numerical drift in the normalization type left outside the
candidate's Diffusers-only boundary; it does not establish the whole model's
cause. Preserve each backend's original norm semantics in any future correction.
No retry, default promotion or accepted-runtime modification.
Artifact: `E:\LTX-2\outputs\compiled_torch_rms_gate_01\report.json`.

## Per-type opaque RMSNorm regional compile checkpoint — 2026-10-07, accepted two-step pair

Approved consolidated sequence passed: four isolated opaque PyTorch RMSNorm
cases finite/bit-exact, then the original eight-second/two-step BF16 real-model
eager-versus-compiled pair. Final video and audio latents finite and bit-exact;
selected native timing/event/failure/source-collection checks passed for both.
48 compiled compute blocks, 384 Diffusers plus 576 PyTorch opaque original
norms, no new Q/K chunking; optional fused kernels off. Fullgraph/static Inductor,
precision-cast emulation, no CUDA Graphs; unchanged budgets and protected DLL.

| Measurement | Eager control | Compiled with both RMSNorm types opaque |
| --- | ---: | ---: |
| Total runner, seconds | 146.5828 | 143.0399 |
| CUDA phase, seconds | 104.887734 | 101.953273 |
| First CUDA step, seconds | 55.217703 | 59.663430 |
| Second CUDA step, seconds | 49.631367 | 42.266355 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.402809 |
| Sampled process RSS peak, GiB | 29.206753 | 28.867348 |

Approximately 14.84% second-step reduction and 3.5429 s lower total in this
ordered pair. First step includes lazy compilation, cache effects not isolated.
No statistical/general-support claim or direct ranking against independent
kernels from another pair. No eight-step compiled run, decode validation,
extra benchmark or default promotion. Earlier numerical rejections preserved.

Artifacts: `E:\LTX-2\outputs\compiled_torch_rms_opaque_20261007_183143_283\report.json`
and `E:\LTX-2\outputs\regional_compile_both_rms_ab_20261007_183149_787`.

## Single compiled eight-step candidate — 2026-10-07, completion/safety accepted

Approved unchanged eight-second BF16 workload, 1280x704/24 fps, seed 43,
real prompt/image, no input audio conditioning, latent output. 48 compiled
blocks, 384 Diffusers and 576 PyTorch original opaque norms; extra kernels off.
Original budgets/native DLL preserved, no retuning or automatic run extension.

| Measurement | Compiled candidate |
| --- | ---: |
| Total runner, seconds | 395.0373 |
| CUDA phase, seconds | 353.873750 |
| First CUDA step, seconds | 46.933320 |
| Steps 2–8 mean, seconds | 43.845735 |
| Steps 2–8 minimum / maximum, seconds | 42.366328 / 44.580965 |
| Sampled baseline-relative GPU peak, GiB | 6.413925 |
| Sampled process RSS peak, GiB | 28.492538 |

Eight complete CUDA intervals; 2412 timeline samples, zero dropped/failure/
pending samples, events released. Selected native fault/admission/pool/transport/
pressure checks passed, encoder/transformer completion collections zero tickets.
64 materializations, zero temporary fallbacks. Final video/audio finite in
CPU-only checks. No matched eight-step parity, decode or general-support claim.
First interval includes lazy compilation; cache/compile costs not isolated.

Historical independent-kernel eight-step observation was 41.690544 s/step and
375.6927 s total in a separate run. Do not interpret that nonmatched comparison
as proven regression or compiler advantage; no accepted default replacement.
Artifact: `E:\LTX-2\outputs\regional_compile_both_rms_8steps_20261007_191352_468\candidate`.

### External Q/K kernel — corrected small-operation oracle gate (October 7)

The historical small gate used Diffusers RMSNorm, but installed LTX attention
Q/K uses `torch.nn.RMSNorm`. A separately approved gate using that original
PyTorch operation passed all 12 BF16 comparisons bit-exactly: zero differing
elements, maximum and RMS error zero; finite outputs and no CUDA execution
error reported. Widths 2048/4096, token pairs 17/17 and 17/5, seed 43,
non-default stream and `rtol=atol=1/128` unchanged. Both selected binary hashes
stayed unchanged. No third-party CUDA math or DDO runtime/default changed.

Command: `& E:\LTX-2\run_external_qk_rope_gate.ps1 -TorchRmsNorm`.
Artifact: `E:\LTX-2\outputs\external_qk_rope_torch_rms_gate_20261007_204831_328\report.json`.
The earlier Diffusers-oracle failure remains preserved, not evidence of failure
against the actual Q/K operation. This is not a model performance result or
final-latent/decoded-media acceptance. No model benchmark followed.

### External Q/K compiled model A/B — final parity rejection (October 7)

The separately approved sequence compared regional compilation with both eager
norm boundaries, without/with the pinned attributed external Q/K operation.
Selected 1280x704, eight-second/24-fps, two-step, seed-43 BF16 native workload;
no input audio conditioning, joint audio generation, latent output/no decode.
Budgets/chunks and both pinned binaries unchanged. No eager/independent-kernel
control in this pair.

| Observation | Compiled control | Compiled + external Q/K |
| --- | ---: | ---: |
| Total runner seconds | 123.6231 | 130.7741 |
| CUDA phase seconds | 82.733921875 | 91.699000000 |
| First CUDA step seconds | 43.068718750 | 52.21701953125 |
| Second CUDA step seconds | 39.64747265625 | 39.46130859375 |
| Sampled baseline-relative GPU peak GiB | 6.322315216 | 6.038021088 |
| Sampled process RSS peak GiB | 28.502681732 | 28.856861115 |

Both runs passed selected native safety/completion/collection checks: 606
timeline samples each, zero drops/failures/pending, released events, zero final
encoder/transformer source tickets and selected failure counters. Each recorded
16 materializations, zero temporary fallbacks. External callbacks configured on
288 attention modules and retired; installed opaque-norm counts are not calls.
The A/B exited 1 on final parity, despite individual runner completion.

CPU-only comparison found finite but non-bit-exact final video/audio, relative
RMS differences 3.038259483% / 2.435441215%. The synthetic kernel gate passing
does not establish model parity or identify the sole cause of this divergence.
Second-step difference only 0.1861640625 s (about 0.47%); one ordered pair does
not demonstrate a reliable advantage. Candidate total 7.1510 s longer; lazy
compilation/cache costs not isolated. No default promotion or CUDA retry.
Command: `& E:\LTX-2\run_corrected_modulation_ab.ps1 -ExternalQkRopeAb`.
Artifact: `E:\LTX-2\outputs\regional_compile_external_qk_ab_20261007_211322_386`.

Comparison coverage is consolidated in the sibling host's
`E:\LTX-2\execution_comparison_summary.md`: measured pairs, numerical acceptance,
published external-reference results kept separate, and unimplemented/deferred
combinations. A new independent Triton fused Q/K norm/RoPE route without/with
regional compilation now has an optional prototype, not CUDA-validated or benchmarked. Its same-input
actual-model operation parity must precede any timing matrix. No CUDA run,
budget/default change or author message is authorized by this documentation update.

Independent prototype preparation: host CPU suite passed 105 tests in 11.33 s;
the two-stage launcher command preview passed without CUDA. Existing allocator,
kernels and defaults are unchanged. Pending separate CUDA approval, the gate
requires 12 bit-exact synthetic comparisons followed by bounded first-block
actual-input comparisons (at most 96 sampled token rows per component, four Q/K
calls across two candidates). Stop on the first failed stage or mismatching pair;
no timing matrix or full-generation acceptance is implied. Host diagnostic:
`E:\LTX-2\run_independent_qk_gate.ps1`.

The approved independent Q/K gate stopped at Triton compilation before any
comparison or model load (exit 1). An inferred reshape dimension needed an
explicit `tl.constexpr` annotation. Preserved report:
`E:\LTX-2\outputs\independent_qk_gate_20261007_223212_835\small\report.json`.
Both protected native/external binary hashes are unchanged. The annotation was
corrected and the CPU suite passed again (105 tests, 10.16 s); corrected-source
CUDA execution remains pending separate approval. No numerical or performance
acceptance is established by this attempt.

The separately approved corrected repeat passed all 12 synthetic CUDA Q/K
comparisons bit-exact. Actual-input strict parity then failed at the first video
self-attention pair: 96 sampled rows / 393,216 BF16 elements per output. External
Q/K differed in 1/2 elements (max absolute 0.0001220703125 / 0.0078125);
independent Triton Q/K differed in 1/3 elements (max absolute 0.0001220703125
for both). All outputs finite; no regional compilation in this diagnosis.
Sequence exited 1 and stopped before later attention calls or a completed step.
Both protected binary hashes remained unchanged. Artifact:
`E:\LTX-2\outputs\independent_qk_gate_20261007_223423_023`.
No timing benchmark or default promotion. These small operation differences do
not establish the cause of earlier final-latent drift or a platform limitation;
sample/full-input eager normalization and separate norm/RoPE arithmetic remain
unisolated. Full generation/retirement is not accepted by a diagnostic cutoff.

A one-pair arithmetic-isolation diagnostic is prepared in the host launcher
(`-ArithmeticIsolation`), without repeating synthetic gates or benchmarks.
It separates original sampled/full normalization and RoPE from candidate
identity-RoPE normalization and fused-rotation differences. Sixteen scalar
records; stop after the first pair; successful collection is not parity
acceptance. CPU suite passed 106 tests in 11.04 s; command preview passed.
CUDA diagnosis remains pending separate approval; protected defaults/binaries
are unchanged.

Approved arithmetic-isolation result (first actual video self-attention pair):
all 16 records finite; original sampled/full RMSNorm and RoPE comparisons exact.
Candidate normalization via identity RoPE differed in external Q/K 2/2 elements,
independent Triton Q/K 1/2 elements, per 393,216-element output. Independent norm
max absolute difference 0.0001220703125 for both; external 0.00048828125 /
0.00390625. Fused rotation versus original RoPE of the same candidate norm was
bit-exact for both candidates. This isolates observed first-pair differences to
normalization, not sampling or rotation in this case, without proving the cause
of prior full-generation drift or a platform defect. Sixteen records collected,
exit 0 means diagnosis complete, not parity accepted. Protected binaries unchanged;
no benchmark/default change. Artifact:
`E:\LTX-2\outputs\independent_qk_gate_20261007_224239_203\actual_inputs\audit.json`.

Independent normalization correction prepared: explicit four-member FP32 FMA
accumulation and fixed 128-logical-lane reduction order, four warps for both
widths. Final BF16 boundary, RoPE, allocator and defaults unchanged. CPU suite
108 passed in 11.53 s; subsequent launcher contracts 15 passed in 7.16 s.
Corrected-source CUDA compilation/parity remains unvalidated. Pending approval,
`run_independent_qk_gate.ps1 -ArithmeticIsolation -ValidateSynthetic` requires
12 bit-exact synthetic comparisons before the bounded first-pair arithmetic
diagnosis. No timing or final-generation acceptance is implied.

Approved grouped-reduction validation stopped during Triton compilation before
comparisons or model load: a loop-local constexpr could not be redefined by
static unrolling. Artifact:
`E:\LTX-2\outputs\independent_qk_gate_20261007_225458_012\small\report.json`.
Removed the constexpr declaration without changing the arithmetic schedule;
CPU suite 108 passed in 10.67 s. Protected binary hashes unchanged. Corrected
CUDA compilation/parity remains pending separate approval; no benchmark.

Approved corrected repeat:
`E:\LTX-2\outputs\independent_qk_gate_20261007_225729_339`.
All 12 synthetic comparisons bit-exact. On the first actual video self-attention
pair (96 sampled rows, width 4096), independent normalization, rotation and
combined Q/K outputs now all match exactly, with zero maximum/RMS error.
Original sampled/full controls exact; unchanged external candidate reproduced
its earlier discrepancies. Independent source SHA256:
`7a6a4ad43121b10ecd1896e99ea5ed2fc590dcec9a6129d0fc64fa40422c6f18`.
Both protected binaries unchanged. Sixteen finite diagnostic records collected;
exit 0 is bounded diagnosis completion, not model/performance acceptance.
Real audio/cross-attention, later blocks, full-shape and compiled-model parity
remain unvalidated. No benchmark, tolerance relaxation or default promotion.

The bounded four-call independent-only gate is prepared in the sibling host:
`run_independent_qk_gate.ps1 -IndependentOnly`. Eight finite bit-exact sampled
Q/K comparisons are required across first-block video/audio self-attention and
both audio/video cross directions; stop on the first mismatch or fourth call.
Text-cross attention without RoPE is outside this callback. Original outputs
continue through earlier calls; no full-step/model or performance acceptance.
Unchanged kernel, budgets, binaries/defaults; no repeated synthetic gate.
CPU suite 109 passed in 12.14 s; command preview passed. CUDA execution awaits
separate approval.

Approved independent-only first-block Q/K gate completed (exit 0,
`bounded_operations_passed`): four calls / eight finite bit-exact outputs.
Video self-attention full token counts 22000/22000, width 4096; audio self
201/201, width 2048; audio-to-video 22000/201 and video-to-audio 201/22000,
width 2048. Shared tables on self-attention; distinct tables on cross directions.
Each comparison samples 96 rows, zero maximum/RMS error. Unchanged independent
source hash and both protected binaries. Artifact:
`E:\LTX-2\outputs\independent_qk_gate_20261007_230632_822\actual_inputs\audit.json`.
Expected fourth-call cutoff; original outputs propagate in earlier calls.
No complete step, candidate propagation, later-block, compiled-model, retirement
or performance acceptance. No default promotion or benchmark.

Optional independent Q/K runner integration prepared in the sibling host;
default off, native route only, no allocator/residency/default changes.
CPU suite passed 116 tests in 13.37 s. Proposed matched two-step eager-control
versus fused-Q/K candidate command:
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -IndependentQkRopeAb`.
Preview passed; selected workload/budgets unchanged. Require selected native
safety/completion/retirement gates, preserved source/binary hashes, finite and
bit-exact final video/audio latents. No CUDA execution yet; compiled comparison
and eight-step cases are not automatically authorized by preparing this pair.

## Independent grouped fused Q/K: approved two-step model pair

Artifact: `E:\LTX-2\outputs\independent_fused_qk_ab_20261007_233301_602`.
Command: `& E:\LTX-2\run_corrected_modulation_ab.ps1 -IndependentQkRopeAb`.
Selected native staged BF16 1280x704, eight seconds/24fps, two steps, seed43,
no input audio, latent-only; existing budgets/chunks unchanged; no compilation.
Both completed with finite and bit-exact final video/audio; A/B exit 0.

| Metric | Control | Independent fused Q/K |
| --- | ---: | ---: |
| Total runner seconds | 152.3389 | 123.8883 |
| CUDA phase seconds | 113.2387421875 | 85.1459453125 |
| First CUDA step seconds | 46.3002734375 | 43.67172265625 |
| Second CUDA step seconds | 66.9156484375 | 41.4525546875 |
| Sampled baseline-relative GPU peak GiB | 6.97265625 | 6.621509552001953 |
| Sampled process RSS peak GiB | 28.825973510742188 | 28.496761322021484 |

Both passed selected native safety/retirement: 606 timeline samples each,
zero drops/failures/pending, events released; final component source tickets
zero and selected failure counters zero. Each 16 materializations, zero temporary
fallbacks. Candidate configured on 288 attentions, denoising complete, callback
retired; accepted source and allocator hashes unchanged. Control second interval
unusually slow (cause unisolated): no reliable speedup percentage or ranking
against earlier different pairs. Selected two-step numerical/integration acceptance
only; no decoded/eight-step/general-model/compiled-candidate acceptance or default
promotion. Further compiled comparison requires separate approval.

## Independent grouped fused Q/K plus regional compilation

Approved command:
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -IndependentQkRopeAb -RegionalCompileAb`.
Artifact: `E:\LTX-2\outputs\regional_compile_independent_fused_qk_ab_20261007_234009_127`.
Same selected two-step BF16 native workload/budgets; both compiled with original
norms opaque, Q/K candidate only difference. Both completed with finite,
bit-exact final video/audio; sequence exit 0. Selected native source/timeline/
admission/transport/retirement checks passed. Each 606 samples, no failed/dropped/
pending, events released; final source tickets zero; 16 materializations and no
temporary fallback. Candidate 288 callbacks configured, completed and retired;
accepted source/native hashes unchanged.

| Metric | Compiled control | Compiled + independent Q/K |
| --- | ---: | ---: |
| Total runner seconds | 122.1532 | 129.0745 |
| CUDA phase seconds | 82.5077109375 | 89.5672421875 |
| First CUDA step seconds | 43.276421875 | 49.12148828125 |
| Second CUDA step seconds | 39.21261328125 | 40.42636328125 |
| Sampled baseline-relative GPU peak GiB | 6.346477508544922 | 6.114017486572266 |
| Sampled process RSS peak GiB | 28.502220153808594 | 28.549884796142578 |

No incremental gain observed: second interval 1.21375 s / about 3.10% longer,
total 6.9213 s longer. One ordered pair is not a general regression claim;
compilation/cache/run variation not isolated. Bounded eager/compiled independent
comparison now recorded with accepted selected parity, without reliable added
kernel speedup. Existing defaults protected; no automatic retry, tuning matrix,
eight-step model/general-platform acceptance or author message.

## Previous Triton kernels with regional compilation: preparation only

An isolated optional compute route now wraps the existing modulation and paired
RoPE kernels as opaque operations. It keeps the original norm forward and the
previous kernels' arithmetic/launch geometry, without changing residency,
budgets, allocator source/binary, or existing runtime defaults. The independent
grouped Q/K normalization candidate is not part of this comparison.

131 targeted CPU capture/routing/launcher regression contracts passed in 16.04
seconds. The protected allocator hash remained unchanged. CUDA correctness and
performance are pending separate confirmation. The prepared host command is
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompiledTritonAb`: two small CUDA
compute cases first, then one matched compiled-control/compiled-plus-previous-
Triton pair at the selected 1280x704 / 8-second / 24-fps / two-step workload.
Stop at a failed gate; require finite bit-exact final video/audio and existing
native safety checks. No automatic eight-step run, default promotion or new
full-normalization fusion. Without a clear incremental benefit, close this
branch rather than extending a tuning matrix.

## Previous Triton kernels with regional compilation: approved bounded result

Approved command: `& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompiledTritonAb`.
Artifact: `E:\LTX-2\outputs\regional_compile_triton_execution_ab_20261008_000702_294`.
Two small compute CUDA contracts passed in 8.06 seconds: selected original,
opaque and compiled outputs finite and bit-exact. The subsequent native real-
model control/candidate completed with finite, bit-exact final video/audio.
Sequence exit 0; selected native gates passed. Each recorded 606 timeline
samples, zero failures/drops/pending, events released, source tickets zero,
16 materializations and no temporary fallback. Candidate 48 block / 288
attention callback bindings configured, completed and retired; these are not
invocation counts. Original norm boundaries, protected allocator and previous
kernel source preserved. Wrapper hash:
`39b6470bc0aa1340a8185f1ed31f2e13626d9e56945b65cd4923c33e2230098a`.

| Metric | Compiled control | Compiled + previous Triton |
| --- | ---: | ---: |
| Total runner seconds | 126.6132 | 132.9614 |
| CUDA phase seconds | 85.1692109375 | 94.8239921875 |
| First CUDA interval seconds | 44.9488203125 | 53.854515625 |
| Second CUDA interval seconds | 40.19797265625 | 40.9497734375 |
| Sampled baseline-relative GPU peak GiB | 6.332401275634766 | 6.185237884521484 |
| Sampled process RSS peak GiB | 28.494911193847656 | 28.857677459716797 |

No gain observed: second interval about 1.87% slower, total 6.3482 seconds
longer. One ordered pair is not a general slowdown claim, and first intervals
do not isolate compilation/cache costs. Numerical/native integration accepted
only for this selected two-step latent workload; no decoded/eight-step/general-
model acceptance or default promotion. Close the bounded combination branch
without repeat or further fusion tuning. Next agreed opportunities are compact
timesteps and in-place residuals; new CUDA execution requires confirmation.

## Compact timestep/residual combination: preparation only

Prepared host command (CUDA approval pending):
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualAb`.
Both sides use regional compilation and opaque original norm boundaries;
candidate alone enables the already-implemented compact timesteps and in-place
residual additions. No compute kernel, budget, allocator binary or default is
changed. One small BF16 CUDA contract must pass exact output/input/projection-row
checks before one matched native real-model pair at the selected 1280x704,
8-second, 24-fps, two-step workload. Retain native checks and bit-exact final
video/audio; require executed compact layout rather than only requested flags.

CPU preparation passed 137 contracts and 11 subtests in 20.23 seconds. These
are not CUDA numerical or performance acceptance. Combined benefit, if any,
must be assessed within this pair without claiming individual attribution.
No automatic repeated/eight-step runs, tolerance relaxation, budget changes,
or promotion; existing image, linear/NVMe and native accepted routes preserved.

## Compact timestep/residual combination: approved bounded result

Approved command: `& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualAb`.
Artifact: `E:\LTX-2\outputs\regional_compile_compact_residual_ab_20261008_001757_883`.
Small BF16 CUDA compute gate passed in 21.36 seconds with bit-exact outputs,
unchanged caller tensors/checkpoint keys, projection rows 16 -> 4. TF32 remained
disabled despite the compiler warning. Both native real-model runs completed;
finite bit-exact final video/audio, sequence exit 0. Selected native gates
passed: each 606 samples, zero failures/drops/pending, events released, final
source tickets zero, 16 materializations and no temporary fallback. Candidate
reports compact and residual controls executed with timestep repeat 880; control
requested both false (missing execution fields remain unavailable). Protected
allocator, budgets, original opaque norms and earlier routes unchanged.

| Metric | Compiled control | Compiled + compact/residual |
| --- | ---: | ---: |
| Total runner seconds | 121.1271 | 129.5201 |
| CUDA phase seconds | 83.921671875 | 91.6996796875 |
| First CUDA interval seconds | 43.69741796875 | 51.6938828125 |
| Second CUDA interval seconds | 40.2056171875 | 39.99374609375 |
| Sampled global baseline-relative GPU peak GiB | 5.560756683349609 | 3.935546875 |
| Sampled denoise baseline-relative GPU peak GiB | 5.5608 | 3.8184 |
| Sampled denoise absolute driver GPU peak GiB | 6.5876 | 4.8452 |
| Sampled process RSS peak GiB | 28.495643615722656 | 29.0654296875 |

Only about 0.53% second-interval reduction; total 8.3930 seconds longer. No
demonstrated speedup; compilation/cache/run variability not isolated. Lower
sampled denoise GPU peak (1.7424 GiB) is the useful memory observation; candidate
global peak occurred in encoder. Driver samples are not a tensor allocation
census or total/shared WDDM commitment, and this combined pair cannot attribute
the effect to either control individually. RSS about 0.57 GiB higher.
Retain optional memory-oriented candidate with selected two-step parity/native
acceptance, not a new default or larger-workload guarantee. No automatic repeat,
eight-step/ten-second/decode/general-model acceptance or budget/image-route change.

## Single compact/residual eight-step validation: preparation only

Prepared command (CUDA approval pending):
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualEightSteps`.
One candidate only, reusing the accepted two-step configuration, native DLL,
original norm boundaries and budgets; only denoise step count increases to
eight. Duration remains eight seconds, 1280x704, 24 fps, seed 43, no input audio,
joint audio generation, latent-only output. No repeated small gate, new kernel,
decoder, ten-second trial or automatic second run. Thirty CPU launcher contracts
passed in 12.54 seconds. Completion requires eight complete timing intervals,
finite saved video/audio and existing native cleanup/admission/transport checks.
This is not a matched eight-step numerical comparison or performance claim.

## Single compact/residual eight-step candidate: approved completion

Approved retry command:
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualEightSteps -DisableTlsKeyLog`.
Artifact: `E:\LTX-2\outputs\regional_compile_compact_residual_8steps_20261008_100346_240`.
The first attempt stopped in SSL preflight before CUDA/model execution; the
approved process-scoped TLS-key-log opt-in passed preflight and its environment
was restored. No TLS certificate validation/encryption change. Same selected
1280x704 / eight-second / 24-fps / 193-frame BF16 workload and budgets as the
accepted two-step pair, now eight steps. No input audio conditioning, joint
audio generation, latent-only output; no decoder or second run.

Completed, exit 0, saved video/audio finite. Both optimizations executed, repeat
880, original opaque norm/compilation configuration completed. Native gates
passed: eight timing intervals, 2412 samples, zero failures/drops/pending,
events released, final source tickets zero; 64 materializations, no temporary
fallback. Protected DLL unchanged; earlier image/linear/NVMe/native routes intact.

| Metric | Single candidate |
| --- | ---: |
| Total runner seconds | 370.9189 |
| CUDA denoise phase seconds | 329.50753125 |
| First CUDA interval seconds | 49.71496484375 |
| Mean CUDA steps 2-8 seconds | 39.968712611607145 |
| Minimum/maximum CUDA steps 2-8 seconds | 39.14959765625 / 40.755296875 |
| Sampled global baseline-relative GPU peak GiB | 4.115234375 |
| Sampled denoise baseline-relative GPU peak GiB | 3.8184 |
| Sampled denoise absolute driver GPU peak GiB | 4.8452 |
| Sampled process RSS peak GiB | 28.501537322998047 |

Lower sampled denoise GPU usage persisted; global peak occurred in encoder
(sampled absolute 5.1421 GiB). Driver samples are not total/shared WDDM commitment
or a tensor census. Prior separate eight-step observations: corrected eager
Triton mean 41.690544 s / total 375.6927 s; opaque-norm regional compilation
mean 43.845735 s / total 395.0373 s. This run has the lowest observed mean/total
of those selected records, not a matched speedup claim. No eight-step numerical
reference comparison, decoded-quality/general-model/ten-second acceptance,
default promotion or automatic next benchmark.

## Single ten-second capacity trial: historical preparation

Prepared command (separate CUDA confirmation pending):
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualTenSeconds -DisableTlsKeyLog`.
One native compact/residual candidate, ten seconds at 24 fps (241 frames),
1280x704, BF16, seed 43, eight steps, no input audio conditioning, joint audio
generation, latent-only output. All settings/budgets from the accepted single
eight-second candidate remain unchanged; only duration and artifact paths differ.
Thirty-nine CPU launcher contracts passed in 14.76 seconds. Require finite
outputs, completion and existing native ownership/admission/transport checks.
No decoder/repeated small gate/second run, budget increase, resolution reduction,
criteria relaxation or default change. CUDA and ten-second capacity are pending.

## Single ten-second capacity candidate: approved completion

Approved command:
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualTenSeconds -DisableTlsKeyLog`.
Artifact: `E:\LTX-2\outputs\regional_compile_compact_residual_10s_8steps_20261008_102137_296`.
One 1280x704 BF16 native compact/residual candidate, ten seconds / 24 fps / 241
frames / eight steps / seed 43, no input audio conditioning, joint audio
generation, latent-only output. Same selected DLL, original opaque norm boundaries,
compilation and budgets as the accepted eight-second candidate; no code/budget/
resolution change or second attempt. Process-scoped TLS key-log opt-in retained
certificate validation/encryption.

Completed without OOM, exit 0; saved video/audio finite. Both optimizations
executed, repeat 880. Native gates passed: eight timing intervals, 2412 samples,
zero failures/drops/pending, events released, final tickets zero; 64
materializations, no temporary fallback. Protected allocator hash unchanged.

| Metric | Ten-second single candidate |
| --- | ---: |
| Total runner seconds | 525.1151 |
| CUDA denoise phase seconds | 485.89515625 |
| First CUDA interval seconds | 71.39725 |
| Mean CUDA steps 2-8 seconds | 59.21119754464286 |
| Minimum/maximum CUDA steps 2-8 seconds | 58.349015625 / 61.05320703125 |
| Sampled global baseline-relative GPU peak GiB | 4.228515625 |
| Sampled denoise baseline-relative GPU peak GiB | 4.2285 |
| Sampled denoise absolute driver GPU peak GiB | 5.2554 |
| Sampled process RSS peak GiB | 29.214431762695312 |

Selected ten-second denoising capacity accepted, not decoded media/quality,
numerical-reference parity, longer-duration or general-model support. This
no-input-audio latent-only workflow is not matched evidence against historical
audio-conditioned limits. Eight/ten seconds differ in workload; do not interpret
their times as a regression A/B. Driver samples are not total/shared WDDM
commitment or a tensor census. Defaults/earlier routes intact; no automatic
next generation or decode. Saved latents permit separately approved decoding
without repeating the transformer.

### Selected ten-second saved-latent decode completion

The separately approved host offline decoder completed with exit code 0, without
OOM or rerunning transformer/text encoder. Existing temporal streaming video
decode and staged audio VAE/vocoder used normal PyTorch allocation, not a new
native allocator gate. Process elapsed 60.6969448 s (loading, decoding, encoding
and source verification included); video stage 51.0414008 s, audio VAE 0.6233013 s,
vocoder 1.5186058 s. Ten host CPU admission tests and a CUDA-free real-artifact
dry-run passed before execution.

ffprobe independently confirmed H.264 1280 x 704, 24 fps, 241 frames, 10.041667 s;
AAC stereo 48000 Hz, 10.010000 s. Pre-encode audio was finite, `[2, 480480]`.
Saved latent hash was unchanged. Host artifact:
`E:/LTX-2/outputs/regional_compile_compact_residual_10s_8steps_20261008_102137_296/candidate/ltx23_modular_10s_decoded.mp4`,
with adjacent `.decode.json` report. Selected media completion accepted; visual/
audio quality and decode-reference parity remain unreviewed. Separate-process
decode is not a matched full-pipeline timing; prior denoise evidence, defaults,
protected runtimes and native binary remain unchanged.

### Ten-second original-audio-conditioned selected capacity gate

The approved host trial used BF16, 1280 x 704, 24 fps, 241 frames, seed 43 and
eight steps with the prior native limits and compiled compact/residual controls.
The original 11.064-second local audio was cropped to the first ten seconds for
conditioning. Artifact: `E:/LTX-2/outputs/regional_compile_compact_residual_audio_10s_8steps_20261008_111849_823/candidate`.

Runner total 496.0545 s (latent-only); CUDA phase 453.3800625 s; first interval
59.23809375 s; mean intervals 2-8 56.30334040178571 s/it. Sampled denoise absolute
driver GPU peak 5.3823 GiB; global baseline-relative peak 4.35546875 GiB; process
RSS peak 28.66124725341797 GiB. No OOM; saved video/audio latents finite. Native
timeline complete with 2412 samples, events released and zero failure/drop/pending;
source tickets collected, selected fault/admission/transport/quarantine counters
zero, 64 materializations, zero temporary fallbacks, protected DLL unchanged.

The model completed, but the new launcher workload assertion initially returned
exit 1 by comparing RunTracker's phase-list `steps` field with integer eight.
After correcting this metadata-only assertion, the same artifact passed the full
safety gate in CPU-only verification; no inference retry or artifact modification.
Actual eight-step completion is checked by complete CUDA intervals. Selected
conditioned denoising accepted; decode/media quality and matched reference parity
remain pending. Do not infer a conditioning speedup from separate ordered runs.

The separately approved offline conditioned decode subsequently completed without
OOM: 60.3013161 s including loading/encoding/hash checks, no transformer/encoder
reload. Adjacent output `ltx23_modular_10s_audio_conditioned.mp4` and `.decode.json`.
ffprobe verified 241 H.264 frames, 1280 x 704, 24 fps, 10.041667 s and AAC stereo
48000 Hz, 10.010000 s. Waveform finite, original latent hash unchanged. Selected
conditioned media completion accepted; user synchronization/quality review pending.
Normal PyTorch allocation was used for the existing decoders; no native/default
change. This separate decode does not replace the latent-only timing evidence.

### Ten-second conditioned dense-versus-compact timestep observation

One approved dense control completed, exit 0, compared with the saved compact
conditioned run. Only timestep compaction changed; residuals in-place, compilation,
BF16, audio/image, 1280 x 704, 24 fps, 241 frames, eight steps, seed 43, native DLL
and ceilings remained the same. Control artifact:
`E:/LTX-2/outputs/regional_compile_dense_timesteps_audio_10s_8steps_20261008_113918_836/candidate`.

| Measurement | Dense | Compact |
| --- | ---: | ---: |
| Mean CUDA steps 2-8 s/it | 64.03968694 | 56.30334040 |
| Runner seconds, latent-only | 712.7422 | 496.0545 |
| First CUDA interval seconds | 222.415859375 | 59.23809375 |
| Sampled denoise absolute driver GPU peak GiB | 7.9995 | 5.3823 |
| Sampled process RSS peak GiB | 29.29014206 | 28.66124725 |

Compact observed 7.7363 s/it lower (about 12.08%) and 2.6172 GiB less sampled
denoise GPU peak. Total difference includes an unusually long dense first
interval; compilation/paging causality was not measured. Single ordered runs
are not statistical or general-model evidence. CPU comparison found final
video/audio latents bit-exact and finite. Dense control passed all selected
native gates, eight complete intervals, 2412 timeline samples, released events,
zero selected failure counters, collected tickets, 64 materializations and no
temporary fallbacks. 45 CPU launcher contracts passed before execution. No OOM,
budget changes, candidate rerun, extra decode or default/legacy route changes.
Retain compact timesteps as an optional selected-workload memory feature with
observed performance benefit; no wider-model support claim.

### Optional FP8 weight-only small native correctness gate

On 2026-10-08, the explicitly approved eight-layer 512-wide Windows sequence
passed BF16 control followed by TorchAO 0.18.0 FP8 weight-only, exit 0. Evidence:
`E:/LTX-2/outputs/fp8_native_gate_05/{bf16_control,fp8}/report.json`.
Each checkpoint was saved and reloaded as safetensors; existing backend/model
math forward was preserved. Three repeated outputs matched each case's own
CUDA reference exactly; this does not assert FP8 equals BF16. FP8 data/scales
were verified exactly. Both used native resident and streamed units, recovered
from intentional RuntimeError/KeyboardInterrupt, reused cached file bytes and
finished with zero transfer events, batch events and tickets after batch close.
Both completed meta retirement and store cleanup, without fault failures or
temporary fallbacks. Preserved DLL and production defaults were unchanged.

Physical parameter bytes were 4,202,496 BF16 and 2,121,728 FP8. Native residency
was 6 MiB for both due to the diagnostic's page/slot geometry; cache was 6 MiB,
staging 4 MiB, combined host ceiling 10 MiB, allocator cache 4 MiB. This is not
an LTX performance/quality benchmark or proof of VRAM reduction. Real LTX
weights remain BF16. Failed preparatory fixtures and corrected gate contracts
are documented in `docs/fp8-weight-only-integration-plan.md`, not counted as
passing runs. Reusable conversion tooling and explicit real-model integration
remain pending.

The subsequent selected public API gate passed both cases, exit 0, using saved
CPU fixtures: `E:/LTX-2/outputs/fp8_public_gate_02/{bf16,fp8}_report.json`.
This validates DDO loading, attachment, metrics and cleanup, not only direct
adapter calls. Three repeated outputs matched each case's own CUDA reference;
physical FP8 data/scales matched exactly. Native streaming, unchanged source
reads, interruption recovery and meta/source-descriptor retirement passed.
Fault failures, temporary fallbacks and cache admission denials were zero.
Both native residency snapshots were 6 MiB, with declared 64 MiB host cache,
4 MiB staging and 68 MiB combined host ceiling; DLL remained unchanged. This is
correctness evidence, not LTX FP8 quality/performance evidence. The reusable
converter and explicit loader/runner option are implemented; full LTX conversion
remains pending, requiring bounded-memory preparation and a separate output
drive. Earlier preparatory failures remain historical evidence, not passing gates.

Bounded offline conversion preparation subsequently passed 129 CPU tests and
six subtests in 16.50 s (`fp8_bounded_integration_cpu_01`). The tensor-at-a-time
converter matched the existing converter's saved payload hashes and the public
loader, without full CPU model loading. Original source files and untouched
tensors were preserved; invalid index and source-change failures retained
diagnostics. Header-only inspection found 4,186 BF16 source tensors, largest
288 MiB. Real output is planned at
`F:/models/ltx23-distilled-fp8-weight-only/transformer`; conversion and real-model
FP8 quality/performance remain pending. This is not a CUDA benchmark result.

On 2026-10-09 the approved bounded CPU conversion of the real transformer
completed, exit 0. Output:
`F:/models/ltx23-distilled-fp8-weight-only/transformer`. The source's 35.3695 GiB
of tensor data became 17.7235 GiB across ten safetensors shards, approximately
49.89% smaller (output includes headers). 1,660 linear weights were converted;
all saved FP8 bytes, FP32 scales and untouched tensors reconstructed exactly.
Source hashes matched between passes; original BF16 files were not modified.
Observed process peak working set was approximately 15.11 GiB. Report:
`conversion_report.json`, `passed`, exact reload true, CUDA initialized false.
No real-model inference ran; quality, VRAM and speed improvements remain
unmeasured. The accepted BF16 baseline and protected runtimes are unchanged.

The first approved real FP8 inference gate (1280x704, 5 s, 24 fps, four steps,
seed 43, audio conditioning) failed before transformer attachment/denoising:
CPU immutable-source binding rejected a recycled temporary flatten-clone
address between shards. Encoder/conditioning completed; no CUDA/OOM error was
reported and no FP8 denoise time was measured. Artifacts:
`E:/LTX-2/outputs/fp8_real_5s_4steps_gate_01.console.log` and gate metrics.
After limiting temporary validation registries to each shard, focused CPU
contracts passed 29 tests. Real CPU loading then passed with 4,186 logical
tensors, 5,846 correctly bound physical sources and exact saved payload hashes;
CUDA remained uninitialized. No DLL, runtime budgets or BF16 route changed.
Corrected CUDA generation remains unvalidated pending another approved attempt.
The consolidated corrected CPU integration suite passed 130 tests and six
subtests in 17.39 s; this does not replace real-model CUDA acceptance.

The approved unchanged retry `fp8_real_5s_4steps_gate_02` passed real checkpoint
loading/source binding but failed component layout preflight before denoising:
root-owned parameters overlap descendant execution units. Meta-only inspection
found two BF16 root tables totaling 24 KiB. The failed-run elapsed was
103.4579 s, load/prepare 32.5 s; this is not FP8 denoising performance.
No CUDA/OOM error was reported. A bounded component-lifetime root-state
extension is proposed, not implemented or accepted. No DLL, weight budget,
overlap guard or protected runtime changed.

The approved FP8-only bounded root-state extension is now CPU-accepted:
138 tests and six subtests in 17.23 s (`fp8_root_final_cpu_01`). It reuses one
existing native lifetime-pinned buffer and immutable source upload, subtracts
its page-rounded capacity from the unchanged total ceiling, includes backing
in residency metrics, and fences retirement/destruction. The dense root guard
and native DLL are unchanged. Real CPU geometry passed with 24 KiB root state,
76 execution units and a largest slot payload of 387,323,904 bytes. This is
not CUDA generation/performance acceptance. A small lifecycle CUDA gate is
required before another LTX attempt; neither ran in this implementation step.

The subsequently approved small root-state CUDA gate passed BF16 and FP8,
exit 0, approximately 6.09 s. Evidence:
`E:/LTX-2/outputs/fp8_root_public_gate_01/{bf16,fp8}_report.json`.
Each case matched its own reference exactly, recovered after RuntimeError and
KeyboardInterrupt, and completed cleanup. FP8 root payload was 24 KiB, uploaded
once into a 2 MiB native backing charged against its 8 MiB total fixture ceiling;
arena ceiling remained 6 MiB. Observed total weight residency was 8 MiB including
root backing. Faults/fallbacks/cache denials were zero and cached read bytes
did not increase across repeats. This is lifecycle acceptance only; real LTX
FP8 quality/performance and four-step generation remain unvalidated.

The separately approved real FP8 retry `fp8_real_5s_4steps_gate_03` completed
on 2026-10-09, exit 0: 1280x704, 5 s, 24 fps/121 frames, four steps, seed 43,
image/prompt/audio conditioning, native BF16 encoder, no compilation or custom
execution kernels, latent output. Transformer/encoder weight ceilings remained
1.5/4 GiB; chunk sizes 1024, phase modulation, compact timesteps and in-place
residuals enabled. Load/prepare: 33.0865 s; attach: 85.4993 s; denoise:
109.2867 s (progress approximately 27.09 s/it); total: 259.2684 s, without decode.
Sampled peak GPU: 5.4516 GiB baseline-relative / 6.4784 GiB absolute; peak RSS:
27.2126 GiB. Fault failures, temporary tensors and cache admission denials: zero.
Root state uploaded once within the unchanged ceiling; completion tickets
retired to zero and process cleanup completed without a reported error.
CPU-only saved-latent validation passed: finite video and audio tensors.
Evidence: `E:/LTX-2/outputs/fp8_real_5s_4steps_gate_03/metrics/modular_run.json`.
This is functional acceptance, not decoded quality, BF16 parity or evidence of
a performance improvement. The compiled 10 s BF16 baseline is not a matched
control; do not compare its per-step result directly with this gate.

The separately approved saved-latent decode completed with exit 0 on 2026-10-09,
without reloading transformer/encoder or rerunning denoising. Existing staged
BF16 decoders used ordinary CUDA allocation, not the native allocator. Elapsed:
33.7275 s (video 25.4463 s, audio VAE 0.5412 s, vocoder 1.2046 s).
Output: `E:/LTX-2/outputs/fp8_real_5s_4steps_gate_03/fp8_decoded.mp4`.
Independent ffprobe confirmed H.264, 1280x704, 24 fps, 121 frames/5.041667 s;
AAC stereo 48000 Hz, 5.010000 s; file size 1,023,116 bytes. Audio was finite.
Source latent SHA-256 remained
`c5a36eab7c22c74d6d90c9e8f42b331f5fb311728b4a8f39bf10f87da15c1373`.
Decode evidence is `fp8_decoded.decode.json` beside the MP4. Visual/audio quality
is not reviewed; matched BF16 quality/performance comparison remains pending.

### Approved five-second BF16 quality control (2026-10-09)

`E:/LTX-2/outputs/bf16_5s_4steps_fp8_quality_control_01` completed with exit 0.
Workload controls match the FP8 gate: 1280x704, 5 s/121 frames at 24 fps, four
steps, seed 43, same prompt/image/audio, chunk sizes 1024, phase modulation,
compact timesteps and in-place residuals, no compilation. Weight ceilings and
preserved DLL are unchanged. BF16 uses the existing dense route, whereas FP8
requires composite component ownership; these timings are an integration-level
comparison, not an isolated dtype/kernel experiment. Cache warmth is uncontrolled.

| Phase/observation | BF16 control | FP8 gate |
| --- | ---: | ---: |
| Transformer load/prepare (s) | 1.6204 | 33.0865 |
| Transformer attach (s) | 6.5577 | 85.4993 |
| Denoise, four steps (s) | 118.1183 | 109.2867 |
| Total without decode (s) | 155.3928 | 259.2684 |
| Sampled global GPU peak, absolute (GiB) | 4.9624 | 6.4784 |
| Sampled process RSS peak (GiB) | 26.6593 | 27.2126 |

BF16 fault failures, temporary tensors and cache admission denials were zero.
The report records 107 commit-pressure entries and recoveries, not zero pressure.
Completion collection retired two tickets to zero; cleanup completed without a
reported error. CPU-only validation found finite video/audio latents with matching
shapes. Relative latent RMSE, `RMS(FP8-BF16)/RMS(BF16)`, was 0.194170 for video
and 0.003182 for audio. This measures final latent divergence, not perceptual
quality or temporal stability; no visual conclusion follows from it alone.
BF16 decode/user quality review remains pending and requires separate approval.

The subsequently approved standalone BF16 decode completed with exit 0 in
32.7062 s, without rerunning denoising. Existing staged decoders used ordinary
CUDA allocation; transformer/encoder were not loaded. Output:
`E:/LTX-2/outputs/bf16_5s_4steps_fp8_quality_control_01/bf16_decoded.mp4`.
Independent ffprobe confirmed H.264, 1280x704, 24 fps, 121 frames/5.041667 s;
AAC stereo 48000 Hz, 5.010000 s; file size 1,000,593 bytes. Audio was finite
and source latents unchanged (SHA-256
`840ca4569bf5cf38e399350ff0e983226ac664ac210d9dfa194ee5b5d38ea2ed`).
Evidence: adjacent `bf16_decoded.decode.json`. Both BF16 and FP8 media are now
available for user review; no visual/audio quality verdict has been recorded.

The user subsequently reviewed both five-second/four-step videos and reported
similar temporal artifacts, with no clear visual degradation attributable to
FP8. This is an initial subjective acceptance for that pair only, not numerical
parity or general-model/eight-step quality acceptance. No enhance LoRA was used.

### Ten-second FP8 comparison stopped at BF16 control (2026-10-09)

The approved sequence started with uncompiled BF16, 1280x704, 10 s/241 frames
at 24 fps, eight steps, seed 43, image/prompt/audio, compact timesteps,
in-place residuals, phase modulation and 1024 chunk sizes. Diagnostic timing
was enabled; weight ceilings stayed 1.5/4 GiB and the protected DLL hash matched.
`E:/LTX-2/outputs/bf16_10s_8steps_fp8_comparison_01` failed with an illegal
CUDA memory access before the first completed-step callback. Exit 1, recorded
elapsed 70.8338 s; no latent artifact or completed denoising timing exists.
Cleanup/event recording also failed after the CUDA context entered error.
The FP8 case was not started; do not report an A/B performance result.
Evidence: `metrics/modular_run.json` and
`metrics/denoise_snapshots/before_denoise.json` in the failed control directory.

Read-only comparison against the accepted compiled ten-second BF16 case found
more differences than compilation: execution auxiliary ownership, auxiliary
CPU-source detachment, pre-transformer host-cache release and host-memory phase
inspection were also disabled. The proposed control reused the five-second
recipe, not all accepted ten-second lifecycle controls. These differences are
established; their causal role in the illegal access is not established.
Do not change DLL/budgets or rerun automatically. A revised comparison must
preserve the accepted BF16 lifecycle controls and document the different FP8
composite-ownership requirements; further CUDA execution needs new approval.

The separately approved corrected BF16 control subsequently passed, exit 0:
`E:/LTX-2/outputs/bf16_10s_8steps_fp8_comparison_02`. Same 10 s/241-frame,
eight-step audio-conditioned workload and ceilings; compilation remains off.
Execution auxiliary ownership, host-cache release, auxiliary source detachment
and phase inspection were restored to the accepted BF16 settings. DLL unchanged.
Load/prepare: 2.1817 s; attach: 8.0553 s; denoise wall time: 462.5770 s;
CUDA phase: 462.48834375 s; first CUDA step: 58.65807422 s;
mean CUDA steps 2-8: 57.68680134 s/it. Total without decode: 502.5720 s.
Sampled GPU peak: 6.4530 GiB baseline-relative / 7.4798 GiB absolute;
process RSS peak: 28.6438 GiB. Fault failures, temporary tensors, cache
admission denials and cache pressure entries/recoveries were zero. No completion
tickets remained. CPU-only verification found finite float32 packed video
`(1, 27280, 128)` and audio `(1, 251, 128)` latents. Source storage observation
after attach and after generation was zero CPU bytes, not a process-RAM claim.
Evidence: completed `metrics/modular_run.json` and eight step snapshots.
This accepts the corrected configuration, not a causal isolation of the prior
illegal access. FP8 has not yet run at 10 s; the A/B result remains pending.

### Completed ten-second BF16/FP8 comparison (2026-10-09)

After separate approval, `E:/LTX-2/outputs/fp8_10s_8steps_comparison_01`
completed with exit 0. Same generation controls as corrected BF16: 1280x704,
10 s/241 frames, 24 fps, eight steps, seed 43, real prompt/image and source
audio, 1024 FF/norm chunks, phase modulation, compact timesteps, in-place
residuals, diagnostic timing and no compilation. Native weight ceilings remain
1.5 GiB transformer / 4 GiB BF16 encoder; staging 1.5 GiB and source/cache
ceilings 24.625 GiB. Protected DLL unchanged. Both cases release host cache
and inspect memory phases. BF16 uses validated execution auxiliary ownership
and source detachment; FP8 instead uses composite component ownership and
bounded root state, without incompatible auxiliary detachment. This is a
single integration-level comparison, not a statistically repeated or isolated
weight-dtype/kernel experiment; cache warmth is uncontrolled.

| Measurement | BF16 corrected control | FP8 weight-only |
| --- | ---: | ---: |
| Transformer load/prepare (s) | 2.1817 | 31.6734 |
| Transformer attach (s) | 8.0553 | 83.5672 |
| First CUDA step (s) | 58.6581 | 67.0129 |
| Mean CUDA steps 2-8 (s/it) | 57.6868 | 67.9183 |
| Denoise wall time (s) | 462.5770 | 542.5450 |
| CUDA phase including dependencies (s) | 462.4883 | 542.4588 |
| Total without decode (s) | 502.5720 | 687.5026 |
| Sampled absolute GPU peak (GiB) | 7.4798 | 7.4578 |
| Global process RSS peak (GiB) | 28.6438 | 25.5203 |
| Denoise process RSS peak (GiB) | 28.6438 | 24.0728 |
| Final source-cache committed/registered bytes | 26,440,892,416 | 19,193,135,104 |
| Cumulative source-cache DMA bytes | 210,610,591,808 | 146,452,988,032 |

The FP8 candidate has approximately 17.3% longer denoising, 17.7% longer
steady-state step time and 36.8% longer total elapsed, despite approximately
30.5% fewer cumulative source-cache DMA bytes. Global RSS is lower, but sampled
GPU peaks are effectively alike. Source-cache counters are scoped native
snapshots/cumulative traffic, not total RAM or shared WDDM occupancy. CUDA-event
time and host progress intervals have different boundaries; use the CUDA-event
values for first/steady-state comparisons. There is no measured speed advantage
for this selected FP8 implementation on this workload; do not infer that all
FP8 backends or architectures are slower or that DMA is the proven bottleneck.

Both routes reported zero fault failures, temporary tensors, cache admission
denials and cache pressure episodes. FP8 completion collection retired 132
tickets to zero. CPU-only verification found finite float32 video
`(1, 27280, 128)` and audio `(1, 251, 128)` latents, without CUDA initialization.
Evidence: both case directories' `metrics/modular_run.json`, eight step snapshots
and saved latent artifacts. No ten-second decode or quality review ran.
Keep accepted BF16 as the performance reference; FP8 remains optional for
checkpoint-size/host-memory savings. Further optimization requires evidence
separating setup, dequantization/math and component scheduling costs, not an
automatic budget increase, compilation opt-in or another benchmark sweep.

### FP8 CPU ownership-index candidate (2026-10-09)

The selected FP8 discovery previously traversed the entire component for each
composite weight and each ordinary parameter/buffer. The candidate captures
owner aliases and physical-storage identities once in a discovery-local index,
then reuses those groups. It still refuses distinct owners sharing composite
payload storage and preserves repeated-module names. No forward, native binary,
residency ceiling, transport, or protected runtime behavior was changed.

CPU validation: **34 passed in 7.39 s** across test_quantized_owners_cpu.py,
test_quantized_arena_cpu.py, and test_quantized_sources_cpu.py, using the
host .venv-vmm with TorchAO 0.18.0, CUDA_VISIBLE_DEVICES=-1, and pytest
external plugin autoload disabled. A new assertion verifies one ownership-index
traversal; tests cover wrapper/parameter/buffer storage aliases. The first DDO
environment attempt skipped three modules because TorchAO was absent. The first
host attempt stopped all 34 tests at their CPU guard because imports had
initialized CUDA before fixture entry; it is not passing CPU evidence.

No candidate CUDA/model performance measurement has run. Reduced attach work
does not establish faster denoising; selected FP8/BF16 comparison results above
remain unchanged. Compilation stays excluded pending a separate compatibility
contract, not automatically enabled as a consequence of this CPU change.

The separately approved existing public CUDA gate then passed, exit 0, using
the prepared root-state fixture in
.internal/test-artifacts/fp8-owner-index-preflight-20261009 and reports in
.internal/test-artifacts/fp8-owner-index-cuda-20261009. Both the unchanged BF16
control and FP8 candidate reported exact repeated outputs, completed cleanup,
and recovery from deliberate RuntimeError and KeyboardInterrupt. FP8 physical
payload verification passed, including the bounded 24 KiB BF16 root state.
The protected DLL SHA256 remained
EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.
Limits: BF16 weight 6 MiB, FP8 weight 8 MiB (existing root-state fixture charge),
host cache 64 MiB, staging 4 MiB, allocator cache 4 MiB. This validates the
selected small integration contract, not LTX performance, quality, or general
model support. No video benchmark ran after the ownership-index change.

### FP8 compile capture CPU contract (2026-10-09)

Two CPU fullgraph/static Dynamo+AOT eager cases passed: compiled backend math
uses updated qdata/scale storage after physical .data replacement and updated
wrappers after one owner publication. Changed scale intentionally changes the
reference output, detecting stale-weight capture. Repeated compiled results
equal current backend eager output exactly; the original Linear forward class
remains unchanged, and final retirement invalidates both payloads to meta.
The corrected consolidated ownership/source/arena/compile suite reports
**36 passed in 8.54 s**, with CUDA hidden and no CUDA initialization. An initial
fixture attempted a forbidden second publication and was corrected without
relaxing the production ownership state machine.

This is not CUDA Inductor, fused-kernel, native-hook compilation, or performance
acceptance. The existing public small CUDA probe now has an explicit opt-in
compile-fp8-linear candidate: only FP8 Linear compute forwards are compiled
before native attach, fullgraph/static Inductor, CUDA graphs disabled, precision
cast emulation enabled. Block payload verification and interruption checks stay
outside compiled compute; the BF16 control and all budgets remain unchanged.
Native retirement must precede removal of compiled forwards. The host runner's
FP8 compilation exclusion remains in place until a separate gate is approved
and passes. No compiled CUDA candidate has run yet.

The separately approved small CUDA Inductor candidate subsequently ran with
the same prepared fixture and unchanged binary/ceilings, evidence in
.internal/test-artifacts/fp8-compile-cuda-20261009. BF16 eager control passed
exact repeated outputs, interruption recovery and cleanup. FP8 compiled eight
Linear forwards, but stopped at the first exact-output check: 1,165/2,048
elements differed (56.9%), maximum absolute difference 0.00048828125, maximum
relative difference 5.0. The relative maximum alone does not establish a large
signal error because near-zero values can magnify ratios; it does not waive
the agreed exact gate. FP8 cleanup completed without a cleanup error, and the
DLL SHA256 remained unchanged. No FP8 compiled interruption/reuse acceptance
or model benchmark followed the failed first numerical check.

Compilation success is not numerical acceptance. Preserve the host FP8
compilation exclusion and existing uncompiled route; do not automatically
relax tolerances. TorchAO's selected weight-only path dequantizes the weight,
performs matmul and then a separate bias addition. Compiled fusion/rounding
boundaries are a diagnostic hypothesis, not a proven cause of this mismatch.

Read-only inspection of the failed gate's generated Inductor code subsequently
confirmed separate FP8-to-FP32 scale multiplication and BF16 weight storage,
followed by extern_kernels.addmm for the compute+bias portion. In contrast,
TorchAO eager uses matmul followed by a separate bias addition. Thus the
compiled graph did change the eager BF16 rounding boundary. This identifies
a concrete graph difference; only a passing corrected gate can establish that
it explains the observed numerical mismatch.

The small opt-in candidate now disables Inductor pattern_matcher and
epilogue_fusion, keeping fullgraph/static compilation, precision-cast emulation
and CUDA graphs disabled. Options are included in the report. This conservative
diagnostic preserves the separate matmul/bias boundary rather than changing
TorchAO math or relaxing tolerances. It does not enable compilation in the host
runner and has not run on CUDA yet. BF16 control, binary and ceilings unchanged.

The separately approved corrected small CUDA gate passed with exit 0. Evidence:
.internal/test-artifacts/fp8-compile-rounding-cuda-20261009. BF16 stayed eager;
FP8 compiled eight Linear compute forwards with pattern_matcher=false,
epilogue_fusion=false, emulate_precision_casts=true and triton.cudagraphs=false.
Both cases passed exact repeated-output equality, deliberate RuntimeError and
KeyboardInterrupt recovery, and cleanup; FP8 physical-payload verification
passed. The DLL hash and fixture ceilings remain those recorded above.

Disabling these fusion settings resolves the observed mismatch in this selected
fixture without relaxing tolerances. Because two fusion settings changed
together, the gate does not isolate their individual effects. The previous
generated addmm remains the concrete diagnosed graph difference. Acceptance is
limited to this small CUDA/public native integration contract, not LTX regional
compilation, speed, media quality, or arbitrary model support. The production
runner still excludes FP8 compilation; no video benchmark followed this gate.

### Selected 10-second no-conditioning model comparison (2026-10-09)

The explicitly approved BF16 control completed: 1280x704, 24 fps, 241 frames,
seed 43, eight steps, no source-audio conditioning, latent output. Generated
audio remains enabled. Total excluding decode was 502.3321 s; denoise wall was
462.1879 s and CUDA steady steps 2-8 averaged 57.708845 s/it. Transformer load
was 2.3029 s and attach was 8.9207 s. Sampled denoise absolute GPU usage peaked
at 7.1186 GiB (baseline-relative delta 6.0918 GiB), RSS at 28.5847 GiB.
CPU inspection confirmed finite video latents (1, 27280, 128) and audio latents
(1, 251, 128). Evidence: sibling host outputs/bf16_10s_noaudio_comparison.

The next approved FP8 compiled candidate failed before completing its first
step with FailOnRecompileLimitHit, fullgraph=True. Total until failure was
79.4792 s, transformer load 32.6908 s and attach 13.3864 s; 1,660 Linear
wrappers were configured. These are setup/failure measurements, not FP8
denoise speed or quality acceptance. Evidence: sibling host
outputs/fp8_10s_noaudio_compiled_comparison. No automatic retry or compiler
limit increase followed. The successful small eight-Linear contract did not
cover real-model geometry diversity.

CPU diagnosis reproduced the limit on the ninth distinct Linear geometry,
whereas twelve equal geometries passed. Dynamo's shared external wrapper
reported a bias-size guard mismatch. Cloning the library forward code alone
still shared that external wrapper and failed; it is not the correction.
The experimental helper now compiles the standard functional Linear entry
with a separate code/cache per weight geometry, passes current weight/bias
views at invocation, and retains TorchAO dispatch and the accepted rounding
options. Compiler limits, native allocator, BF16 and uncompiled routes remain
unchanged. This correction requires separate CUDA acceptance; CPU evidence
alone is not a model performance result.

Corrected CPU ownership/source/arena/compile contracts: 39 passed in 8.65 s,
CUDA hidden. Twelve distinct geometries each produced a graph and exact current
backend output, including changed scale storage after capture. Global compiler
limits remained unchanged. The first consolidated attempt passed 16 tests but
23 fixture setups failed on access to the default pytest temporary directory;
the corrected run used a new project-local temporary directory and disabled
pytest cache writes. The CUDA probe now invokes the same compilation helper
as the host, rather than a separate bound-forward implementation. No corrected
CUDA run or model benchmark has executed yet.

The separately approved corrected-helper CUDA gate subsequently passed, exit 0.
Evidence: .internal/test-artifacts/fp8-geometry-cuda-20261009. BF16 remained
eager; FP8 used eight Linear dispatches sharing one weight-geometry cache.
Both reported exact repeated outputs, RuntimeError/KeyboardInterrupt recovery
and completed cleanup. FP8 physical payload verification passed, including
the bounded root-state fixture. The protected DLL SHA256 remained
EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.
Weight ceilings remained 6 MiB BF16 / 8 MiB FP8, host cache 64 MiB, staging
4 MiB and allocator cache 4 MiB. Compilation options stayed unchanged.

Scope: CUDA numerical/ownership acceptance for the selected small fixture,
not real-model speed or CUDA geometry-diversity acceptance. Twelve geometries
were exercised by the CPU contract, not by this one-geometry CUDA fixture.
No corrected full-model benchmark has run. Emitted TorchAO enum-registration,
serialization-import and stable-cache-hash warnings did not fail this gate;
this does not establish durable cross-process compiler-cache reuse.

### Corrected FP8 compilation: selected full-model execution (2026-10-09)

The separately approved ten-second/eight-step candidate completed, exit 0:
sibling host outputs/fp8_10s_noaudio_geometry_comparison. Workload remains
1280x704, 24 fps/241 frames, seed 43, no source-audio conditioning, generated
audio active, latent output without decode. The same protected DLL and native
ceilings were retained. The 1,660 standard FP8 Linears used 19 weight-geometry
caches; denoise_completed and callback_retired were both true. No compiler
limit increase or fallback to eager execution was applied by the helper.

| Selected no-source-audio case | CUDA steady steps 2-8, s/it | Denoise CUDA, s | Total without decode, s |
| --- | ---: | ---: | ---: |
| BF16 control | 57.708845 | 462.089156 | 502.3321 |
| FP8 geometry-isolated Linear compile | 63.442852 | 516.996156 | 596.6240 |

FP8 CUDA step times: 72.875188, 63.910988, 64.192656, 64.001320,
63.927133, 63.920270, 64.115055, 60.032543 s. The first interval includes
lazy compilation; it is not an isolated compiler-duration measurement.
FP8 is 9.94% slower in steady-state and 18.77% slower in one-shot total than
the selected BF16 control. BF16 retains auxiliary ownership/source detachment;
FP8 uses composite/root ownership. This compares supported routes, not dtype
alone with identical ownership behavior.

The older audio-conditioned uncompiled FP8 result was 67.918261 s/it and
687.5026 s total. The new steady-state is 6.59% shorter, but audio conditioning
and ownership-index setup differ between runs; do not attribute that entire
difference to compilation. A matched no-source-audio uncompiled FP8 benchmark
has not run. BF16 remains the selected performance reference; the candidate
does not meet a speed-improvement acceptance criterion against BF16.

Reported FP8 global sampled GPU peak is 6.293381 GiB baseline-relative, not
absolute driver usage; global RSS peak is 25.480087 GiB. Transformer source DMA
was 146,452,988,032 bytes. No cache-pressure entries, failed observations,
temporary fallbacks or driver allocation-operation failures were reported.
Source tickets collected from 132 to zero, native timeline events were released,
and compiled callbacks retired after native retirement. CPU inspection confirmed
finite float32 video latents (1, 27280, 128) and audio latents (1, 251, 128),
with CUDA uninitialized during inspection. These are selected execution/cleanup
checks, not decoded visual quality or exact full-model eager parity.

The raw report retains its static validation description stating CUDA acceptance
is pending; preserve this recorded field and use completed status, CUDA timing,
compile completion and the separately recorded gates as actual execution evidence.
No automatic additional benchmark, residency adjustment or default change follows.

### Independent local host transformer: selected BF16 migration gate (2026-10-09)

The separately approved sibling-host migration check completed with exit 0:
`outputs/local_transformer_bf16_regression/metrics/modular_run.json` in the
LTX-2 host. The report records implementation
`diffusers_modules.local.transformer_ltx2_video`, loaded through the trusted
custom-block component specification. DDO and the protected native DLL were
unchanged (SHA256 EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6).

Matched requested conditions: BF16, 1280x704, 10 s at 24 fps/241 frames,
seed 43, eight steps, real prompt/image, no source-audio conditioning but
generated audio active, latent output without decode. Transformer residency
remained 1.5 GiB, encoder residency 4 GiB, staging 1.5 GiB and native source
cache/registration 24.625 GiB. Phase modulation, compact timesteps, in-place
residuals, rotary reuse, 1024-token chunks and auxiliary ownership were retained.
No compilation, custom compute kernel or quantization was enabled.

| Selected BF16 implementation | CUDA steady steps 2-8, s/it | Denoise CUDA, s | Total without decode, s |
| --- | ---: | ---: | ---: |
| Previous fork-backed control | 57.708845 | 462.089156 | 502.3321 |
| Independent local transformer | 59.715444 | 477.993750 | 518.7514 |

Steady-state was 3.48% longer and total was 3.27% longer, within the explicitly
approved 5% investigation threshold. This single sequential comparison does
not establish the cause of the timing difference or an improvement; retain the
previous control as the performance reference. Sampled baseline-relative GPU
peak was 6.390427 GiB versus 6.091766 GiB (about 0.30 GiB higher); the new run's
denoise baseline-adjusted absolute estimate was 7.4173 GiB. Global sampled RSS
peak was 28.587563 GiB. These are sampled observations, not allocation censuses.

CPU inspection with CUDA uninitialized confirmed exact equality of both saved
float32 tensors against the preserved control: video (1, 27280, 128), audio
(1, 251, 128), maximum and mean absolute difference zero, all finite.
Transformer source tickets were zero at final collection; native timeline
pending/failures/dropped were zero and event_cleanup was released. No temporary
fallback, fault/admission/source-read failure or source-pressure episode was
reported. Allocation diagnostics completed without failure. The selected BF16
migration gate passed; it does not validate FP8, optional kernel/compile variants,
decoded visual quality, the image route, or arbitrary models. No additional CUDA
run or budget adjustment is automatically authorized by this result.

### Local FP8 chunk-reuse candidate: CPU only (2026-10-09)

The sibling host implements opt-in per-FeedForward converted-weight reuse,
without changing the native allocator, accepted BF16 route or existing FP8
default. Retained BF16 temporaries are explicitly capped and event-owned; no
full-model BF16 cache is added. Native projection boundaries remain installed.
The selected CPU fixture demonstrated two conversions instead of 54 over 27
token chunks, with exact backend outputs and no retained converted outputs.
Consolidated final CPU acceptance: 92 tests and 11 subtests, 12.89 s. The local
saved-checkpoint preflight also passed exact outputs, interruption recovery
and cleanup with 128 KiB retained-weight peak and CUDA uninitialized
(.internal/test-artifacts/fp8-chunk-preflight-20261009/report.json).

This is not CUDA ownership, capacity or performance acceptance. Compiled compute
was stubbed in the CPU configuration contract; event lifecycle contracts used
test events. The preflight native budgets are requested controls, not executed
native observations. A small compiled/native CUDA fixture is prepared in the
host but has not been executed or approved. No generation benchmark, budget
increase or speed-improvement claim follows from these CPU results.

### FP8 chunk-reuse CUDA gate: setup failure (2026-10-09)

The subsequently approved `outputs/fp8_chunk_reuse_cuda_gate_01` attempt stopped
before candidate model execution: checkpoint conversion ran after native allocator
bootstrap and correctly refused with `Run conversion in a fresh CPU-only process`.
The partial checkpoint directory is preserved. No report was generated because
preparation preceded the harness exception handler. This is a harness setup
failure, not CUDA acceptance or evidence of a compute regression.

The corrected harness requires a saved fixture from a passed CPU-only preflight
for CUDA execution, records its file hashes, and includes preparation failures
in report handling. The converter safety guard and native runtime are unchanged.
A corrected CUDA attempt requires renewed approval; no video benchmark follows
automatically.

Corrected fixture-reuse CPU validation passed with CUDA uninitialized, exact
outputs, interruption recovery, complete cleanup and a 128 KiB retained-weight
peak. File hashes are recorded in
`.internal/test-artifacts/fp8-chunk-fixture-reuse-cpu-20261009/report.json`.
This does not validate compiled compute or native CUDA ownership.

### Local FP8 chunk reuse: selected small native gate passed (2026-10-09)

The separately approved corrected host gate completed with exit 0. Evidence:
`E:/LTX-2/outputs/fp8_chunk_reuse_cuda_gate_02/report.json`. It reused the passed
CPU fixture without conversion after native bootstrap. The protected DLL hash
remained EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.

Eight compiled graphs were observed. Repeated outputs matched the backend
reference exactly; deliberate RuntimeError and KeyboardInterrupt recovery passed.
Across normal, interrupted and recovery calls: 22 FF scopes, 44 conversions and
120 reuse hits. Retained-weight peak was 131072 bytes (the explicit 128 KiB cap),
with zero pending/quarantined bytes after removal. Two event-specific waits took
0.0001157 host seconds; this is not GPU timing. Cleanup completed and parameters
returned to meta. The gate asserted zero native fault failures, temporary
fallbacks and cache admission denials, with residency within the 6 MiB ceiling.
Requested staging/source/cache limits remained 4 MiB/64 MiB/4 MiB respectively.

This accepts only the small compiled/native ownership contract, not LTX quality,
real-model memory headroom or performance. No image/linear/NVMe route, allocator
implementation, default or full-model BF16 cache was changed. A real-model
comparison remains separately approval-gated; no speedup is claimed.

### Local FP8 FF chunk reuse: selected 10-second comparison (2026-10-09)

The approved sibling-host comparison completed with exit 0. Evidence:
`E:/LTX-2/outputs/fp8_10s_noaudio_chunk_reuse/metrics/modular_run.json`.
Conditions matched the saved compiled FP8 control: 1280x704, 10 s at 24 fps
(241 frames), seed 43, eight steps, real prompt/image, no source-audio conditioning,
generated audio active and latent-only output. Native transformer/encoder
residency remained 1.5/4 GiB, staging 1.5 GiB, source cache/registration
24.625 GiB and the protected DLL was unchanged. FF/normalization chunks were
1024 tokens; phase modulation, compact timesteps, in-place residuals and rotary
reuse were retained. The candidate additionally used explicit 256 MiB scoped
converted-weight retention and the independent local transformer. The older
control used the fork-backed transformer, so this is not an isolated same-build
toggle comparison, although saved outputs are exact.

| Selected FP8 route | CUDA steady steps 2-8, s/it | Denoise CUDA, s | Total without decode, s | Sampled baseline-relative GPU peak, GiB |
| --- | ---: | ---: | ---: | ---: |
| Preserved compiled Linear control | 63.442852 | 516.996156 | 596.6240 | 6.293381 |
| Local FF converted-weight reuse | 62.768249 | 510.155531 | 588.9593 | 5.920681 |

Observed reductions were 1.06% steady and 1.28% total (7.66 s total), with
0.372700 GiB lower sampled GPU delta. This single comparison does not establish
a reproducible speedup. Candidate sampled baseline was 1.026855 GiB; global RSS
peak was 25.477177 GiB. GPU observations are sampled baseline-relative deltas,
not allocation censuses. The BF16 references remain faster in steady state.

CPU-only inspection confirmed exact equality against the preserved FP8 output:
float32 video latents (1, 27280, 128), audio latents (1, 251, 128), maximum/mean
absolute difference zero, finite tensors and matching saved metadata. CUDA stayed
uninitialized during this inspection; decoded visual quality was not tested.

Actual reuse metrics: 192 configured projections, 768 FF scopes, 1536 conversions,
19968 reuse hits, peak retained weights exactly 268435456 bytes. Pending and
quarantined bytes were zero after removal; denoise completed and callbacks retired.
There were 384 event-specific waits totaling 467.962575 host seconds. These waits
overlap execution of queued GPU work and are already included in wall time; do
not sum them with denoise time or interpret them as removable extra overhead.
The single-FF-sized cap imposes frequent completion waits rather than retaining
a full-model BF16 cache. No default, allocator or protected runtime was changed.
Further capacity changes or benchmarks require separate approval.

Native fault failures, temporary fallbacks, cache admission denials and pressure
entries were zero; final source tickets were zero. Native timeline completed
with zero pending/failures/dropped entries and allocation diagnostics completed
without failure. These are selected-run observations, not arbitrary-model support.

### Scoped FP8 operator diagnostic: compute attribution (2026-10-09)

The separately approved diagnostic completed with exit 0, keeping the same
10-second workload, 256 MiB retained-weight ceiling and native budgets, but using
three denoise steps and two profiler captures of `transformer_blocks.0`.
Evidence: `E:/LTX-2/outputs/fp8_chunk_reuse_operator_diagnostic/metrics/modular_run.json`
and `operator_traces/operators_invocation_02.json` in that output directory.
Operator diagnostics report complete, two samples and benchmark_eligible=false.
The first capture includes compilation; use the third-invocation capture for
the following attribution, not the instrumented pipeline timing as a benchmark.

The third capture contained 1222 CUDA activity events, including 1179 kernels.
Sum of kernel durations was 1.286211 s; this is a sampled kernel-duration sum,
not an exclusive pipeline-wall-time decomposition or a driver-paging census.
Attention operator device duration was 0.641213 s (six calls), with the two
largest attention kernels totaling 0.636344 s. Matrix multiplication operator
device duration was 0.484956 s (86 calls). These CPU-operator device aggregates
refer to underlying kernels: never add operator and kernel values together.

GPU child annotations placed video self-attention at 0.789409 s and video FF
at 0.254316 s, both inclusive of their projections/other child kernels.
The identified fused conversion kernel family (`triton_poi_fused__to_copy_mul_t_0`)
totaled 0.002862 s over 34 calls; this name-based identification is not a census
of every possible quantization-related operation. The two compiled add families
totaled 0.016762 s over 86 calls. Aggregate attention and matmul costs dominate
the sampled block, explaining why FF conversion reuse gave only a small gain.
This is not evidence that an attention-backend switch would help or that the
same costs explain a difference against an unmatched external implementation.

The audio FF CPU range included a 1.223307 s cudaEventSynchronize call; its GPU
annotation was only 0.001324 s. Waiting in that range retires queued earlier GPU
work under the retained-weight cap, not 1.22 s of audio FF computation or extra
time to add to the kernels. Reported reuse totals were 288 scopes, 576 conversions,
7488 hits, 144 event waits and zero pending/quarantined bytes after retirement.
Native faults, temporary fallbacks, source-cache admission denials, pressure
entries and final source tickets were zero. Timeline completed without pending,
failed or dropped entries; allocation diagnostics completed without failure.
No additional benchmark, budget/backend change or default promotion follows.

### Five-second FP8 group-offload control: unsupported pinning (2026-10-09)

The separately approved A/B sequence stopped in its first case, before the first
denoise step completed. Evidence:
`E:/LTX-2/outputs/fp8_5s_group_control/metrics/modular_run.json` (status failed,
exit 1). The standard Diffusers loader successfully loaded the ten-shard FP8
checkpoint and the explicit recipe guard passed. Encoder/connectors/VAEs completed
on official group routes, then transformer leaf-level group offloading installed.
No native allocator, FP8 compute compilation or scoped weight reuse was requested.

The first transformer projection triggered group offload's
`_pinned_memory_tensors`: it calls `tensor.pin_memory()` for its streamed CPU
parameter dictionary. TorchAO 0.18 Float8Tensor dispatch refused
`aten.pin_memory.default` with NotImplementedError. This is an incompatibility
of the selected streamed group/TorchAO route, not OOM, failed FP8 reconstruction,
model-forward failure or native DDO evidence. Source inspection confirms the
selected group implementation avoids that pinning branch when use_stream=false;
that alternative has not been executed or accepted for this workload.

Partial total was 45.0354 s, sampled global GPU delta 2.251953 GiB and RSS
25.983688 GiB; these are failed-run/setup observations, not denoise benchmarks or
capacity limits. There are no matched throughput numbers or saved video latents.
The native second case was not started, honoring stop-on-first-failure approval.
No Diffusers/TorchAO patch, stream-setting change or retry was performed. The
working native FP8 route remains unchanged; any amended comparison requires
renewed approval and explicit labeling of the offload stream difference.

### FP8 five-second streamless group control — failed (2026-10-09)

The separately approved amendment kept the same checkpoint, 1280x704, 5 s at
24 fps, eight steps, seed 43 and local forward options. Only transformer official
leaf-group `use_stream` changed to false; auxiliary events confirm true. Evidence:
`E:/LTX-2/outputs/fp8_5s_group_no_stream_control/metrics/modular_run.json`.

The first projection failed before completing step one: standard Float8Tensor
CPU-to-CUDA `.to()` reached TorchAO alias correction and PyTorch rejected assigning
CPU storage to a CUDA tensor. This bypassed pinning but did not establish a usable
control. No OOM was reported. Partial total 31.8042 s, GPU delta peak 2.251953 GiB,
RSS peak 25.983658 GiB are setup/failure observations, not throughput results.
No saved latents/parity or matched speed numbers are available. The native second
case was not run, respecting stop-on-first-failure approval. Existing native,
BF16/image and allocator defaults were not changed; no backend dispatch or
dependency patch was introduced. Both failures apply to the selected installed
stack, not all official group/TorchAO versions or platforms.

### Isolated FP8 CUDA transfer diagnostic — passed (2026-10-09)

`E:/LTX-2/outputs/fp8_transfer_probe_cuda_01.json`: ordinary CUDA, Torch
2.12.1+cu132 / TorchAO 0.18.0, one freshly quantized 16x16 Linear, no LTX or DDO.
Dense control and four parameter/data default/copy=True transfers passed; FP8
bytes/scales were exact, outputs finite, synchronization successful. Peak PyTorch
allocation 8,524,800 bytes is a tiny diagnostic, not generation capacity evidence.
The runner failure was not reproduced: fresh quantization and transfer invocation
do not cover the saved-checkpoint/official-hook lifecycle. Do not claim copy=True
fixes the failed group control or that generic CUDA transfer is broken. No
benchmark retry, dispatch patch or accepted-runtime change followed this result.

### Saved FP8 official-hook diagnostic — passed (2026-10-09)

`E:/LTX-2/outputs/fp8_saved_group_cuda_01/report.json`: one saved/reloaded 16x16
Diffusers Linear, standard loader and official streamless leaf hooks. Direct
group-style transfer and two forwards passed with exact CUDA-reference outputs
and restored CPU FP8 bytes/scales. Peak PyTorch allocation was 8,525,312 bytes.
No DDO runtime, dependency patch or full-model retry. The LTX failure remains
unreproduced; host `torch.inference_mode()` was not covered by this initial probe.
This is a diagnostic limitation, not proof the failed comparison now works.

### FP8 hook inference-context reproduction — failed (2026-10-09)

`E:/LTX-2/outputs/fp8_saved_group_inference_cuda_01/report.json` reuses exactly
the passing saved-checkpoint fixture hashes. Direct transfer outside
inference_mode passed; the first official streamless hook forward inside
torch.inference_mode reproduced the full runner's CPU/CUDA storage-alias error
in TorchAO `aten.to.dtype_layout` / PyTorch `_functionalize_unsafe_set`.
The selected hook contract therefore has an isolated execution-context trigger,
not an established native allocator, LTX-forward or capacity defect. The streamed
pin_memory failure remains separate. No runtime changes or full-model retry
followed this probe. A group-control-only no_grad context is a proposed workaround,
not yet a validated generation baseline or fix for streamed offload.

### FP8 five-second official group/native A/B — completed (2026-10-09)

Matched saved FP8 checkpoint/local transformer: 1280x704, 5 s at 24 fps,
121 frames, eight steps, seed 43, real prompt/image, no source audio but generated
audio active, latent-only output. Same 1024-token FF/normalization chunks,
phase/compact modulation, residual/RoPE reuse; no FP8 compilation or chunk weight
reuse. Evidence: `E:/LTX-2/outputs/fp8_5s_group_no_grad_control/metrics/modular_run.json`
and `E:/LTX-2/outputs/fp8_5s_native_control/metrics/modular_run.json`.

| Measurement | Official leaf group, no stream/no_grad | DDO native, inference_mode |
| --- | ---: | ---: |
| Transformer load/prepare + attach (s) | 15.7939 + 0.2060 | 34.3141 + 12.3416 |
| Denoise phase (s) | 411.2244 | 235.2354 |
| Total tracked execution, latent output (s) | 443.4385 | 316.2860 |
| Sampled global GPU delta peak (GiB) | 2.283203 | 5.523609 |
| Sampled absolute driver GPU peak (GiB) | 3.3101 | 6.5505 |
| Sampled global process RSS peak (GiB) | 27.634579 | 25.981510 |

Native denoise time fell 42.7963% (1.7481x speedup); total fell 28.6742%, saving
127.1525 s despite greater setup cost. Single selected A/B, not a claim against
working streamed FP8 group offload or repeated-run statistics. The control uses
no_grad to avoid the reproduced alias defect; native retains inference_mode.
Auxiliary components use official streamed hooks in both, with native global
allocator installation an intended difference. No per-step CUDA timer: phase/8
is not steady-state. Total excludes VAE decode/MP4 encoding.

Saved Float32 video `[1,14080,128]` and audio `[1,126,128]` latents and metadata
were exactly equal in CPU comparison, finite, max/mean absolute error and relative
RMS zero. Native fault failures, temporary fallbacks, source-cache admission
denials/pressure entries, source-read failures and quarantine bytes were zero;
132 source tickets collected to zero. GPU deltas use the 1.026855 GiB baseline,
not a weight-residency measurement. Native kept the selected DLL hash unchanged,
1.5 GiB weight/staging and 24.625 GiB source/registration ceilings. This is selected
FP8 compatibility/performance evidence, not generic recipe support or BF16 quality
equivalence. No further benchmark or default change was made.
