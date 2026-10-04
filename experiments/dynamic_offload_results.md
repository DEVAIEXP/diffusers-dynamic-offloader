# DDO native VMM benchmark results

Selected development acceptance, updated 2026-10-04. DDO is model-agnostic;
LTX is the benchmark workload, not a library dependency or architecture policy.
Intermediate validation runs are retained locally in the internal implementation
report, separately from this public summary. Recorded measurements and raw
benchmark artifacts are unchanged.

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
