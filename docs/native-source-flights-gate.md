# Bounded multi-window source submission (2026-10-02)

Executed once after explicit confirmation: eight CUDA cases PASS (5.13 s),
span probe PASS, model exit 0. Functional/concurrency criteria PASS, performance
criterion FAIL: Generate 24.8827 s exceeds 22.5 and the preserved milestone.
Warm 1.423592 s/it passes <=1.46. No promotion/retry; approval exhausted.
Isolated candidate `build-source-flights-20261002/Release/ddo_vmm.dll`, SHA256
`A958C53EB1FFC4CD2B243923373C72C918EA8341CB198C0353BF7A05941EA8EC`.
Dispatcher option remains OFF by default; protected/milestone binaries intact.

## Correction and CPU evidence

Dispatcher submission no longer waits for the previous whole window. Its fixed
flight array retains up to min(cache slots, existing source workers, eight)
submitted Work/Entry owners. No new I/O threads, native source bytes, host/GPU
ceiling, registration policy or runner scheduling. Demand priority still comes
from the existing scheduler; native chunk queue/backpressure remains 256 chunks.

Ready windows are independently drained/retired, without oldest-window blocking.
When there are active reads, native CPU polling checks at most eight futures
and waits up to 1 ms between idle checks; a newly queued demand wakes it. No
CUDA query/synchronization is used for source completion observation. This may
add bounded retirement latency; the performance gate must measure it, not
assume benefit from concurrency. Partial/exceptional enqueue keeps its owner
until all chunks finish. Stop cancels queued metadata, drains submitted reads,
then joins before storage release. Reaping retains ownership if retirement
throws. Host cache ownership and GPU residency remain distinct.

ON/OFF Release builds compiled. Selected Python CPU suite: 75 PASS in candidate
(4.76 s) and OFF control (4.73 s). New flight CPU contract: 32 lifecycle
iterations plus eight-flight bound, zero failures; checks two pending windows,
unordered completion, partial failure retention, demand reuse, cancellation,
throwing retirement and close. Existing read-wait 32 lifecycle iterations,
scheduler 169 checks, span ledger 1,278 checks and stream dependencies PASS.
These metadata/mock contracts are not real I/O failure injection or GPU proof.

Optional public source-cache snapshot fields document owned windows, peak since
clear and flight limit. Peak >1 proves retained multi-window submissions, not
simultaneous disk activity. No ABI struct reinterpretation; older DLLs omit fields.
CPU covers normalization/absent exports and actual null rejection before CUDA.

Eight small CUDA cases collected ONLY, not run. New ten-window case exceeds
the eight-flight bound with 20 MiB total payload, checks two correct passes,
retained file owner, registration recycling, no reread/bypass, bounded peak and
clear/reset. Tiny reads need not reach peak eight; the model checks peak >1.

## Observed model result

Window peak = limit = eight; owned windows 8 before denoise -> 0 afterward.
This confirms retained multi-window submission actually happened. It does not
prove simultaneous disk activity. Source preparation 0.028299 s; DDO prepare
1.6399 s, first GPU step 6.071087 s, GPU phase 16.036243 s, host denoise
15.3484 s, Generate 24.8827 s, total 46.1006 s. Previous single-window candidate
Generate 24.4543 s and milestone repeats 23.2125/23.2392 s remain better
observations; one run does not establish causal regression or variance bounds.

Before denoise completed source bytes 4,353,687,552; committed storage
26,038,239,232 bytes, source registrations/DMA zero. Final source bytes
25,846,148,360; consumer readiness wait 4.974880 s (previous 5.093906),
registration 0.517319 s, DMA 187,588,136,576 bytes; weight backing 3,858,759,680
unchanged. These nested cumulative timers are not additive/exclusive timings.
Increasing retained concurrency alone did not substantially remove cold wait.

Image hash identical `877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Tickets 736 -> 0 nonblocking; zero native failures/fallback/pressure/mismatch/
quarantine/pool failures. Evictions 53 -> 53 and wait 0.1299724 s unchanged;
pool accounting conserved before/after. Peak GPU-used delta 6.5024 GiB (not
absolute GPU usage), RSS 29.5051 GiB; no monitor errors. Candidate selected/hash
verified, protected/milestone hashes unchanged. Artifacts in sibling host
`outputs/ltx_image_modular_vmm/source_flights_20261002`.

This closes the hypothesis that merely removing the whole-window barrier is
sufficient for the target in this case. Do not diagnose that already-fixed
barrier again or increase flight/memory limits without new evidence. Further
work must identify the remaining cold critical path before another runtime
patch; no new gate or implementation is authorized by this finished sequence.

## Commands documenting the executed approval (not authorization to rerun)

Hypothesis: bounded multi-window submission removes the known whole-window
barrier and reduces cold source readiness/whole Generate without changing warm
throughput or memory bounds. Previous unlocked single-window Generate 24.4543 s
failed; compare against accepted milestone repeats 23.2125/23.2392 s, not just
against the failed candidate. Do not count time shifted into prepare as a gain.

Execute eight small CUDA cases, then span probe, then ONE unchanged model case.
Stop at first failed/skipped required gate or memory anomaly. No rerun/default
promotion/parameter adjustment. DDO Python; real prompt BF16, eight steps,
1280x704/seed43, weight 4 GiB/effective staging 1.5 GiB, source/registration
24.625 GiB and allocator cache 512 MiB unchanged.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$python = 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$env:PYTHONPATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-source-flights-20261002\Release\ddo_vmm.dll'
$env:DDO_VMM_TORCH_CACHE_MIB = '512'
& $python -m pytest ddo-vmm-allocator/tests/test_source_dispatcher.py ddo-vmm-allocator/tests/test_source_cache.py::test_persistent_storage_survives_registration_recycling_without_reread ddo-vmm-allocator/tests/test_native_ranges.py::test_host_ring_native_file_read_ticket_feeds_async_dma ddo-vmm-allocator/tests/test_native_ranges.py::test_host_ring_ticket_copies_each_completed_source_slot_to_arena ddo-vmm-allocator/tests/test_native_ranges.py::test_host_ring_native_file_ticket_uses_positioned_2mib_chunks -x -q -p no:cacheprovider --basetemp ddo-vmm-allocator/build-source-flights-20261002/pytest-cuda-gate
if ($LASTEXITCODE -ne 0) { throw 'Source/ring CUDA gate failed; stop' }
# Require eight PASS with no skip before proceeding.
& $python ddo-vmm-allocator/scripts/vmm_span_pool_probe.py --library $env:DDO_VMM_LIBRARY_PATH
if ($LASTEXITCODE -ne 0) { throw 'Span gate failed; stop' }

Set-Location E:\ProjetosIA\ltx2.3-image-diffusers
$env:HF_HOME = 'F:\hf_cache'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
& $python run_dynamic_modular_vmm_distilled.py `
  --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto `
  --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5 `
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --no-pag --local-files-only --metrics-level 2 `
  --memory-monitor rss --ddo-gpu-timing `
  --output-dir outputs/ltx_image_modular_vmm/source_flights_20261002
if ($LASTEXITCODE -ne 0) { throw 'Model gate failed; stop' }
```

Acceptance: correct small payloads/lifetimes/clear-restart/fallback/recycling;
model image SHA256 `877876A4...`, selected candidate and complete timing,
zero native errors/unexpected fallback/quarantine/pressure/plan mismatch,
final tickets and dispatcher owned windows zero, 1 < peak windows <= limit = 8,
no denoise pool eviction/wait delta, conserved accounting and no monitor error.
Warm <=1.46 s/it and whole Generate <=22.5 s. Record prepare, first GPU step,
source-read/readiness/registration/DMA, window peak and memory; no exclusive
timing attribution or whole-plan closure from a single passing case.
