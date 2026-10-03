# Source completion without global admission locking (2026-10-02)

Executed once after explicit confirmation: seven CUDA cases PASS (4.94 s),
span probe PASS, model exit 0 and identical image. Performance criterion FAILED:
Generate 24.4543 s exceeds 22.5 and milestone repeats 23.2125/23.2392 s.
Warm 1.412124 s/it passes <=1.46. No promotion/retry; approval exhausted.
Candidate `build-source-unlocked-20261002/Release/ddo_vmm.dll`, SHA256
`DC4FAA4DE93B13F059E097F08EEA393163D394EFBE8E3EF0AA010D64589F828B`.
Source dispatcher remains default OFF. Protected/milestone binaries untouched.

## Implemented and CPU checked

Dispatcher builds use shared read completion. A waiter copies the completion
under the ring mutex, increments its slot waiter count, releases the mutex
for the blocking get, then retires under the mutex. Reuse, unregister and
storage release reject active waiters. Destroy rejects new source waits and
waits for existing waiters before destroying storage/mutex. Owners must stop
new API submissions before close; this is not a raw-handle lifetime manager.
Both success and exceptions retire waiters. Dispatcher-OFF retains its original
consumable future and locked read wait. No runner, ceilings or worker count changed.

Release builds ON/OFF compiled. Candidate Python CPU suite: 74 PASS (4.79 s).
Dispatcher-OFF CPU suite: 74 PASS (4.69 s); no CUDA baseline revalidation.
New CPU test runs 32 success/failure lifecycle iterations with two concurrent
waiters, foreground access to the same mutex while completion is pending,
close waiting for retirement, shared results/exceptions and fresh generation.
Zero failures. Scheduler 169 checks PASS, spans 1,278 checks PASS, stream
dependencies PASS. CPU contracts do not prove driver safety or speedup.

Hypothesis: removing the observed global lock serialization allows native
admission/source filling to overlap setup and compute, rather than moving
all cold work into prepare. The prior candidate's Generate 25.8173 s failed;
the accepted comparison remains milestone repeats 23.2125/23.2392 s.

## Executed model result

DDO prepare 1.6631 s (prior locked candidate 7.5827), source preparation
0.025003 s (prior 6.627046), first GPU step 6.082628 s, GPU denoise
15.967527 s, host denoise 15.3018 s, total 45.5254 s. The global admission
serialization was removed, but this does not establish whole-phase benefit.
Before denoise 4,278,190,080 source bytes had completed; all host storage was
committed (26,038,239,232 bytes), source registration/DMA still zero. Final
source read 25,846,148,360 bytes, consumer readiness wait 5.093906 s,
registration 0.430762 s, DMA 187,588,136,576 bytes. Nested timers are not additive.

PNG SHA256 `877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED`.
Tickets 732 -> 0, collected nonblocking. Zero native errors, temporary fallback,
pressure, plan mismatch, pool failure/quarantine; eviction count 53 -> 53 and
wait 0.1350962 s unchanged. Mapped accounting conserved. Peak GPU-used delta
6.3424 GiB (not absolute VRAM), RSS 29.5031 GiB; monitors no error.
Candidate selected/hash verified; protected and milestone binaries unchanged.
Artifacts: sibling `outputs/ltx_image_modular_vmm/source_unlocked_20261002`.

Read-only follow-up found the remaining explicit whole-window barrier in
`dispatch_reads`: submit one window, wait for its completion, acknowledge,
then take another window. Eight chunk workers do not imply multiple windows
in flight. Any next candidate must separate bounded submission from retirement
without releasing owners early; this observation alone does not prove the
barrier accounts for all remaining time or promise a speedup. No patch/new
CUDA sequence is authorized by this completed gate.

## Commands documenting the executed approval (not authorization to rerun)

Seven small CUDA cases (four source/cache cases plus three existing ring
contracts), then span probe, then ONE unchanged model run. Stop on any failure,
skip or memory anomaly. Do not change ceilings or run another case/repeat.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$python = 'E:\ProjetosIA\diffusers-dynamic-offloader\.venv-vmm\Scripts\python.exe'
$env:PYTHONPATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-source-unlocked-20261002\Release\ddo_vmm.dll'
$env:DDO_VMM_TORCH_CACHE_MIB = '512'
& $python -m pytest ddo-vmm-allocator/tests/test_source_dispatcher.py ddo-vmm-allocator/tests/test_source_cache.py::test_persistent_storage_survives_registration_recycling_without_reread ddo-vmm-allocator/tests/test_native_ranges.py::test_host_ring_native_file_read_ticket_feeds_async_dma ddo-vmm-allocator/tests/test_native_ranges.py::test_host_ring_ticket_copies_each_completed_source_slot_to_arena ddo-vmm-allocator/tests/test_native_ranges.py::test_host_ring_native_file_ticket_uses_positioned_2mib_chunks -x -q -p no:cacheprovider --basetemp ddo-vmm-allocator/build-source-unlocked-20261002/pytest-cuda-gate
if ($LASTEXITCODE -ne 0) { throw 'CUDA contracts failed; stop' }
# Require seven PASS, no skip, before proceeding.
& $python ddo-vmm-allocator/scripts/vmm_span_pool_probe.py --library $env:DDO_VMM_LIBRARY_PATH
if ($LASTEXITCODE -ne 0) { throw 'Span probe failed; stop' }

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
  --output-dir outputs/ltx_image_modular_vmm/source_unlocked_20261002
if ($LASTEXITCODE -ne 0) { throw 'Model gate failed; stop' }
```

Require correct payloads/clear-restart/registration recycling, bounded fallback,
safe ring/span lifetime; model PNG SHA256 `877876A4...`, complete timing,
final tickets zero, zero native failures/temporary fallbacks/quarantine, no
denoise eviction/wait delta, conserved accounting and no monitor anomaly.
Warm <=1.46 s/it and Generate <=22.5 s; include prepare in Generate and record
first GPU step, source preparation/readiness, registration, DMA and memory.
Do not accept a smaller first step alone. DDO Python, real prompt/BF16,
effective staging 1.5 GiB, other capacities and timing instrumentation unchanged.
No promotion, repeat or parameter adjustment is included in this approval.
