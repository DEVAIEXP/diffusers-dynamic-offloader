# Consolidated native robustness regression (PASS, 2026-10-03)

## Preserved milestone and scope

The selected real-prompt encoder-to-transformer staged gate passed at warm
1.395602s/it and RunTracker total38.5135s. This regression does not change that
implementation, DLL, ceilings, encoder opt-in default or protected runtime routes.
No repeated full-model benchmark is proposed. See
[staged outcome](native-staged-encoder-gate.md).

Preparation on2026-10-03:200 CPU tests PASS in5.71s, including preset/backend
routing, profile isolation, immutable loaded sources, component/alias ownership,
pending-event retention, partial cancellation, first-error preservation,
source-capacity config, timing and source-lifecycle ABI. CUDA remained
uninitialized in the actual ABI-null-handle test. Initial attempt stopped with
198PASS/1FAIL in6.73s because DDO_VMM_LIBRARY_PATH was omitted; corrected explicit
candidate selection passed the same suite. No production fix was needed.
CPU assertions are not hardware/performance proof. Protected linear/NVMe route
coverage here is configuration/mocked contracts, not a fresh performance gate.

## Hypothesis

The same candidate safely preserves native byte/event/alias owners across
cooperative interruption, repeated forwarding, source replacement, temporary
loss of current-context selection, GPU eviction and host-registration recycling.
Released staging capacity becomes available again without auto-expanding ceilings.
These are existing small contracts, consolidated after the staged milestone;
no new fault injection, device reset, GPU oversubscription or driver crash.

## Executed one-shot CUDA command

Eleven selected cases: four component variants (ordinary/nested owners with
RuntimeError/KeyboardInterrupt), two default-route interruption variants,
one128-forward/four-reload contract, one missing-current-context contract, and
three cache contracts. Run once and stop on the first failed case (`-x`).
DDO Python; explicit unchanged lifecycle OFF candidate. The preparation did not
collect/run CUDA; the command below subsequently ran once after user approval.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
if ((Get-FileHash -LiteralPath $env:DDO_VMM_LIBRARY_PATH -Algorithm SHA256).Hash -ne '755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729') { throw 'Unexpected DLL hash; stop gate' }
$env:TEMP='E:\ProjetosIA\diffusers-dynamic-offloader\.tmp-final-robustness-cuda-20261003'
$env:TMP=$env:TEMP
if (Test-Path -LiteralPath '.pytest-final-robustness-cuda-20261003') { throw 'Gate artifacts already exist; do not overwrite or retry' }
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_component_ownership_cuda.py tests/test_vmm_forward_cancel_cuda.py::test_interrupted_forward_resumes_then_unloads_without_readback tests/test_vmm_model_lifecycle_cuda.py::test_generic_model_128_forwards_with_unload_reload_and_source_replacement tests/test_vmm_context_recovery_cuda.py::test_missing_context_preserves_owners_then_upload_and_cleanup_recover ddo-vmm-allocator/tests/test_source_cache.py::test_persistent_storage_survives_registration_recycling_without_reread ddo-vmm-allocator/tests/test_source_cache.py::test_cached_source_survives_gpu_eviction_without_reread ddo-vmm-allocator/tests/test_source_cache.py::test_staging_ring_consumes_and_refunds_cache_admission_capacity -q -x --tb=short -p no:cacheprovider --basetemp .pytest-final-robustness-cuda-20261003
if ($LASTEXITCODE -ne 0) { throw 'Robustness gate failed; stop sequence' }
```

## Fixed limits and acceptance

- All11 cases PASS, no skips; unexpected failure or visible memory anomaly stops
  the sequence. No retry, parameter adjustment or full-model run follows.
- Component/default-model tests keep0.07GiB active,2MiB staging/pages,64MiB
  source/registration; component buffers bounded at1MiB. Reference tolerances
  remain BF16 rtol/atol0.016 for component cases and the existing FP32 tolerances
  for default cases. Original loader Parameters released, shared aliases intact,
  all managed parameters on CUDA, no CPU source bindings.
- Cooperative exceptions resume numerically correctly; immediate unload clears
  hooks, retires managed aliases and closes source registry, allowing deletion.
  Default-route128 forwards/four source generations preserve warm reads and
  bounded resources. Healthy cases have no unexpected fault failure, temporary
  fallback, source denial/bypass or collected transfer owner/ticket leak.
- Context test uses2MiB page/cache and6MiB process source ceiling. Temporarily
  detaches current-context selection without destroying/resetting a context and
  makes no Torch calls while detached. Rejected close/clear retain handles and
  owners; restoring the same context permits cleanup and another correct DMA
  without source reread. This is not recovery from device loss or a poisoned
  CUDA driver/context.
- Registration recycling retains4MiB source bytes with2MiB registration limit;
  unregistration preserves bytes, repeated DMA matches source without reread,
  clear zeros committed/registered bytes. GPU eviction test has2MiB residency
  in a4MiB arena/cache; evicted weight rematerializes correctly using retained
  source bytes. It does not establish WDDM migration or performance.
- Staging-admission case has4MiB process source limit: a2MiB ring plus a2MiB
  admitted cache segment intentionally denies another segment once; ring close
  refunds capacity and demand admission succeeds. That one asserted denial is
  expected, unlike normal model-forward denial/bypass, and is not a failure.

The small suite uses ordinary Torch allocations around native contracts, not
fresh-child global allocator installation. The preceding real staged gate is
the separate evidence for that bootstrap path. No protected linear/NVMe CUDA
benchmark is included; no unsupported cross-platform/general-model promise.

After approval, record exact executed counts, duration and failure details here
and in the experiment sidecar. DLL promotion, compatibility removal, history
rewrite, publication, larger ceilings and real-model fault injection require
separate authority. Default reporting booleans/baseline persistence and final
documentation remain follow-up work, not silently included in this gate.

## Executed result

Approved sequence completed once: **11 PASS in5.63s**, no skips, exit0. No
production changes, rerun, benchmark or budget adjustment. Candidate SHA256
755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729 and
protected default SHA256
4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3 verified
unchanged after execution. Test output reported no failure; this small suite
has no separate sampled process-wide RAM/GPU peak monitor.

All existing numerical/alias/public-resource assertions passed in the four
ordinary/nested component variants and the two default interrupted-forward
variants, including resumed forwarding and immediate interrupted unload.
Default128-forward/four-reload observations were inspected: source reads8448/
18816/8448/18816 bytes match the changed source generations and remain identical
from invocation0 to31 in each cycle; committed/registered source bytes stay
67,108,864 in each attached cycle. Collected tickets/transfer owners, CPU source
bindings, fault failures and temporary fallbacks are0. Default interrupted
warm/resumed reports retain12,672 source-read bytes and the same64MiB source
population, with those failure/owner counters0. Registry/hook cleanup and source
file deletion assertions passed, without a post-unload global GPU leak claim.

Missing-current-context selection rejects close/clear without dropping owners;
restored same context permits correct DMA/continued reuse/cleanup. Registration
recycling preserves committed native bytes without reread; GPU eviction reloads
correctly from retained source; staging admission intentionally denies the
second segment once and admits it after ring capacity is refunded. These passed
assertions do not imply poisoned-driver/device-loss recovery or WDDM migration.

Local test artifacts retained under `.pytest-final-robustness-cuda-20261003/`,
including `lifecycle_observations.json` and both `cancel_observations.json` files.
Small-contract robustness regression is closed for this selected scope. Prior
real staged performance result remains unchanged, not remeasured here. Linear/
NVMe route evidence remains the200 CPU contracts and their preceding validated
milestones; no new protected-route performance gate is claimed.

Next work: final public reporting/metrics and documentation consistency. Any
new CUDA execution, promotion, publication or history rewrite still needs its
own explicitly scoped approval; this execution authorization is exhausted.
