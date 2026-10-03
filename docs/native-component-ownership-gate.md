# Opt-in component ownership gate

Latest source: nested-owner consolidated CUDA gate PASS (2026-10-03): seven
tests passed in 5.78 s using the exact nested command below and recorded lifecycle
OFF DLL hash. CPU regression previously passed 153 contracts in 6.23 s.
Nested non-root owners are coalesced into their enclosing execution unit;
root direct-parameter/descendant overlap remains refused. No budgets/DLL change.
Actual encoder capacity, loaded-source binding and output validation remain pending.

## Nested-owner gate (approved and executed: PASS)

Seven selected CUDA cases in one process: two nested-owner exception variants,
two previously validated component variants, two default interrupted-forward
variants and one default 128-forward/four-reload case. All seven passed in 5.78 s.
No automatic retry, ceiling expansion, rebuild, DLL promotion or benchmark.

The nested fixture actually computes using a parent's direct projection
Parameter followed by a child normalization. It interrupts the parent after
projection, before normalization, rather than merely attaching an unused branch.
Shared embedding/head stays one binding; all 12 unique parameter groups must
be file-backed, and component metadata must show five disjoint enclosing units.

Hypothesis: enclosing geometry preserves BF16 output and all native lifetimes
across parent/child execution, RuntimeError/KeyboardInterrupt, reuse and unload,
without regressing existing component/default contracts. Numerical tolerance
remains rtol/atol 0.016. Same ceilings and DLL as the passing boundary gate:
0.07 GiB active weights, 2 MiB staging/pages, 64 MiB source cache/registration,
1 MiB component buffers. No budget recalibration, large model or performance run.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
$env:TEMP='E:\ProjetosIA\diffusers-dynamic-offloader\.tmp-component-ownership-cuda-nested-20261003'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_component_ownership_cuda.py::test_nested_parameter_parent_numerical_reuse_interrupt_unload tests/test_vmm_component_ownership_cuda.py::test_component_shared_weights_norms_buffers_reuse_cancel_unload tests/test_vmm_forward_cancel_cuda.py::test_interrupted_forward_resumes_then_unloads_without_readback tests/test_vmm_model_lifecycle_cuda.py::test_generic_model_128_forwards_with_unload_reload_and_source_replacement -q -x -p no:cacheprovider --basetemp .pytest-component-ownership-cuda-nested-20261003
```

Acceptance: seven PASS, matching reference BF16 outputs for three input lengths
and repeated calls; shared aliases intact, no retained original loader Parameter,
zero CPU-source bindings, stable source reads after warm use, no fault failure,
temporary fallback, source admission denial/bypass, collected transfer owner leak
or capacity overrun. Both interruption types resume and immediate interrupted
unload retires aliases, removes boundaries/hooks, closes source registry and
permits checkpoint deletion. Stop at first failure or memory anomaly; no automatic
retry or scope expansion. Real encoder validation remains a separate later gate.

## Earlier checkpoints

Checkpoint: corrected consolidated CUDA gate PASS (2026-10-03): 5 passed in 5.74 s,
using the exact boundary command below, DDO .venv-vmm and the recorded lifecycle
OFF DLL hash. Both component interruption variants and all three default-route
regressions passed. No retry, ceiling expansion, rebuild or DLL promotion.

The small BF16 component passed reference comparison, shared embedding/head
identity, native normalization parameters, bounded device buffer, original CPU
Parameter release, nine file-backed/zero CPU-source bindings, stable warm source
reads, zero fault failures/temporary fallback/source denial/bypass, completion
collection and immediate interrupted unload with checkpoint deletion. Default
route interruption cases and 128-forward/four-reload regression also passed.
This proves the selected small CUDA contracts only, not real encoder capacity,
full-pipeline bootstrap, performance, driver-loss recovery or platform expansion.

Historical second approved CUDA attempt PARTIAL/FAILED: 1 passed, 1 failed in
5.73 s. Native RuntimeError component reuse/numerical/unload passed. The
KeyboardInterrupt variant failed on resumed execution because Torch's
always_call exception handler catches Exception only, leaving the unit active;
unload consequently refused backing release. Three remaining CUDA cases were
not executed. No passing consolidated gate or encoder/performance claim.

Correction (2026-10-03): guarded opt-in units now wrap the unchanged forward
with a BaseException-aware execution boundary, including preamble cleanup and
primary-error preservation. Default-route hook pairs remain unchanged. CPU
regression 152 PASS in 6.12 s includes RuntimeError/KeyboardInterrupt/SystemExit
fencing, resume, reentry and boundary removal. Corrected CUDA retry below still
requires fresh approval; no rebuild, DLL promotion or ceiling change.

First approved CUDA attempt FAILED; stopped at the first test, 1 failed
in 5.16 s. The test accidentally selected the default encoder Diffusers group
route rather than native attachment; its old-Parameter release assertion failed
before numerical execution. The four remaining CUDA cases were not executed.
The fixture now explicitly requests `route='dynamic_offload'` and asserts the
selected route/hook. Expanded CPU regression: 149 PASS in 5.98 s. No allocator,
default preset, DLL, ceiling or benchmark change. Corrected CUDA retry is pending
fresh approval, not covered by the first execution authorization.

## Integration scope

`vmm_component_tensor_ownership=True` dispatches the public allocator attachment
to a component adapter using the existing native arena, immutable source batches,
fault/signature/unpin and execution slots. Each identity-unique parameter has one
binding and all direct aliases share its Parameter. A shared embedding/output
Linear therefore has one range consumed by two execution units. Normalizations,
convolutions and other direct parameters are not silently left on CPU. No weight
forward is replaced, no model-class condition or CPU weight cache is introduced.

The first experimental contract deliberately has no speculative lookahead;
validated default Linear lookahead/residency remains unchanged. Native admission
and slot geometry retain the requested GPU ceiling, without automatic expansion.
Buffers are copied once to device under a separate explicit byte ceiling, default
1 MiB, not under a newly inferred model residency budget. Parameter and buffer
source ownership is dropped after successful publication. Retirement replaces
all aliases with shared meta placeholders only after final-consumer completion;
the unloaded model is unusable until loaded again, not transparently restored.

Preflight requires eval mode, full selection `module_pattern='.*'`, immutable
file descriptors for every CPU contiguous nonempty parameter, native file
transport, supported ordinary tensors, buffer capacity and disjoint execution
units. The outermost ModuleList item encloses descendant parameter consumers;
standalone owners are separate units. Nested non-root parameter owners are
enclosed in the outer parameter-owning unit, still subject to native ceilings.
Root overlap and in-place embedding renormalization are refused before allocation.
Reentrant units are refused without retiring an enclosing active invocation.
Observable managed-weight output views are refused after lifetime cleanup;
custom forwards must not stash or consume escaping weight views elsewhere.
Dynamic parameter/buffer replacement, training, concurrent forward/unload,
unmanaged extra streams, multi-GPU/context and driver failure recovery are not
supported by this initial component contract. Full encoder numerical support is
pending, not inferred from module type or these CPU observations.

Public config is opt-in only; standard DDO argument helper adds
`--ddo-vmm-component-tensor-ownership` and
`--ddo-vmm-component-buffer-limit-mb`. Protected linear/NVMe backend builders
reject the component flag. The sibling LTX harness/default encoder is not changed.
Hosts consume `metrics_snapshot()`, with `runtime.component_ownership` metadata,
not the adapter's binding/ownership internals. Metric catalog: [metrics](metrics.md).

## Approved corrected consolidated CUDA execution (PASS)

Hypothesis: the shared native range is safe for embedding and output use, native
unit scope includes normalization, buffer ownership survives interruption, and
cleanup can close descriptors/backing without breaking aliases or existing
small default-route reuse contracts. This is not a performance hypothesis.

One process, stop on first failure (`-x`): two new component exception variants,
two existing interrupted-forward variants, and the existing 128-forward/four-
reload default-route regression. Five selected tests; no LTX model/download.
DDO `.venv-vmm`, existing lifecycle OFF DLL (no rebuild):
SHA256 `755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729`.
Small tests use ordinary Torch allocations around native arena contracts, not
the fresh-child global allocator/full-pipeline benchmark contract.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
$env:TEMP='E:\ProjetosIA\diffusers-dynamic-offloader\.tmp-component-ownership-cuda-boundary-20261003'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
& .\.venv-vmm\Scripts\python.exe -m pytest tests/test_vmm_component_ownership_cuda.py tests/test_vmm_forward_cancel_cuda.py::test_interrupted_forward_resumes_then_unloads_without_readback tests/test_vmm_model_lifecycle_cuda.py::test_generic_model_128_forwards_with_unload_reload_and_source_replacement -q -x -p no:cacheprovider --basetemp .pytest-component-ownership-cuda-boundary-20261003
```

Limits: 0.07 GiB native active weights, 2 MiB aggregate staging, 2 MiB pages,
64 MiB source-cache/registration ceilings; component buffer ceiling 1 MiB.
No ceiling escalation, automatic retry, larger model or benchmark is approved by
this proposal. Stop on the first failed test or memory anomaly.

Acceptance: all five tests pass, CUDA reference outputs within BF16 rtol/atol
0.016 in the new gate; aliases remain identical; all parameters/device buffer
actually on CUDA, no retained loader Parameter, nine file-backed and zero CPU
parameter-source bindings. Source reads stable after initial use, no fault,
temporary fallback, source denial/bypass, event-owner leak or capacity overrun.
RuntimeError/KeyboardInterrupt interruption resumes correctly, immediate
interrupted unload clears hooks, retires aliases to meta, closes registry and
permits checkpoint deletion. Existing default-route contracts must still pass.

Explicit node selection excludes the three CPU-only fixtures in the older files;
the original whole-file command would also have selected them if it had continued.
No cases or limits are added to the five proposed CUDA contracts.

After approval/run, record the actual result here and in the benchmark sidecar.
Do not interpret compilation/CPU PASS as permission to execute this command.
