# Native VMM support checkpoint (2026-10-03)

## Routes and evidence

`linear/auto` still uses the established linear preset; `vmm/auto` selects the
experimental fast native route and `vmm/nvme_baseline` selects the protected
legacy fallback. The allocator is a pinned Git submodule; native sources and
adapters live there. No candidate DLL was promoted during the gates below.
Windows CUDA is the initial validated platform; Linux validation is pending,
with no WSL/macOS VMM support promise. Ordinary installs do not require the
optional Windows `.[vmm-dev]` environment.

- Transformer: selected real-model three-generation reuse gate passed warm
  1.401990/1.397915/1.394595s/it. [Evidence](native-hardened-reuse-gate.md).
- Encoder: selected BF16/text-only prompt matches reference embeddings/mask;
  two calls and native unload passed. [Evidence](native-real-encoder-gate.md).
- Staged integration: encoder native → unload → transformer native → VAE/save,
  one generation, warm1.395602s/it, RunTracker total38.5135s. Includes separate
  setup/cold-step timing and memory/cache boundary checks, not process startup
  or pixel equality against a differently advanced RNG. [Evidence](native-staged-encoder-gate.md).
- Small robustness:11 CUDA contracts PASS in5.63s, no skips; cooperative
  interruption/reuse/unload, changed sources, absent current-context selection,
  eviction, registration reuse and staging refund. [Evidence](native-final-robustness-gate.md).

These are bounded selected contracts, not performance promises for every model
or arbitrary prompt/input length. Protected linear/NVMe routing/profile tests
passed on CPU; these latest gates did not repeat their performance benchmarks.
Neither an installed dependency nor a compiled candidate constitutes a passing
CUDA/performance gate. Historical workloads and different memory definitions
must not be compared as equivalent runs.

## Component integration contract

Native bootstrap must be the first executable action, through
`ddo_vmm_bootstrap.ensure_vmm_fast_process()` before importing/initializing CUDA
or constructing model payloads. Direct initialization remains diagnostic-only.
Runner arguments select backend/preset; users need not construct DDO_BACKEND.

The component-ownership route is explicit and experimental, not automatically
enabled by a text-encoder class name. Use the DDO weight-aware loader to preserve
immutable file descriptors, call eval, inspect loaded-source capacity, then
attach with native transport/full selection and fixed active/source/registration/
buffer limits. DDO owns source bindings, native transfers, events, aliases and
cleanup. Layout inspection is metadata, not GPU admission or calibrated residency.
No CPU weight retention/cache policy or architecture heuristic belongs in a host.

Shared parameters use one identity-preserving owner. Ordinary immutable tensors
and bounded buffers are required; backend-specific/quantized forwards must not
be rewritten and remain on compatible routes. Root overlap, unsupported tensors,
in-place embedding renormalization and unsupported payload converters are refused.
Initial managed units use one compute-stream contract; concurrent forwards/unload,
unmanaged streams, hidden escaping weight views, compiled-forward generality,
multi-GPU and poisoned-driver/device-loss recovery are not validated support.
The small absent-current-context test restores a healthy context, not a failed
driver. Native unload invalidates parameters to shared meta placeholders; reload
the component for further inference, rather than expecting restored CPU weights.

For memory-sensitive pipelines, consume output/readback, unload and delete the
encoder plus its pipeline references before loading transformer weights. Final
event collection and unload are lifecycle boundaries; do not add per-step
global synchronization or auto-expand ceilings. Full-pipeline linear/compat
examples are not proof that simultaneous native components fit the same budget.

## Reporting boundary

Hosts consume the public DDO report rather than hook state or native attribute
names. After a completed component phase, the existing minimal protocol is:

```python
import json

report = result.metrics_snapshot(collect_completed=True)
if args.show_profile:
    print(report.format_profile())
if args.show_metrics:
    print(json.dumps(report.as_dict(), indent=2))
# Save result/report through the host's normal JSON writer using report.as_dict().
```

`collect_completed=True` collects completed native source tickets without a
per-step synchronization; it does not unload the module. Use DDO cleanup in a
finally block and preserve the original forward error if cleanup also fails.
The sibling RunTracker accepts the same report with
`tracker.record_ddo_metrics(report, component="text_encoder")` and serializes
it without inspecting internal metrics; its show_metrics setting controls the
profile print. [Metric catalog and host observation definitions](metrics.md)
describe units, collection points, interpretation and diagnostic actions.

## Remaining delivery work

User-approved scope update: Windows development acceptance closes on the
selected performance and critical robustness evidence above. Do not add an
open-ended longevity/model/environment matrix or chase the remaining small
performance difference. New critical defects or lifetime changes justify
targeted tests; other models/environments will receive user-driven validation
and bug reports. Encoder ownership remains explicit opt-in. This acceptance
does not promote a DLL, promise arbitrary-model safety or finish publication.

Critical protections retained: separate fixed host storage/registration/GPU
ceilings, no early byte/slot/backing release before completion, ownership retained
on unproven cleanup, cooperative cancellation/unload and clear failure metrics.
A failed/poisoned CUDA context must not be treated as successfully recovered;
no device reset or speculative remapping strategy is authorized by this closure.

Next implementation block: [native Linux portability](native-linux-port-plan.md).
The current file source/cache/transport still has Windows-only branches;
compiling a `.so` alone cannot run the accepted native route.
Use the [Linux continuation checklist](linux-handoff.md) for a new session:
separate `.venv-linux`, CPU-only native contracts, explicit library build
selection and the remaining source/cache/DMA block. Preparation passed seven
shared C++ CPU contracts on Windows and66 Python recipe/routing contracts;
the Linux-only POSIX contract and actual `.so` remain unvalidated.

Reporting-only precision/route labels passed82 CPU tests in5.09s and syntax
compilation, without a new CUDA run. The recorded real-model artifacts predate
those additive fields and are not rewritten. Final release validation of the
chosen source/submodule revision/build, Linux portability/user support checks and
publication audit remain separate work. No extra Windows benchmark is scheduled
by this release wording. Do not rewrite private history, remove compatibility,
publish or promote a candidate without the existing explicit scope/approval.
Any provenance audit must retain truthful licensing/attribution obligations;
removing references does not establish independent implementation or license
compatibility. No such cleanup or publication was performed at this checkpoint.
