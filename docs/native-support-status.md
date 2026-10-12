# Native VMM support checkpoint (2026-10-03)

## Current addendum (2026-10-10)

Subsequent native Linux evidence on 2026-10-11: an isolated idle-trim candidate
compiled and passed its approved small CUDA ownership probe, while the prior
accepted `.so` remained unchanged. The primary host then passed the selected
LTX image/five-second video resident/leaf VAE pairs and the separately approved
Qwen Image 2.1 base BF16 512x512/20-step leaf/linear/VMM trio. Delivered output
hashes match within each selected workload comparison, with no reported native
or monitor failures. The native Qwen route owns the transformer only; its encoder
uses explicit Diffusers compatibility and its VAE is resident. No model-specific
DDO change, budget tuning or automatic idle trim was added. Technical candidate
identity is in native-linux-idle-trim-gate.md; pipeline evidence belongs to the
sibling host's docs/benchmarks/ pages and dated results. This does not validate
quantization, all presets, 1024 Qwen Linux, WSL, arbitrary models or driver loss.

The original BF16 Windows/Linux checkpoint below remains valid for its preserved
binaries and workloads. Subsequent video and FP8 evidence is recorded in the
[benchmark sidecar](https://github.com/DEVAIEXP/ddo-examples/blob/main/experiments/dynamic_offload_results.md).
Experimental saved TorchAO FP8 transport is model-agnostic and has selected small
CUDA and Windows real-model acceptance, including exact group/native latent parity.
It is not an LTX-only feature, automatic quantized admission, a general model claim
or Linux FP8 validation. See [FP8 contract and boundaries](fp8-weight-only.md).
LTX-specific compute/pipeline changes remain in the sibling host, not native
residency policy. No selected default or binary is promoted by this documentation.
Local FP8 implementation/documentation commits are now complete. The current
[delivery review](release-readiness.md) records the CPU-tested installed-package
bootstrap correction; artifact delivery, clean-install checks and publication
review remain outstanding;
older immediate-delivery wording below is a historical checkpoint.

## Routes and evidence

`linear/auto` still uses the established linear preset; `vmm/auto` selects the
experimental fast native route and `vmm/nvme_baseline` selects the protected
legacy fallback. The allocator is a pinned Git submodule; native sources and
adapters live there. No candidate DLL was promoted during the gates below.
Windows CUDA is the initial real-model validated platform; native Linux now has
a selected staged real-model PASS (2026-10-04),
with no WSL/macOS VMM support promise. Ordinary installs do not require the
optional Windows `.[vmm-dev]` environment.

Linux2026-10-04: initial native source/ring/storage/cache/batches implemented and
.so compiled; nine C++ CPU and122 Python CPU contracts PASS. Optional
`.[vmm-dev-linux]` supplies the separate Linux development environment. The
approved allocator probe and seven small CUDA contracts PASS (7 in3.64s,
no skips): stream reuse, retained source/registration/batch ownership and
public numerical/buffer/cancellation cases. GPU execution required permission
escalation outside the sandbox. That initial gate alone was not real-model evidence.
[Linux checkpoint](native-linux-transport-gate.md).

Subsequent user-approved correction distinguishes strict overcommit mode2 from
physically bounded modes0/1. Nine C++ CPU and122 Python CPU PASS (3.25s), .so
recompiled; the preceding CUDA evidence belongs to the preserved original
binary, not this corrected candidate. The subsequently approved corrected-binary
sequence passed three CUDA cache contracts in3.10s, then one native BF16 staged
encoder→unload→transformer→VAE/save generation:1280×704,seed43,real prompt,
eight steps. CUDA steady state1.050970s/step, phase13.145578s, tracker32.2142s;
sampled absolute GPU6.499573GiB and RSS27.3482GiB. No temporary fallback,
admission denial/bypass, quarantine, execution mismatch or monitor error;
encoder source counters zero at transformer boundary, collected tickets zero.
[Executed manifest and limitations](native-linux-staged-gate.md).

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

Post-benchmark delivery change (2026-10-04): `DynamicOffloadApplyResult.cleanup_on_error()`
provides an opt-in error-only lifecycle scope. The LTX public native runner uses
it after transformer attachment through setup/denoise/readback/reporting;
successful unload remains at the existing boundary. Original errors are retained
if cleanup also fails; no force release or poisoned-context recovery is implied.
99 selected CPU contracts, lint and runner syntax PASS. No new CUDA/performance
run for this host wrapper; earlier benchmark artifacts are not rewritten.

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

Native Linux follow-up (2026-10-12): the cgroup-root correction passed the real
CPU contract, isolated build/linkage, approved small trim/source-cache CUDA gate
and one targeted LTX image generation with official Diffusers 0.41.0. PNG parity
with the preserved Linux control passed. See
[candidate evidence](native-linux-cgroup-review-gate.md) for the hash and scope.
The host now locks CPU TorchAudio only on Linux/WSL, retaining CUDA Torch and
Windows selection. Accepted old binaries remain preserved, without default
promotion. This does not validate a fresh-install matrix or complete native
wheel/source-build installation delivery.

### Unified native installation roadmap (2026-10-12)

The user requested one public installation flow, not separate runtime code or
runner recipes for each Linux distribution. Keep one allocator source/API and
bootstrap contract. Platform and ABI-specific binaries are build artifacts, not
different implementations.

- Deliver compatible native wheels for supported platforms, with explicit Linux
  minimum glibc/C++ runtime requirements and verified binary manifests. Ordinary
  wheel installation must not require a compiler or a user-supplied library path.
- Provide a documented, explicit source-build path for environments without a
  compatible wheel. Review toolkit/compiler requirements and preserve accepted
  artifacts. Do not compile at import or inference startup.
- Consolidate example-host setup guidance: separate Windows/Linux/WSL virtual
  environments, system FFmpeg shared libraries, optional development/quantization
  extras, and the Linux/WSL TorchAudio CPU override. Make the audio selection
  reproducible rather than relying permanently on a post-sync manual replacement.
- Verify wheel discovery, ABI compatibility and the supported source-build path
  before marking native installation delivery complete. Python editable package
  assembly alone does not compile or install the native allocator.

WSL's selected finite validation sequence is now closed; see
[WSL evidence](native-wsl-idle-trim-gate.md). The corrected Ubuntu 22.04 build
passed three small source/cache CUDA contracts and selected staged BF16 LTX
image, conditioned video, and Qwen image/leaf-VAE generations. Registration
capacity is constrained in this tested environment; model runs used 384 MiB
staging and a separate 512 MiB source-registration ceiling. This does not
establish arbitrary-model WSL support, full-cache registration performance,
or automatic tuning. Unified native installation delivery remains pending.

Historical preparation boundary: the preserved
Linux idle-trim binary has the expected hash but cannot load with this WSL
distribution's glibc/C++ runtime. Build a separate WSL candidate from the same
source; never overwrite accepted Linux or Windows binaries or upgrade system
glibc to accommodate the old artifact. A successful build is not WSL CUDA or
model acceptance. Continue one user-executed command at a time, with explicit
approval before CUDA gates.

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

Native Linux portability and its selected real-model gate are now complete;
see [executed evidence](native-linux-staged-gate.md). Use the
[Linux continuation checklist](linux-handoff.md) for a new session, with separate
`.venv-linux`/`build-linux`, explicit build selection and preserved Windows
artifacts. The older pre-port checklist is historical, not outstanding transport
work. Compiling a new candidate alone never transfers existing CUDA acceptance.

Reporting-only precision/route labels passed82 CPU tests in5.09s and syntax
compilation, without a new CUDA run. The recorded real-model artifacts predate
those additive fields and are not rewritten. Final release validation of the
chosen source/submodule revision/build, broader user support checks and publication
audit remain separate work; no exhaustive matrix is required for development
closure. Local Linux checkpoint commits and the public/internal documentation
split are the immediate delivery steps. No extra Windows benchmark is scheduled
by this release wording. Do not rewrite private history, remove compatibility,
publish or promote a candidate without the existing explicit scope/approval.
Any provenance audit must retain truthful licensing/attribution obligations;
removing references does not establish independent implementation or license
compatibility. No such cleanup or publication was performed at this checkpoint.
