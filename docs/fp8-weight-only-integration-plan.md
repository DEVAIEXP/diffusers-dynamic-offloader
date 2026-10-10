# FP8 weight-only integration gate

Current status (2026-10-10): selected CPU/native CUDA contracts and Windows
real-model FP8 execution have passed, including the five-second official-group
comparison with exact latent parity. The implementation is model-agnostic;
selected LTX evidence is not a model restriction or arbitrary-model guarantee.
See [current support and responsibility boundary](fp8-weight-only.md).
The dated sections below preserve incremental engineering evidence; their pending
statements describe those checkpoints, not today's outstanding work.

## Scope and boundary

Use Diffusers and TorchAO `Float8WeightOnlyConfig`, initially E4M3 with per-row
scales. Preserve model and Linear forwards; TorchAO owns tensor dispatch and
dequantization. DDO must not implement quantization arithmetic or a new forward.
Weight-only does not imply FP8 matrix multiplication: the inspected upstream
recipe computes in the original precision after weight dequantization.

The native source path currently assumes ordinary dense tensor storage and
binds materialized views to parameters. A quantized wrapper owns payload tensors
and reconstruction metadata; logical BF16 dtype is not physical FP8 storage.
Loader metadata admission also rejects quantized models. Do not bypass these
checks or pass wrappers to raw-pointer transport. Existing backend detection is
not comprehensive TorchAO admission.

## Validation sequence

1. Pin an optional TorchAO candidate in the dedicated host environment without
   replacing Torch. Run an import/CPU-only small Linear contract: original forward
   identity, payload/scales/metadata, finite output and serialization round-trip.
   Assert CUDA remains uninitialized. Stop on incompatibility.
2. Inspect the installed tensor flatten/unflatten and Diffusers checkpoint
   protocols. Specify physical byte accounting, aliases, slices, reconstruction,
   ownership and refusal rules before implementing narrow transport support.
   Keep backend dispatch intact; no runner residency heuristics.
3. Request separate approval for a small CUDA comparison against the same
   TorchAO layer using ordinary allocation. Check output agreement, workspace,
   cleanup and native ownership before a model run.
4. Request separate approval for matched BF16/FP8 model runs. Preserve BF16
   checkpoints; save conversion separately. Measure transfer bytes, preparation,
   steady-state, memory and quality. Do not promise acceleration or quality parity.

Preserve accepted linear, NVMe, image and native BF16 defaults. TorchAO remains
optional. This plan does not authorize CUDA execution or rebuilding the DLL.

## Primary references

- [TorchAO recipe](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py)
- [Float8Tensor storage and dispatch](https://github.com/pytorch/ao/blob/main/torchao/quantization/quantize_/workflows/float8/float8_tensor.py)

Verify upstream observations against the installed pinned release before coding
the storage integration.

## CPU evidence (2026-10-08)

Environment: Windows host `.venv-vmm`, Torch 2.12.1+cu132, TorchAO 0.18.0.
Installed only TorchAO with `uv pip install --no-deps`; Torch was not replaced.
Command: run `scripts/probe_torchao_weight_only_cpu.py` using that environment.

The 16-by-32 BF16 Linear retained its original class forward. Its weight exposed
`qdata` and `scale` through the upstream flatten protocol, with reconstruction
metadata including block size, logical dtype and dispatch settings. Physical
storage was 512 bytes E4M3 plus 64 bytes FP32 per-row scales, versus 1024 bytes
for the original BF16 weight (bias excluded). Logical wrapper dtype remained BF16.

Output was finite and exactly matched the installed TorchAO dequantization and
matmul-then-bias ordering. The initial fused-linear reference differed by at most
0.00390625 due to BF16 operation ordering; the reference was corrected without
loosening tolerance. Wrapper reconstruction and `torch.save`/weights-only
`torch.load` state-dict round-trip preserved output and payload bytes exactly.
CUDA remained uninitialized throughout.

This is not Diffusers safetensors checkpoint acceptance, native transport
acceptance, model quality evidence, or a performance benchmark. A pytree Enum
registration deprecation warning occurred; import and the CPU gate still passed.

## Isolated adapter contract

`ddo-vmm-allocator/src/ddo_vmm_allocator/quantized_payload.py` now describes
the verified per-row E4M3/BF16 recipe without retaining loader tensors. It
accounts for both payloads and reconstructs the backend wrapper over supplied
CPU buffers without changing forwards or implementing dequantization.
Unknown metadata, versions, activation quantization, shape/dtype mismatches,
noncontiguous/gradient payloads and shared storage are refused. Alias support
is not yet implemented. The contract is intentionally not attached to a runtime.

The first pytest invocation passed 13 functional cases but failed 13 teardown
CUDA assertions: importing the existing allocator package eagerly pulled in
TorchAO prototype MX kernels, whose Triton availability check initialized CUDA.
This was not an FP8 kernel launch, but it violated the CPU-only gate. The new
module was subsequently loaded in isolation for CPU tests. Native integration
must retain allocator bootstrap before any potentially CUDA-initializing imports;
the isolated gate does not certify the package's general import behavior.

Native source registration, checkpoint payload mappings, GPU wrapper views and
event-safe publication/retirement are still pending. Neither a Python payload
cache nor a per-forward dequantization shim has been introduced.

## Explicit checkpoint source association

`quantized_sources.describe_float8_sources_cpu` validates immutable file slices
for both physical payloads and returns a tensor-free source contract. It requires
explicit loader mappings, complete shape/dtype/byte metadata and absolute paths;
missing payloads or overlapping file ranges fail closed. It neither reads files
nor claims that matching metadata proves payload value equivalence.

The CPU tests use the installed Diffusers quantizer's safetensors serialization
hook and loading reconstruction hook, as well as upstream TorchAO reconstruction.
Both a single shard and payloads split across two shards preserve FP8 bytes,
scales and output exactly. This validates the small checkpoint format boundary,
not full model `from_pretrained` loading or native source execution.

The consolidated command selects `tests/test_quantized_sources_cpu.py` and
`ddo-vmm-allocator/tests/test_quantized_payload_cpu.py` with the host Python.
Set `CUDA_VISIBLE_DEVICES=-1` only for this CPU process to prevent upstream
prototype-import capability queries from initializing CUDA. All 24 selected
contracts passed; the tests also assert CUDA remains uninitialized.

Pending: native source registration, one execution-unit span containing data,
scales and ordinary bias, GPU wrapper publication without replacing backend
dispatch, final-consumer event ownership, retirement and temporary fallback.
Normal runtime defaults do not import the experimental source helper.

## Composite publication and retirement

The opt-in `Float8WeightOwnerGroup` reuses `TensorOwnerGroup` publication,
rollback and event-gated retirement. Two validation/placeholder extension methods
were extracted without changing dense ownership semantics. Complete payload
views reconstruct a TorchAO wrapper; the manager does not dequantize or replace
the model forward. Successful publication drops the original loader wrapper.
Retirement invalidates every alias with a meta wrapper only after completion.

The selected CPU sequence passed 46 tests: payload contracts, single/split-shard
checkpoint sources, composite owners and existing dense ownership regression
contracts. It verifies wrapper alias identity, physical byte accounting, exact
CPU output, incomplete-view refusal, changed-owner refusal, partial-publication
rollback and failed/pending event retention. CPU event doubles are not real CUDA
event acceptance. Tests ran with GPU visibility disabled for this process.

The composite helper remains separate from the active model arena attachment.
Native integration still must enumerate physical bindings for data and scales,
place both in the same actual execution unit with bias, publish/restore the
complete wrapper for execution slots and fallback, and retire views only after
the real native upload/compute events. No CUDA gate or benchmark has run for FP8.

## Native binding candidate (not CUDA-accepted)

The explicit `attach_float8_component_parameters` API now expands composite
weights into physical bindings on the existing component arena adapter. FP8
data, scales and ordinary bias share the original execution unit. Live binding
tables target the backend wrapper's physical tensors, so existing native slot
and fallback storage changes reach TorchAO without a replacement math forward.
Logical wrapper ownership takes over both payload views after publication.
The initial native candidate refuses multiple logical module owners of one FP8
weight; the broader CPU alias contract does not imply native alias acceptance.

The partial-fallback candidate retains upload and compute events together. A
failed completion fence leaves the component active and prevents backing close.
These changes are opt-in; dense hooks retain their prior behavior. A consolidated
CPU sequence passed 89 tests, including physical unit geometry, live payload
rebinding, partial fallback failures and existing dense/component regressions.

Package adapter exports are now lazy: a fresh `torch_allocator` import does not
import Torch, Diffusers or TorchAO. This closes the pre-install CUDA-initialization
hazard discovered after installing the optional backend. Accessing the existing
adapter exports still resolves the same public objects.

`scripts/probe_fp8_native_cuda.py` was syntax-checked and its first approved
execution stopped during BF16 control attachment, before FP8 execution.
The omitted process cache ceiling resolved to zero and was rejected by
`NativeSourceCache` with `ValueError: process cache limit exceeds its model/micro
configuration ceiling`. Evidence is retained at
`E:/LTX-2/outputs/fp8_native_gate_01/bf16_control/report.json`.
No native inference acceptance or FP8 performance conclusion follows this run.
The corrected probe explicitly sets an 8 MiB process ceiling covering its
already configured 4 MiB staging ring plus 4 MiB cache. Neither buffer, GPU
budget nor DLL changes. Re-execution requires separate user confirmation.

The second approved execution passed cache construction but stopped during
BF16 execution-pool creation; FP8 still did not run. Evidence is retained at
`E:/LTX-2/outputs/fp8_native_gate_02/bf16_control/report.json`. The native pool
rounds each slot to the selected 2 MiB page and requires the two slots' total
to be strictly smaller than the weight ceiling. Thus 4 MiB of slots cannot
be admitted under the probe's 4 MiB ceiling. This is diagnostic configuration
incompatibility, not an FP8 inference failure. Do not relax production admission.
A proposed 6 MiB diagnostic weight ceiling keeps two 2 MiB slots and leaves
one 2 MiB page of headroom; it requires approval before changing or rerunning
the probe. Host staging/cache sizes, allocator cache and DLL remain unchanged.

The approved 6 MiB probe (`fp8_native_gate_03`) executed its BF16 first forward
with finite output exactly matching the reference and completed cleanup/meta
retirement. It then failed the required positive execution-buffer upload
counter; FP8 did not run. Native streaming reserves 4 MiB for the two slots and
leaves a 2 MiB faultable prefix. The original two 128-wide layers fit entirely
inside that prefix, so this fixture cannot validate slot execution. Do not
remove the streaming assertion or claim this gate passed. A revised fixture
must place real execution units beyond the prefix, with CPU geometry checks
before any further CUDA run. Probe snapshots now precede assertions so a
future failed contract retains its last native metrics.

The user-approved revised fixture uses eight 512-wide layers, a 6 MiB weight
ceiling, two 2 MiB execution slots, 6 MiB host cache/registration, 4 MiB staging
and a 10 MiB combined host process ceiling. The allocator cache remains 4 MiB.
Only this diagnostic fixture changes; production runtimes and DLL are intact.
Its CPU-only preflight passed with CUDA uninitialized, saving evidence at
`E:/LTX-2/outputs/fp8_native_geometry_04_cpu/preflight.json`:

| Case | Physical parameter bytes | Largest unit bytes | Units beyond 2 MiB prefix |
| --- | ---: | ---: | --- |
| BF16 control | 4,202,496 | 525,312 | blocks.3 through blocks.7 |
| FP8 weight-only | 2,121,728 | 265,216 | blocks.7 |

Both reloaded serialized checkpoints fit the explicit 6 MiB cache, both unit
layouts fit a page-rounded 2 MiB slot, and CPU forwards remain finite.
Intentional interruptions now target the final block, which is streamed in
both cases. This validates geometry, not actual native CUDA admission or
performance. Further CUDA execution still requires approval.

The approved revised CUDA fixture (`fp8_native_gate_04`) executed its first
BF16 forward exactly, recorded five execution-buffer uploads, stayed within
6 MiB native weight residency, had zero fault/admission/fallback failures and
completed meta retirement/cleanup. It stopped at an incorrect probe assertion
requiring zero transfer-owned events while reusable batches were still open;
the snapshot contained three retained transfer events and eight batch events.
FP8 did not run. Native ownership snapshots count retained objects, not only
unfinished transfers. Existing native submission reaps completed transfers
across batches; batch close waits for completion and releases remaining tickets.
The corrected diagnostic checks bounded event retention during open-batch
execution and asserts zero transfers, batch events and live tickets after
explicit batch close, before cache/store close. Production native code, DLL,
buffer sizes and numerical acceptance remain unchanged. This corrected CUDA
sequence requires approval; the previous run is not a passing full gate.

## Selected small Windows CUDA gate accepted

The corrected user-approved sequence passed both cases, exit 0, on 2026-10-08.
Reports are retained at `E:/LTX-2/outputs/fp8_native_gate_05/{bf16_control,fp8}/report.json`.
The preserved DLL hash remained
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Each case reloaded its own safetensors checkpoint, matched the corresponding
upstream CUDA reference exactly on three repeated forwards, and recovered
after intentional RuntimeError and KeyboardInterrupt at the streamed final
block. FP8 physical bytes and scales matched the saved CPU payloads while active.
No model math forward was replaced by DDO.

| Measurement | BF16 control | FP8 weight-only |
| --- | ---: | ---: |
| Physical parameters, bytes | 4,202,496 | 2,121,728 |
| Execution-buffer uploads, including interruption/recovery | 35 | 7 |
| Final native weight residency, bytes | 6,291,456 | 6,291,456 |
| Source file bytes read | 4,205,104 | 2,127,400 |
| Fault failures / temporary fallbacks | 0 / 0 | 0 / 0 |
| Transfer events / live tickets after batch close | 0 / 0 | 0 / 0 |

Read bytes remained unchanged after the first repeated pass. Final cache/store
close and meta retirement completed in both cases. Reusable batch events were
also zero after close. These are small correctness/ownership contracts, not
timing or full-model quality evidence. Equal native residency reflects page and
slot geometry; the smaller checkpoint does not establish lower model VRAM or
faster denoising. Optional upstream Enum deprecation/import warnings remain
visible in logs; they did not prevent actual checkpoint reconstruction in this
selected gate. Normal loader/configuration/runner integration and real LTX
conversion/validation remain pending; existing BF16/linear/NVMe routes and the
native binary were not changed.

## Reusable conversion utility implemented

`scripts/convert_fp8_weight_only.py` now provides local, CPU-only component
conversion, with optional TorchAO 0.18.0. It uses upstream quantization and
Diffusers `save_pretrained`, keeps BF16 originals intact, refuses existing or
nested output paths, records dependency/recipe/payload hashes, and reconstructs
serialized data/scales and untouched tensors exactly across output shards.
It retains failed diagnostics and does not mark a partial output usable.
The full source component must fit CPU RAM; output sharding is not streaming
source loading. Shared linear storage and pre-quantized input are refused.

The first CPU sequence caught missing upstream quantizer environment
initialization before serialization (26 passed, two failed). Calling the
upstream `validate_environment` with no device map initialized its serialization
state without CUDA. The next sequence caught an unsupported `256B` test shard
size string (27 passed, one failed); the fixture uses upstream's supported
integer byte size instead. The final consolidated CPU sequence passed all
28 tests, with CUDA hidden and initialization forbidden. Evidence fixtures:
`E:/LTX-2/outputs/fp8_conversion_cpu_03`.
These fixture/setup failures did not modify real model weights or the runtime.
Usage and limitations are in `docs/fp8-weight-only-conversion.md`.
No real LTX checkpoint was converted and normal loader/runner opt-in remains
the next integration step.

## Explicit loader/runtime/host candidate (CPU accepted)

The opt-in config `vmm_float8_weight_only` now selects a local composite loader
and native FP8 component adapter. The loader checks the exact supported recipe,
local shard index and metadata, constructs a Diffusers architecture on meta,
reconstructs CPU wrappers through upstream deserialization and associates all
physical sources with final owners. It does not convert during loading or
replace model math forwards. Invalid flags are refused before existing hooks
are removed. Linear/NVMe/default dense routes retain their previous behavior.

Initial CPU tests caught association to temporary loader payloads instead of
final owners. Upstream flattening clones serialized payloads; descriptor binding
now targets final model qdata/scales and uses upstream serialization to establish
their physical names and value/geometry equivalence. Tests also caught direct
Parameter serialization (fixed by upstream-supported detach), a fixture which
mistook alignment padding for unique bytes, and a CPU double using registration
below the public resolver's existing 64 MiB segment granularity. No native rule,
DLL, benchmark ceiling or production default was relaxed for these corrections.

Final consolidated validation: 123 tests and six subtests passed, CUDA hidden
and initialization forbidden, with fixtures at
`E:/LTX-2/outputs/fp8_integration_cpu_06`. It includes real single/split-shard
conversion/loading, exact physical payload hashes, source-layout discovery,
native dispatch doubles, existing dense/component ownership regressions and
modular host configuration/parsing. CPU doubles are not CUDA public-route proof.

The video host now has `--transformer-fp8-weight-only`, requiring native fast
VMM, a separate `--model-path` root and `--local-files-only`. It retains auxiliary
model loading and explicitly excludes compilation/custom execution kernels and
auxiliary source detachment during initial FP8 acceptance. Transformer component
ownership is explicit; encoder config never inherits the FP8 flag.
Neither real LTX conversion nor a CUDA test ran in this integration block.
Next gate must validate the new public load/attach/cleanup boundary, separately
from the already accepted direct-native small gate, before real-model claims.

`scripts/probe_fp8_public_cuda.py` prepares the named public-boundary gate:
save a small Diffusers BF16 component, convert it with the reusable utility,
load through `from_pretrained_with_dynamic_offload`, attach through
`enable_offload`, consume public metrics and retire through
`remove_dynamic_offload`. It checks three exact repeated outputs against each
case's own CUDA reference, real native streaming, physical FP8 data/scales,
unchanged reads and interruption recovery. Neither DLL nor LTX weights change.
The selected CUDA cases stop at the first failure and require approval.

Its CPU-only preflight passed without CUDA initialization, with evidence at
`E:/LTX-2/outputs/fp8_public_preflight_cpu_01/preflight.json`. Public capacity
inspection now recognizes explicitly loaded FP8 components instead of sending
their wrappers into dense discovery. It bound all 16 BF16 and 24 FP8 physical
sources and observed eight execution units in each case. The proposed GPU
weight ceiling remains 6 MiB, with two 2 MiB execution slots. Public transport
uses one 64 MiB cache/registration segment, 4 MiB staging and a 68 MiB combined
host ceiling; this is a declared small-gate configuration, not a change to
production/real-model ceilings. No CUDA execution has occurred for this script.

The first approved public CUDA attempt (`fp8_public_gate_01`) stopped during
fixture preparation, before any BF16/FP8 inference or attachment. Bootstrap had
already initialized CUDA, and the reusable converter correctly refused that
process with `RuntimeError: Run conversion in a fresh CPU-only process`.
The converter guard is retained. The prepared gate now requires
`--checkpoint-root` from the passing CPU-only preflight and performs no
conversion after bootstrap. It reads the preserved small CPU checkpoints;
CUDA reports go to a separate new output directory. Re-execution requires
confirmation. Real LTX weights remain unchanged.

## Selected public API CUDA gate accepted

The corrected sequence passed BF16 control and FP8, exit 0, on 2026-10-08,
using `--checkpoint-root E:/LTX-2/outputs/fp8_public_preflight_cpu_01` and reports
at `E:/LTX-2/outputs/fp8_public_gate_02/{bf16,fp8}_report.json`.
It exercised the public loader, `enable_offload`, public metrics and
`remove_dynamic_offload`, without conversion in the CUDA process. Both cases
produced finite outputs exactly matching their respective CUDA references on
three repeated forwards, reused cached source bytes, exercised native streaming,
recovered from RuntimeError/KeyboardInterrupt at the final streamed block, and
completed meta retirement plus removal of source descriptors. FP8 payload data
and scales were exact. No fault failures, temporary fallbacks or cache admission
denials occurred. Preserved DLL hash and the declared memory controls remained
unchanged. Snapshots before interruption showed 15 BF16 and three FP8 streaming
uploads; this difference is fixture geometry, not a speed comparison.
Both snapshots reported 6,291,456 native resident bytes. This closes the selected
small public integration contract, not full LTX performance/quality acceptance.

Read-only real-model preparation found the BF16 transformer in
`F:/hf_cache/hub/models--OzzyGT--LTX-2.3-Distilled-1.1/snapshots/f070fb11c2af41e90ad9012d1f495b5131753a4d/transformer`,
four shards totaling approximately 35.37 GiB. At inspection, E: had only
approximately 4.83 GiB free, F: approximately 326.09 GiB, and free physical RAM
approximately 36.73 GiB. Do not start the current full-component CPU converter
with this narrow RAM margin or write the real checkpoint to E:. A bounded
shard-based conversion extension, tested for unchanged upstream metadata and
exact payload reconstruction, is the next preparation step. No actual LTX
weights have been converted and no real-model CUDA benchmark is authorized.
It bootstraps the preserved DLL before CUDA and runs one two-layer BF16 component
control followed by one two-layer FP8 candidate, stopping at the first failure.
Each uses a 4 MiB native weight ceiling, 4 MiB source cache/registration and 4 MiB
allocator cache. It checks exact upstream-reference output, physical FP8 bytes
and scales while the unit is active, repeated cache reads, execution slots,
interruptions and meta retirement. No LTX model, compile test, performance claim
or DLL rebuild is part of this gate. GPU execution requires user confirmation.

DDO's normal loader, configuration and runner selection are not yet connected to
this experimental native API. That opt-in follows small CUDA acceptance; do not
remove the existing quantized-payload admission checks in advance.

## Reusable conversion deliverable

Before converting real model weights, provide a saved reusable BF16-to-FP8
weight-only conversion script using the supported upstream TorchAO/Diffusers
checkpoint format. It must preserve original checkpoints, refuse output
overwrite, record the recipe and dependency versions, and validate reloading
the serialized data and scales. Native small-gate acceptance remains a
prerequisite for proposing real-model CUDA validation.

## Bounded conversion CPU gate accepted

The reusable converter now exposes `--stream-shards`. It constructs the
architecture on meta, validates source index/keys/shapes, and performs two
tensor-at-a-time passes. Complete upstream metadata is replicated across
output shards; serialized names and quantization math remain upstream-owned.
Source and converted payload hashes are checked between passes. Output buffers
are limited by the selected shard size except for oversized individual tensors.
Working quantization/serialization buffers are additional, not sampled RSS.
The existing full-model mode and all DDO runtime routes remain unchanged.

Selected CPU acceptance: 129 tests and six subtests passed in 16.50 s, including
single/split input and output, exact agreement with the full-model converter,
public DDO reload, unchanged embedding/norm/integer-buffer payloads, no full
CPU model loading, source preservation, invalid-index refusal and changed-source
failure diagnostics. CUDA was hidden throughout. No native binary changed.

Header-only inspection of the real transformer found 4,186 BF16 tensors and a
largest tensor of 301,989,888 bytes (288 MiB). The user-selected output root is
`F:/models/ltx23-distilled-fp8-weight-only`, with `transformer` as the converted
component directory. Drive F: had approximately 326 GiB free. Real conversion
is the next separately proposed CPU operation; no real weights have been
converted and no quality/performance claim follows from these CPU contracts.

## Real transformer conversion accepted (2026-10-09)

The user-approved CPU command completed with exit 0 using `--stream-shards`
and `--max-shard-size 2GB`, writing the new component at
`F:/models/ltx23-distilled-fp8-weight-only/transformer`. Original BF16 checkpoint
files were not modified. The converter checked source tensor hashes across
both passes and exact reconstruction of every saved quantized payload, FP32
scale and untouched tensor. CUDA was not initialized.

The source contained 37,977,676,288 tensor bytes (35.3695 GiB). Converted output
contains 10 safetensors shards totaling 19,030,440,296 bytes (17.7235 GiB),
approximately 49.89% smaller; shard headers are included in output bytes.
There are 1,660 converted linear weights. Largest source tensor: 288 MiB;
peak retained output buffer: 1,993,896,832 bytes, excluding working buffers.
The observed process peak working set reached 16,224,194,560 bytes
(approximately 15.11 GiB); this is separate from the converter's buffer metric.

`conversion_report.json` records `status=passed`,
`serialized_reload_exact=true`, `cuda_initialized=false`, recipe and dependency
versions. Report SHA256:
`FE1B43127101677556DFA9A64645EABEA9845F8EF15E3F2CE6202D54C476A8AD`.
This accepts checkpoint serialization, not real-model DDO attachment, generation
quality or performance. Those require a separately approved inference gate;
the selected BF16 benchmark and runtime settings remain unchanged.

## First real inference attempt and CPU correction (2026-10-09)

The approved 1280x704, 5 s, 24 fps, seed 43, four-step audio-conditioned
gate stopped before transformer attachment/denoising, exit 1. Encoder,
connectors and conditioning phases completed. The failure was a CPU loader
source-binding refusal, not a CUDA illegal access or OOM. Evidence:
`E:/LTX-2/outputs/fp8_real_5s_4steps_gate_01.console.log` and its
`metrics/modular_run.json` (`status=failed`, elapsed 33.7198 s). No FP8
denoising timing or generation acceptance follows from this attempt.

CPU-only reproduction proved a temporary upstream flatten clone reused a
prior shard's freed address: `transformer_blocks.2.audio_attn1.to_v._weight_scale`
in shard 2 collided with an 8,192-byte descriptor from shard 1. A fresh registry
accepted its actual geometry. Temporary flatten clones must therefore have a
shard-local registry. Global serialized-key uniqueness and the final persistent
registry of actual published model owners remain enforced. No allocator,
checkpoint, native DLL, residency budget or protected runtime was changed.

The focused CPU correction gate passed 29 tests, including a deterministic
cross-shard address-reuse regression. The real saved checkpoint subsequently
loaded in a fresh CPU-only process: 4,186 logical tensors, 5,846 bound/available
physical sources, exact agreement with every conversion-report payload hash,
CUDA initialized false, exit 0. This is loader acceptance only; a corrected
real-model CUDA attempt still requires separate user approval.

The selected consolidated CPU integration/regression suite also passed:
130 tests and six subtests in 17.39 s, with CUDA hidden and a fresh
`fp8_recycled_integration_cpu_01` artifact directory.

## Second real inference attempt: root-state admission (2026-10-09)

The explicitly approved unchanged retry at
`E:/LTX-2/outputs/fp8_real_5s_4steps_gate_02` failed during component layout
preflight, before native transformer attachment/denoising. The prior loader
correction worked: all 5,846 physical sources were bound. Load/prepare took
32.5 s; final failed-run elapsed was 103.4579 s. No CUDA illegal access or OOM
was reported. Four-step generation has not passed.

Meta-only architecture inspection identified two root-owned plain parameters:
`scale_shift_table` (2x4096 BF16, 16,384 bytes) and
`audio_scale_shift_table` (2x2048 BF16, 8,192 bytes). Root-owned state overlaps
descendant execution units; the existing component guard correctly refuses it.
Collapsing the complete component into a root execution unit is not acceptable
under the unchanged 1.5 GiB weight ceiling.

Proposed follow-up, not implemented or CUDA-accepted: explicitly bounded
component-lifetime root parameter state on native allocation, separately
accounted and source-validated, with completion-fenced cleanup. Keep this opt-in
to the FP8 component route and retain the protected dense component guard.
Do not relabel parameters as buffers, remove the overlap guard globally, raise
budgets, or introduce an architecture-specific residency policy. CPU contracts
must cover root/child execution, bounded refusal and retirement before another
separately approved CUDA validation. Preserved DLL and BF16 paths are unchanged.

## Bounded root-state implementation: CPU acceptance (2026-10-09)

The approved extension is implemented only for the explicit FP8 component
adapter. Root-owned plain parameters with immutable sources are separated from
streamed execution units. Root/descendant aliases and root tensor subclasses
are refused. Root parameter payload plus component buffers must fit the existing
explicit `component_buffer_limit_bytes`; parameters remain parameters, not buffers.
Dense component discovery retains its root overlap refusal by default.

Publication reuses the existing **single** `NativeExecutionBuffer`, not the
rotating two-slot pool: native address/mapping, one immutable native batch upload,
and a lifetime pin. No loader CPU tensor survives successful publication. Its
page-rounded capacity is deducted from the declared weight ceiling before the
streamed store is created, using the native driver granularity and page rounding.
Mismatched capacity or insufficient remaining ceiling refuses attachment.
Public residency totals include this backing separately. No budgets are raised.

Final-consumer fences precede retirement to meta and native backing destruction.
Partial upload/publication failure retains backing until completion is proven;
cleanup failure cannot silently release it. Native implementation/DLL, legacy
dense, linear and NVMe routes are unchanged.

Selected consolidated CPU acceptance: 138 tests and six subtests passed in
17.23 s (`fp8_root_final_cpu_01`). Contracts cover once-only publication,
unchanged math forward, source/limit/alias refusals, granularity rounding,
ceiling subtraction, actual single-buffer transport with CPU native-boundary
doubles, residency totals, incomplete-consumer retention and partial-upload
cleanup failure. This is not CUDA acceptance.

Real saved LTX CPU geometry also passed: two root parameters totaling 24,576
bytes, 76 disjoint streamed execution units, largest aligned execution-slot
payload 387,323,904 bytes, 19,019,550,208 unique physical parameter bytes,
zero component buffers and 1,660 composite FP8 weights. CUDA was not initialized.
A small root-state CUDA lifecycle gate must pass before another real LTX run;
both require separately presented commands and user approval.

## Small root-state CUDA gate prepared; execution pending

`scripts/probe_fp8_public_cuda.py --root-state` retains the original BF16
control and exercises the FP8 candidate with two nontrivial BF16 root parameters
totaling 24 KiB, consumed before and after its eight child blocks. The opt-in
small candidate has an 8 MiB total weight ceiling, of which 2 MiB is reserved
for root backing and 6 MiB remains for arena/slots. The existing BF16 control
retains 6 MiB; production runner ceilings and the original default gate are
unchanged. Host cache/registration: 64 MiB; staging: 4 MiB; combined host
ceiling: 68 MiB; allocator cache: 4 MiB; preserved DLL hash remains required.

CPU fixture preparation passed, exit 0, with CUDA hidden:
`E:/LTX-2/outputs/fp8_root_public_preflight_cpu_01/preflight.json`.
Geometry has eight streamed units, 26 FP8 physical sources including the two
root sources, and 24 KiB root state. This is not native admission acceptance.

The proposed CUDA command uses that checkpoint root, `--root-state`, and new
output `E:/LTX-2/outputs/fp8_root_public_gate_01`. It asserts three outputs exact
against each case's own CUDA reference, active FP8 payload equality, root
parameter equality, one root setup upload, unchanged cached read bytes,
streamed uploads beyond root setup, total residency within the declared ceiling,
recovery after RuntimeError/KeyboardInterrupt and final meta/source retirement.
Stop at the first failure; no real-model run is authorized by CPU preparation.

## Small root-state CUDA lifecycle gate accepted (2026-10-09)

The explicitly approved command completed with exit 0 in approximately 6.09 s,
using the prepared root-state fixture and unchanged preserved DLL. Reports:
`E:/LTX-2/outputs/fp8_root_public_gate_01/{bf16,fp8}_report.json`.
Both cases record `passed`, repeated outputs exact against their own CUDA
reference, recovery after RuntimeError/KeyboardInterrupt and complete cleanup.
FP8 physical payload bytes/scales and root parameter values were exact.

The FP8 ownership report records two root parameters totaling 24,576 bytes,
one setup upload, 2,097,152-byte native capacity/budget charge, an unchanged
8,388,608-byte fixture total ceiling and 6,291,456-byte remaining arena ceiling.
Its final pre-interruption snapshot has 8,388,608 total resident weight bytes,
including 2,097,152 root bytes; four execution-buffer uploads include one root
setup upload plus streamed work. There are 26 immutable file-backed bindings
covering 2,146,304 physical parameter bytes. Cached source read bytes remained
2,152,248 across repeats. Fault failures, temporary tensors and cache admission
denials were zero. Root/model payloads retired to meta and source descriptors
were removed. The unchanged BF16 control retained its 6 MiB ceiling.

This accepts the selected small root-state lifecycle, not real LTX generation,
FP8 quality or speed. The next proposed real run is the unchanged 1280x704,
5 s, four-step audio-conditioned command, with fresh output
`E:/LTX-2/outputs/fp8_real_5s_4steps_gate_03`. It still requires user confirmation.

## Real FP8 four-step generation accepted (2026-10-09)

After separate user approval, `fp8_real_5s_4steps_gate_03` completed with exit 0:
1280x704, 5 s, 24 fps/121 frames, four steps, seed 43, real image/prompt and
audio conditioning. The saved FP8 transformer used native VMM with the unchanged
1.5 GiB weight ceiling; the native BF16 encoder retained 4 GiB. Chunk sizes were
1024, with phase modulation, compact timesteps and in-place residuals enabled.
Compilation and experimental execution kernels were disabled; output was latent,
not decoded video. The preserved native DLL was unchanged.

Metrics: total 259.2684 s; transformer load/prepare 33.0865 s; attachment
85.4993 s; denoising 109.2867 s (progress approximately 27.09 s/it).
Sampled peak GPU usage was 5.4516 GiB relative to the 1.0269 GiB baseline,
6.4784 GiB absolute; peak process RSS was 27.2126 GiB. These are sampled
driver/process observations, not native residency or PyTorch allocator peaks.
The final transformer report records zero fault failures, temporary tensors and
source-cache admission denials. Root state uploaded once; its 2 MiB capacity
was charged within the original weight ceiling. Completion collection retired
132 tickets to zero; the process completed cleanup without a reported error.

CPU-only verification found finite float32 video latents `(1, 14080, 128)` and
audio latents `(1, 126, 128)`; CUDA remained uninitialized during that check.
Evidence: gate directory `metrics/modular_run.json` and
`ltx23_modular_1280x704_5.0s_seed43.pt`. Preserve the raw report's historical
validation label; this acceptance record supersedes its stale pending label.
This establishes selected real-model execution, not visual quality, BF16 parity
or a speed improvement. Existing 10 s compiled BF16 results are not matched
controls. Decode/quality review and a matched BF16 comparison require approval.

The subsequently approved standalone decode completed with exit 0 in 33.7275 s.
It used the existing staged BF16 video/audio decoders under ordinary CUDA
allocation, without transformer/encoder loading, native allocation or another
denoising run. Output `fp8_real_5s_4steps_gate_03/fp8_decoded.mp4` is H.264,
1280x704, 24 fps, 121 frames/5.041667 s, with stereo 48000 Hz AAC audio
(5.010000 s). Audio is finite and the source latent hash is unchanged.
The adjacent `fp8_decoded.decode.json` records successful completion. Media
structure is accepted; visual/audio quality still awaits user review and this
does not establish parity or a performance gain versus BF16.

## Approved matching BF16 workload control (2026-10-09)

`bf16_5s_4steps_fp8_quality_control_01` completed with exit 0 under the same
generation inputs and optimization controls as the real FP8 gate. CPU-only saved
video/audio validation passed. No fault failures, temporary tensors or source
cache admission denials were reported; 107 commit-pressure entries recovered.
Completion tickets retired to zero. Transformer load/prepare was 1.6204 s,
attach 6.5577 s, denoise 118.1183 s, total 155.3928 s without decode.
Global absolute sampled GPU peak was 4.9624 GiB and process RSS 26.6593 GiB.
FP8 denoising was shorter in this single comparison, but its setup made the
total longer. Dense versus composite ownership and uncontrolled cache warmth
prevent attributing timing differences solely to the weight format.
Relative final latent RMSE against BF16 was 0.194170 video and 0.003182 audio;
this is not a visual-quality metric. BF16 decode and user review are pending.

After separate approval, standalone BF16 decode completed in 32.7062 s, exit 0.
The saved MP4 has the same verified resolution, 24 fps, 121 frames and stereo
48000 Hz audio as the FP8 media. Source latents remained unchanged and audio
was finite. Evidence: `bf16_5s_4steps_fp8_quality_control_01/bf16_decoded.decode.json`.
Both media are ready for user review; temporal stability and perceptual quality
remain unjudged. No additional generation or quantization change was performed.

The user later reported comparable temporal artifacts in both decoded four-step
videos, with no obvious FP8-specific quality loss. This accepts the selected
subjective comparison only, not general precision equivalence.

## Ten-second comparison blocked by failed BF16 control (2026-10-09)

The approved uncompiled BF16/FP8 sequence stopped at its first case:
`bf16_10s_8steps_fp8_comparison_01` failed before any completed denoise step
with illegal CUDA memory access (exit 1, elapsed 70.8338 s). Cleanup and event
recording subsequently failed in the poisoned context. No FP8 run began, no
latent artifact was saved, and no ten-second FP8 speed conclusion is available.
The protected DLL hash and budgets were unchanged. Preserve the failed metrics
and before-denoise snapshot; do not rerun automatically.

The proposed uncompiled control omitted accepted ten-second BF16 execution
auxiliary ownership, auxiliary CPU-source detachment, host-cache release and
phase inspection, in addition to compilation. This configuration difference
is confirmed but is not a proven cause of the illegal access. Future comparison
must retain the validated BF16 lifecycle rather than infer ten-second safety
from the passing five-second recipe. FP8 uses separate composite ownership;
its compatibility restrictions remain intact. New execution requires approval.

The separately approved corrected BF16 case
`bf16_10s_8steps_fp8_comparison_02` then passed all eight steps and cleanup,
with finite saved video/audio latents. It restores accepted auxiliary ownership,
host-cache release, auxiliary CPU-source detachment and phase inspection while
leaving compilation off. No DLL/budget change was made. Total without decode:
502.5720 s; denoise wall time 462.5770 s; first CUDA step 58.65807422 s;
mean CUDA steps 2-8 57.68680134 s/it. Sampled absolute GPU peak: 7.4798 GiB;
RSS: 28.6438 GiB. Fault failures, temporary fallbacks, cache admission denials
and pressure episodes were zero. This is selected configuration acceptance,
not proof of which restored control prevented the prior illegal access.
The FP8 ten-second case still requires confirmation and has not been executed.

## Selected ten-second FP8 comparison completed (2026-10-09)

The separately approved `fp8_10s_8steps_comparison_01` passed all eight steps,
cleanup and CPU-only finite-latent validation. Workload controls, preserved DLL
and budgets match the corrected BF16 generation, with backend-appropriate
ownership: FP8 composite/root ownership instead of BF16 auxiliary ownership.
No compilation, external/custom kernels, decode or automatic retry was enabled.
This single pair is an integration-level comparison with uncontrolled cache
warmth, not isolated dtype performance or a general model claim.

FP8 load/prepare: 31.6734 s; attach: 83.5672 s; denoise wall: 542.5450 s;
first CUDA step: 67.01291406 s; mean CUDA steps 2-8: 67.91826116 s/it;
total without decode: 687.5026 s. Corrected BF16 was 462.5770 s denoise,
57.68680134 s/it steady-state and 502.5720 s total. Thus FP8 is slower in
denoising and total for this selected case, despite fewer source-cache DMA
bytes (146,452,988,032 versus 210,610,591,808). It does not meet the speed goal.
Global RSS peak falls to 25.5203 GiB from 28.6438 GiB; sampled absolute GPU
peak is nearly unchanged (7.4578 versus 7.4798 GiB). Full table and conditions
are in the benchmark sidecar. Do not confuse native cache traffic with driver
shared-memory usage or infer the bottleneck without further evidence.

Fault failures, temporary tensors, cache denials and pressure episodes were
zero. Final tickets retired to zero; video/audio latents are finite. No ten-second
FP8 visual-quality acceptance follows from these contracts. Preserve BF16 as
the accepted performance route and FP8 as optional size/host-memory reduction.
No follow-up generation or compile integration was automatically started.
