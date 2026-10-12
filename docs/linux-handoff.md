# Linux continuation checklist (2026-10-03)

Read this first in a new Linux chat. This document carries project state without
requiring the Windows conversation history. It is a handoff, not a declaration
of Linux runtime support.

## Current continuation addendum (2026-10-11)

Windows subsequently moved the primary host and optional DDO development
profiles to official PyPI `diffusers==0.41.0`. Selected BF16 LTX image/video
and Qwen 512-square generations passed with preserved output hashes. Separate
INT4 SDNQ DDO/streamed-leaf image pairs also passed for LTX image and Qwen;
these use normal PyTorch allocation, not native VMM. See the host's
`docs/official-diffusers.md` and current benchmark evidence. These package and
quantized gates do not inherit Linux/WSL acceptance. Preserve the previously
accepted Linux fork/audio overrides and binaries until a reviewed environment
transition; inspect with the direct platform interpreter or `uv run --no-sync`.
No new C++ source or binary change was introduced by the SDNQ image closure.

The primary runner host is now the sibling `ddo-examples` repository, not the
historical LTX hosts. Read its `docs/platform-validation-handoff.md` for the
finite post-Windows VAE and platform closure, current import/environment checks
and the prompt to resume. Native Linux and WSL are separate validation targets;
WSL has no native VMM support promise. The historical records below remain tied
to their preserved workloads and binaries; do not rerun their old port checklist
or use their setup as the current host environment without inspection.

The selected Windows LTX image/five-second-video resident versus no-stream leaf
VAE pairs now pass with identical saved outputs; resident host defaults remain
unchanged. Current VAE routing, linear convolution and LTX decode/reporting work
is Python-only. Earlier native idle-cache trim work does modify C++ pool/allocator
sources, so inspect the existing Linux `.so` and optional exports before assuming
it includes that feature. Preserve the accepted binary/hash and use an agreed
separate candidate build path if rebuilding. The existing build helper fixes its
output directories; a dry-run is not permission to overwrite `build-linux`.

Current DDO, allocator, host and independent custom-block changes include local
uncommitted/ignored files. Record their state before transferring or freezing a
version; do not infer that remote repositories contain them. This addendum
authorizes no CUDA run, kernel policy change, automatic full benchmark matrix,
commit, push or binary promotion. CPU-first preparation and reviewed finite CUDA
commands require their existing approval boundaries.

Subsequent 2026-10-11 Linux continuation is now recorded in
`docs/native-linux-idle-trim-gate.md`: isolated candidate compilation and small
CUDA trim gate passed, followed by the user-approved primary-host BF16 image
and five-second video resident/leaf pairs. The prior accepted `.so` is unchanged.
Pipeline output parity is within each Linux pair; detailed timings/memory belong
to the sibling host's `docs/benchmark-matrix.md` and dated evidence log. Do not
automatically repeat those closed gates. WSL, environment reproducibility and the
review/local-commit checkpoint remain separate pending work.

Linux environment preparation on 2026-10-04 is now recorded in
[Linux environment](linux-environment.md). Its optional `vmm-dev-linux` profile
and explicit `UV_PROJECT_ENVIRONMENT=.venv-linux` commands supersede manual
Torch-wheel selection below for this machine. System C++ tools are installed;
the native source/ring/cache/batch port compiled and passed CPU gates. See
[Linux transport checkpoint](native-linux-transport-gate.md) for current evidence
and the approved small CUDA sequence, now PASS. The subsequently approved
[staged real-model gate](native-linux-staged-gate.md) also passed on the corrected
binary: native encoder and transformer,1.050970s/step, tracker32.2142s.
New GPU runs still require reviewed commands/limits and confirmation. Remaining
port-work wording below describes the pre-implementation handoff.

After that gate, native overcommit-mode admission was corrected with CPU/build
evidence first. The original CUDA-validated .so is preserved; current corrected
candidate subsequently passed three cache contracts and the single staged model
generation. No kernel policy or DDO ceiling was changed. Linux allocator checkpoint
`433d3e47ff7a6e7f5635ebedf0acc3c3191f6308` is committed locally and pinned by
the DDO closure checkpoint; no push. Bounded acceptance is not general model support.

## Accepted state: do not reopen Windows tuning

Local checkpoints before this preparation:

- DDO `195546d`, branch `vmm-support`.
- Allocator `7cf9f73`, branch `main`, pinned by DDO.
- Sibling LTX host `fbea833`, branch `main`.

Allocator preparation checkpoint: `84f17ba` adds the isolated CPU project,
Linux build helper and explicit thread linkage. DDO's handoff commit pins it.

New preparation changes may sit above these revisions. Inspect `git status` and
`git log -3 --oneline` in each checkout before changing anything. Do not reset
shared files or fetch/reset the submodule to an older remote revision: these
checkpoints were committed locally, not pushed. The user already accesses the
same directories from Linux; resolve their actual mount paths, not `E:\...`.

Selected Windows evidence:

- Transformer three-generation warm times: 1.401990 / 1.397915 / 1.394595 s/it.
- Staged native text encoder → unload → native transformer → VAE/save:
  warm 1.395602 s/it, tracker total 38.5135 s, excluding bootstrap/import startup.
- Text-only BF16 encoder numerical gate passed; native encoder remains opt-in.
- Eleven selected CUDA robustness contracts passed, with no skips. Healthy
  context-selection recovery is **not** poisoned-driver/device-loss recovery.

See [support status](native-support-status.md), the linked gate records and
`experiments/dynamic_offload_results.md`. Do not repeat the Windows benchmark
automatically or expand into an indefinite model/platform matrix. Preserve
`linear_runtime` and the protected NVMe `vmm_runtime` behavior.

Exact local Windows artifacts (do not overwrite/promote):

- Candidate `ddo-vmm-allocator/build-lifecycle-off-20261002/Release/ddo_vmm.dll`:
  `755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729`.
- Default `ddo-vmm-allocator/build/Release/ddo_vmm.dll`:
  `4AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3`.

These hashes identify binaries, not proof that the current source reproduces
them. Linux uses a separate build and needs its own compiler/ABI/CUDA evidence.

## First Linux session: inspect, then CPU only

Open the DDO checkout in the Linux app/CLI. Read its `AGENTS.md`, allocator
`AGENTS.md`, this document and `docs/native-linux-port-plan.md`. If editing the
sibling host, read its instructions too. Record actual checkout/mount paths.
Use a Linux-local build filesystem if the shared mount cannot support execution
or CMake artifacts; agree on that location before changing the fixed recipe.

From the DDO root, inspect versions and state:

```bash
pwd
git status --short --untracked-files=no
git log -3 --oneline
git -C ddo-vmm-allocator status --short --untracked-files=no
git -C ddo-vmm-allocator log -3 --oneline
uname -a
cmake --version
c++ --version
```

Create a **new** environment, never reuse Windows `.venv-vmm` or sibling `.venv`.
If `.venv-linux` already exists, inspect it first; do not replace it blindly.
With uv already installed:

```bash
uv venv .venv-linux --python 3.12
uv pip install --python .venv-linux/bin/python pytest -e ./ddo-vmm-allocator
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode cpu --dry-run
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode cpu --jobs 2
```

This configures `build-linux-cpu`, builds seven shared CPU helper contracts and,
on Linux, the POSIX source contract, then runs only those CPU executables.
No CUDA toolkit, CUDA library or GPU initialization is part of this project.
Stop on the first failure. Pass means eight native CPU tests actually executed
on Linux, including `posix_source_cpu`; Windows' seven passes do not prove that.
The helper never installs tools or runs a CUDA probe/benchmark.

## Historical implementation checklist — completed on 2026-10-04

The POSIX immutable file source and bootstrap discovery exist. The accepted
Windows source/cache/transport **still has Windows-only branches**. Do not
launch the real model just because a `.so` can be loaded.

Finish, inside the allocator (not in host Python):

1. `native/src/source_io.h` and `ddo_vmm.cpp`: portable borrowed file handles,
   ring worker reads, lazy host storage, registration/unregistration and release.
   Keep retained source ownership until reads drain. Storage should be ordinary
   native bytes; no loader CPU tensors or alternate Python caching path.
2. `ddo_vmm_source_cache.cpp`: admission/headroom, persistent storage, trim,
   source tickets and immutable batch submission. Linux anonymous storage and
   host/container limits must be observed conservatively; unknown capacity
   needs explicit refusal/diagnosis, not invented free bytes.
3. Preserve separate committed, registered and GPU ceilings; unregistering must
   conserve cached bytes. Preserve last DMA/compute fences, partial submission
   ownership and safe clear/close refusal. No automatic budget expansion,
   speculative remapping, device reset or WDDM/shared-residency emulation.

Before this lifecycle work, read the **complete** `docs/native-source-cache-plan.md`
and applicable native instructions. The guard locations and implementation block
are recorded in [Linux port plan](native-linux-port-plan.md). Use the real Linux
compiler while implementing, rather than speculative Windows-only portability
edits without a compiler. Any new platform primitive needs a small CPU contract.

## CUDA environment and compilation, after CPU succeeds

Inspect `nvidia-smi`, `nvcc --version`, Python and the installed Torch version.
Choose a Linux CUDA-enabled Torch wheel compatible with the actual driver and
host, and record its version/index. Do not automatically reuse the Windows
`+cu132` wheel selection, or assume toolkit, driver and Torch CUDA versions must
be identical. Linux model dependencies need explicit review; ordinary DDO users
must not acquire the host's heavy dependencies.

After choosing/installing Torch in `.venv-linux`, install DDO's generic extras:

```bash
uv pip install --python .venv-linux/bin/python -e '.[dev,vmm]'
uv pip freeze --python .venv-linux/bin/python
```

Do not use Windows `.[vmm-dev]` or blindly `uv sync` the sibling host into its
Windows `.venv`. If needed, install the sibling host's reviewed Linux runner
dependencies into this same Linux environment, including its pinned custom
Diffusers revision. LTX runner/custom blocks stay in the sibling repository.
Keep trusted module cache local as before; do not replace the harness with a
different pipeline or silently change prompt, dtype, dimensions or step count.

Build the library (compilation only, no GPU execution):

```bash
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode library --dry-run
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode library --jobs 2
```

Output is `ddo-vmm-allocator/build-linux/libddo_vmm.so`. The explicit feature
selection is span pool ON, source dispatcher OFF, lifecycle fault hooks OFF,
standalone GPU probes OFF. Defaults and Windows binaries are not changed.
Thread linkage is explicit. Record source revisions, CMake/compiler/toolkit/
driver/Torch versions and the printed SHA256; inspect actual export/link results.
Compilation does not mean Linux transport or CUDA safety passed.

## Bounded CUDA validation: approval required

After implementing and compiling the transport, propose one consolidated small
gate: immutable-source read → bounded storage/registration → DMA/numerical
component forward → cancellation/unload. Present the **actual command**,
hypothesis, fixed ceilings and numerical/ownership acceptance to the user and
wait for confirmation. Stop on the first failure or memory anomaly, no automatic
retries or full benchmark. Review existing pytest fixtures: a Windows-only skip
must not masquerade as a Linux pass. Do not authorize GPU execution through the
CPU build helper.

Only after that gate, propose the user's selected real-model staged run with
explicit Linux limits. Performance may differ from Windows: Linux does not
provide the same WDDM residency behavior. Record actual routes, setup/cold/warm
timing, CPU versus native ownership, fallback and sampled absolute versus
baseline-relative memory. Missing old fields remain unavailable, not zero.

Update support/benchmark evidence, commit the allocator locally first and then
pin its new revision in DDO. No push, history rewriting, compatibility removal,
publication or Windows DLL promotion is authorized by this handoff.

## Historical port resume prompt (superseded by the 2026-10-11 host handoff)

> Continue the DDO Linux port. Read AGENTS.md, allocator AGENTS.md,
> docs/linux-handoff.md and docs/native-linux-port-plan.md first. Preserve the
> accepted Windows route and frozen fallbacks. Inspect this Linux environment,
> create/use .venv-linux without touching Windows environments, execute only
> the isolated native CPU contracts, then finish the native Linux storage/cache/
> ring/batch block with the real compiler. Before any CUDA execution, present
> command, hypothesis and approval criterion and wait for my confirmation.
> Do not repeat Windows tuning or start an open-ended test matrix.
