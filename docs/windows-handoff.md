# Resume on Windows after Linux closure

Updated 2026-10-04. This document provides continuation context for Codex on
Windows. The Linux stage and local naming migration are complete within the
validated scope; publication and general support are not complete.
It does not authorize CUDA execution, benchmarks, push or another Git cleanup.

## 1. State and startup instructions

Read DDO, allocator and host `AGENTS.md` before acting. Preserve the user's
local files. The checkouts are shared between operating systems; CRLF/LF changes
may appear as modified files without logical changes. Use
`git diff --ignore-space-at-eol --stat` to distinguish these cases; do not run
reset, file checkout, global normalization or automatic cleanup.

| Project | Linux | Expected Windows path |
| --- | --- | --- |
| DDO | `/run/media/eliseu/AI2/ProjetosIA/diffusers-dynamic-offloader` | `E:\ProjetosIA\diffusers-dynamic-offloader` |
| Allocator | `ddo-vmm-allocator` inside DDO | Same submodule inside DDO |
| LTX host | `/run/media/eliseu/AI2/ProjetosIA/ltx2.3-image-diffusers` | `E:\ProjetosIA\ltx2.3-image-diffusers` |

Confirm Windows paths; do not reuse absolute Linux paths.

| Project | Active branch | Functional revision at session closure |
| --- | --- | --- |
| DDO | `vmm-support` | `505662ddad0c6ffd1c46fab35a7dcbce62ca86ea` |
| Allocator | `codex/ddo-native-vmm` | `04f80996ade8feac39c9adf2f22fcf84e3d5833d` |
| LTX host | `codex/ddo-native-vmm` | `dd6167af0f6d103996e722dc4d6fce8ba33c895d` |

The commit adding this handoff may follow DDO's functional revision.
DDO's gitlink points to `04f80996...`. Everything is local: **no push occurred**.
Do not run `git submodule update --remote` or reset to an older remote.
The user wants to test other models personally before deciding on publication.

## 2. Architecture that must remain intact

- `linear/auto` retains the stable path and its presets, without native dependencies.
- `vmm/nvme_baseline` retains the protected NVMe fallback.
- `vmm/auto` selects `vmm_fast_runtime`. No new native tuning presets were added;
  `auto` selects the route rather than calibrating residency.
- `ddo_vmm_bootstrap.ensure_vmm_fast_process()` installs the allocator in a fresh
  child process before CUDA. Direct initialization is diagnostic-only.
- C++/CUDA, arenas, sources, buffers, admission, DMA, events and ownership belong
  to the submodule/DDO. Do not put caching or residency heuristics in the runner.
- DDO remains model-agnostic. LTX runners and custom blocks stay in the host.
- Native text encoding works for the validated BF16/text-only workload, but
  requires explicit opt-in rather than automatic architecture/class selection.
- Preserve quantized forwards and Diffusers compatibility routes.

## 3. Linux environment created and verified

DDO's `.venv-linux` was also used to run the host, not LTX's Python.
`pyproject.toml`/`uv.lock` gained the optional `.[vmm-dev-linux]` profile,
retaining Windows `.[vmm-dev]` and keeping heavy dependencies out of normal installs.

Observed environment: Ubuntu 26.04.1 x86_64, Python 3.12.14, uv 0.12.20,
8192 MiB RTX 3060 Ti, driver 595.91.07, GCC 15.2.0, CMake 4.2.3,
Ninja 1.13.2. Optional toolkit 13.2.1; nvcc 13.2.78.

Linux profile pins: Torch 2.12.1+cu132, torchvision 0.27.1+cu132,
transformers 5.15.1, accelerate 1.14.0, peft 0.20.0, einops 0.8.2,
safetensors 0.8.0 and custom Diffusers revision
`e71ebc5858961e2a37d6a7b161d00ae0063c3ab7`.

Executed Linux preparation, not commands to run on Windows:

```bash
UV_PROJECT_ENVIRONMENT=.venv-linux uv sync --locked --extra vmm-dev-linux --inexact --link-mode copy
uv pip install --python .venv-linux/bin/python --link-mode copy -e ./ddo-vmm-allocator
```

Always select the environment explicitly: plain `uv sync` may replace a shared
`.venv`. Preserve Windows `.venv-vmm`; do not use `.venv-linux/bin/python` there.
Do not automatically sync the entire host: its remote DDO dependency may precede
the local editable checkout. The dynamic module cache remains host-local at
`.hf_modules`. Do not silently switch loader/pipeline or remote-code behavior.

The user installed the system C++ tools. The toolkit came from the optional
profile; no driver, sysctl, swap or memory ceiling was changed to force tests.
`uv lock --check`, editable installation and both Python wheel builds passed.
Wheels **do not include a DLL or .so**: allocator checkout, build and library
selection are still required.

## 4. Native portability implemented

Actual Linux transport was implemented and compiled, not just a suffix change:

- Retained POSIX descriptors and positioned `pread` on immutable sources.
- Native anonymous `mmap` storage, without caching loader CPU tensors.
- Reused native workers, slots, persistent cache, batches and tickets.
- Storage commitment separated from CUDA registration; unregistering keeps bytes.
- Recycling and cleanup preserve ownership until upload/compute events complete.
- Admission through `MemAvailable`, cgroup v2/ancestor limits and RLIMIT AS/DATA.
  Unknown/invalid observations refuse growth; cgroup v1 is not validated support.
- Overcommit correction: mode 2 also considers remaining commit; modes 0/1 use
  finite physical limits and retain other bounds. This does not guarantee OOM
  protection or eliminate allocation/registration failures.
- Windows primitives and native event policy were preserved. No WDDM emulation
  on Linux or automatic budget growth.

Separate builds: `build-linux-cpu` and `build-linux`. Executed Linux recipe:

```bash
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode cpu --jobs 2
export CUDAToolkit_ROOT="$PWD/.venv-linux/lib/python3.12/site-packages/nvidia/cu13"
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode library --jobs 2
```

Span pool ON; dispatcher, lifecycle test hooks and build probes OFF.
CMake test scope, `libcudart.so.13` discovery, optional CCCL and CMake flags were
corrected: `-DDDO_*` is the correct spelling, not `-DDO_*`.
The .so uses CUDA Driver API/libcuda; compilation is not a CUDA gate.

## 5. Completed evidence

| Gate | Result and scope |
| --- | --- |
| Linux C++ CPU | 9 contracts passed, including actual POSIX source and storage/headroom |
| Linux closure Python CPU | 122 passed, no skips; protected routes, bootstrap, profiles, source and lifecycle |
| Initial probe + small CUDA | Probe passed; 7 contracts passed in 3.64 s on the pre-overcommit-correction .so |
| Corrected .so CUDA | 3 cache/registration/batch contracts passed in 3.10 s, no skips |
| Linux real model | One staged generation with native encoder and transformer passed |
| Subsequent interface/cleanup | 99 CPU contracts passed; wrapper received no new CUDA test |
| Final renaming | 55 CPU tests passed in 3.16 s, no skips |

The first sandboxed probe could not find CUDA; the same authorized gate passed
outside it. Do not diagnose this as an allocator defect or change drivers to
resolve it. Request appropriate permission when the environment restricts GPU access.

## 6. Accepted real benchmark

LTX 2.3 distilled, BF16 encoder/transformer, real prompt, 1280x704,
seed 43, 8 steps, one generation, no PAG. Staged execution:
encoder → readback/unload → connectors → transformer → VAE → save.

Fixed limits: 4 GiB active weights, 24.625 GiB storage/registration each,
1.5 GiB total staging, 512 MiB global allocator cache. In the host,
`--vmm-staging-buffer-gb 0.5` means 0.5 GiB per logical slot, three slots.
In DDO's generic CLI, `--ddo-vmm-staging-buffer-gb` means the total.
Do not transfer the host number to the generic API without conversion.

| Measurement | Accepted Windows | Accepted Linux |
| --- | ---: | ---: |
| Prompt phase | 12.6624 s | 11.7280 s |
| Transformer load / prepare | 0.3965 / 0.6261 s | 0.3953 / 0.7808 s |
| First CUDA step | 6.729011 s | 5.788787 s |
| CUDA steady state, steps 2–8 | 1.395602 s/step | 1.050970 s/step |
| Transformer CUDA phase | 16.498256 s | 13.145578 s |
| Complete Generate | 24.2141 s | 19.0590 s |
| RunTracker total | 38.5135 s | 32.2142 s |
| Sampled peak RSS | 27.3479 GiB | 27.3482 GiB |

Totals exclude startup/imports/bootstrap. These are selected observations from
different environments, not speed guarantees or a statistical study.
Host callbacks are not CUDA intervals; do not sum nested timings.
Linux: absolute driver peak 6.499573 GiB, baseline 0.722290 GiB and delta
5.7773 GiB, sampled every 100 ms. Do not interpret delta as absolute use or a
sampled peak as a guaranteed instantaneous ceiling.

A plausible 1280x704 image was inspected; cross-platform pixel equality was not
proven. No temporary fallback, admission denial/bypass, quarantine, execution
mismatch or monitor error occurred in these results. Encoder source counters
were zero before the transformer; tickets were collected to zero.
Transformer cache 24.25 GiB, separate staging 1.5 GiB; 25,846,148,360 bytes read
versus 187,588,136,576 DMA bytes show aggregate reuse, not a per-step read trace.

Host artifacts; do not overwrite:

- Windows: `outputs/ltx_image_modular_vmm/native_encoder_staged_gate_20261003/`.
- Linux: `outputs/ltx_image_modular_vmm/native_linux_staged_gate_20261004/`.
- Linux used snapshot `5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9` under
  `/run/media/eliseu/IA/hf_cache/hub/`. Confirm the Windows cache path;
  do not infer an identical checkpoint from the model name.

Executed command, metrics and artifact hashes are in the
[Linux gate](native-linux-staged-gate.md) and [manifest](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmark-reproduction.md).
Do not automatically repeat the benchmark to validate this documentation.

## 7. Preserved libraries

| Artifact relative to allocator | SHA256 |
| --- | --- |
| `build-linux/libddo_vmm.so` | `a87979f8773a010ddc2c781a37f3ef55112ca558a9d93083b972f51246a2b245` |
| `build-lifecycle-off-20261002/Release/ddo_vmm.dll`, accepted Windows candidate | `755fcdd8dd1a72b371402750c071ca4b2146c0ac6bfe6b94c9b78f50f1ebb729` |
| `build/Release/ddo_vmm.dll`, unpromoted Windows default | `4ac3d8b5396e0cd47a5da0f08560ea0c6783697153bd1072b23dfb7f57ca7aa3` |

Hashes remain unchanged after Linux work and local migration.
The earlier .so is preserved as `build-linux/libddo_vmm.pre-overcommit-mode-20261004.so`.
Its CUDA acceptance does not prove the corrected .so, which has its own gate.
No DLL was rebuilt/promoted on Linux. A future Windows rebuild of current source
needs its own validation; do not overwrite accepted binaries.

## 8. Interface, metrics and cleanup correction

The generic CLI gained independent `--ddo-vmm-source-registered-gb` capacity,
already used by the host and native configuration. Protected-route defaults
remain intact; older namespaces without this field remain accepted.
Source storage and CUDA registration are distinct controls.

`DynamicOffloadApplyResult.cleanup_on_error()` was added. The public native host
uses this scope after attachment through setup, denoise, readback and reporting.
On success it adds no cleanup/synchronization; normal unload remains outside
the scope. On failure it delegates existing cleanup and preserves the original
error if cleanup also fails. No force-free, driver reset or corrupted-context
recovery is promised. Older routes were not redirected.
This change received CPU tests, not a new benchmark/actual CUDA error test.

The runner consumes public `DynamicOffloadMetrics` reports through snapshots,
`as_dict()` and `format_profile()`. `RunTracker.record_ddo_metrics()` combines and
saves them without knowing allocator internals. Catalog/units are in
[metrics](metrics.md); capacity/CLI/ENV distinctions are in
[configuration](configuration.md).

Do not claim every option is interchangeable across CLI/config/ENV:
`build_config_from_args()` does not merge ENV; host defaults may override values.
Component ownership/buffer limits still have no ENV parsing.
Uniform precedence is separate work and was not changed in this session.

Project identifiers were standardized to DDO names, without aliases for old
flags. Shared host options use `--ddo-vmm-module-pattern`, `--ddo-vmm-page-mb`
and `--ddo-vmm-dxgi-{local,shared,system}-headroom-gb`.
Separate diagnostics use `--ddo-vmm-diagnostic-adapter` and
`--ddo-vmm-diagnostic-*`. This does not make diagnostics official presets or
change limits.

## 9. Local history and documentation

Naming migration passed first in isolated copies, then was applied to local
branches with old-hash verification. 275 complete historical trees were checked,
without losing commits/topology/authorship/dates; four historical gitlinks
received coordinated mappings. The historical experimental runner had coexisted
with the validated one; its path was separated as
`run_dynamic_modular_vmm_diagnostic.py` to avoid collisions.

Stashes, remote references, uncommitted files and binaries were preserved.
Local branch messages use the new naming. This did **not** remove licenses,
attributions, factual accounts, results or all history/remote/reflog references.
Do not confuse naming migration with completed legal audit/publication.
Remotes were neither reconciled nor updated.

Verified complete Git backups and maps remain private/ignored under
`.internal/history-backups/snapshot-20261004-ZsQBAQ/`; see its `README.md`,
`migration-plan.md` and `closure.md`. Do not delete these backups.
Do not rerun migration scripts from `.internal/scripts/`: they contain Linux
paths and old leases and are not generic continuation tools.

The public report was reduced to essentials in
[results](https://github.com/DEVAIEXP/ddo-examples/blob/main/experiments/dynamic_offload_results.md). Original detailed history
is in `.internal/reports/dynamic_offload_results-implementation.md`.
Internal review: `.internal/reports/delivery-review-20261004.md`, with a cleanup
addendum; migration closure is in `closure.md`. `.internal`, builds, environments
and local outputs are absent from remote clones: preserve/copy them if resuming
on another disk. Bundles do not cover all ignored/uncommitted files or objects
reachable only through reflogs.

Added/revised documentation: Linux profiles/pyproject, recipe/port/gates,
README, native configuration, metrics, full-pipeline/staged use, manual
validation, reproduction manifest and support. Older documents retain hashes
and descriptions of earlier stages: use this handoff for current state without
rewriting their results as new tests.

## 10. Approved next steps and limits

The selected development target is closed on both platforms.
Do not reopen tuning or create an endless model matrix. The user will personally
test other models on Windows/Linux and report issues; critical fixes need
small gates proportionate to the defect.

Still incomplete: arbitrary model/UNet support, multimodal, multi-GPU, compiled
forwards, driver-loss recovery, binary distribution in wheels, a self-contained
public fixture and a final matched official Diffusers benchmark.
Native encoder remains opt-in. Do not promise WSL/macOS VMM.
Installed bootstrap without a checkout is not a validated distribution contract.

Before CUDA, benchmarks, benchmark-parameter changes, compatibility removal or
another history cleanup: present command, hypothesis and criterion and wait
for confirmation. Stop at the first failure/anomaly; no automatic retries or
limit growth. No push until the user's explicit decision.
Reconcile local/remote history before publishing; do not merge/reset/force-push
assuming older hashes still identify the same revisions.

The first Windows step is reading and inspection, not inference:

```powershell
Set-Location 'E:\ProjetosIA\diffusers-dynamic-offloader'
git status --short
git log -3 --oneline
git ls-tree HEAD ddo-vmm-allocator
git -C ddo-vmm-allocator status --short
git -C ddo-vmm-allocator log -3 --oneline
git -C ..\ltx2.3-image-diffusers status --short
git -C ..\ltx2.3-image-diffusers log -3 --oneline
uv pip freeze --python .venv-vmm\Scripts\python.exe
Get-FileHash .\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll -Algorithm SHA256
```

First confirm `.venv-vmm` and the DLL exist. If environment adjustments are needed,
propose Windows installation with an explicit environment and local editable
DDO/allocator, without replacing Linux's environment or downloading another model.
Use [manual validation](manual-validation.md) for user-conducted tests.

## Prompt to paste into Codex on Windows

> Read AGENTS.md in all three projects and DDO's docs/windows-handoff.md. The
> Linux stage and local naming migration are complete; preserve state and do
> not automatically repeat Git migration, tuning or benchmarks. Inspect paths,
> branches, pin, .venv-vmm and the accepted Windows DLL. I want to test other
> models personally before deciding on push. Before any CUDA/benchmark, propose
> command, hypothesis and criteria and wait for my confirmation. Do not change
> linear/NVMe, increase limits, replace environments or publish without permission.
