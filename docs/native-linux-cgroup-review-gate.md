# Native Linux cgroup-root regression and candidate (2026-10-12)

Status: CPU regression, isolated compilation/linkage and the approved small
CUDA trim/source-cache gate PASS. Official Diffusers was installed subsequently,
as recorded below. No generation, binary promotion or commit ran.
Previously accepted workloads remain tied to their preserved environments/hashes.

## Reviewed change and CPU execution

Allocator HEAD: `bf3280600f0a8695547b1847459989e75b8eac9a`, with dirty
`native/src/linux_host_memory.h` and `native/tests/linux_host_test.cpp`.
The root-identity fix avoids joining an empty relative path to the canonical
cgroup mount base. It does not disable host, ancestor, process or overcommit
admission. Other untracked registration-probe files were preserved untouched.

Executed on native Linux with GCC 15.2.0:

```bash
mkdir ddo-vmm-allocator/build-linux-cgroup-review-20261012
g++ -std=c++20 -O2 -I ddo-vmm-allocator/native/src \
  ddo-vmm-allocator/native/tests/linux_host_test.cpp \
  -o ddo-vmm-allocator/build-linux-cgroup-review-20261012/linux_host_test
ddo-vmm-allocator/build-linux-cgroup-review-20261012/linux_host_test
```

Exit 0: `Linux host/storage CPU contracts passed`. Includes root membership,
exact subtree mount-root identity, existing ancestor/high bounds, malformed
metadata/escape refusal, overcommit modes, actual Linux headroom observation
and anonymous storage lifetime. This executable does not initialize CUDA.

## Separate library candidate

CMake 4.2.3, GCC 15.2.0, CUDA toolkit 13.2.78 from DDO's older `.venv-linux`.
Release, span pool ON; joined spans, source dispatcher, test hooks, allocation
diagnostics, probes and execution ops OFF, matching the accepted Linux recipe.
Configured/built in `ddo-vmm-allocator/build-linux-cgroup-review-20261012`.
Build reached 100%, exit 0; six existing unused alternative allocator helper
warnings remained. No CUDA tests ran during the build.

Candidate `libddo_vmm.so` SHA256:
`5a49e0a6acd672186af116ee31fe1409bd33c605b2846bfc5f9d4ae42d79907c`.
`ldd` resolved dependencies including native Linux `libcuda.so.1`; exports include
Torch malloc/free/idle trim and source-cache lifecycle/accounting functions.
Link/export inspection is not CUDA ownership or model acceptance.

Preserved hashes rechecked unchanged:

- `build-linux/libddo_vmm.so`:
  `a87979f8773a010ddc2c781a37f3ef55112ca558a9d93083b972f51246a2b245`.
- `build-linux-idle-trim-20261011/libddo_vmm.so`:
  `db63138da2fdcca328a244ff885ca44ecd8b33683825b2ab0f6f39a95b3cba79`.

## Environment inspection, no sync

Direct host `.venv-linux` imports and `uv pip freeze` were inspected.
Diffusers remains editable at `/run/media/eliseu/IA/Projetos/diffusers`,
`0.41.0.dev0`; host pyproject now declares official `diffusers==0.41.0`.
Torch remains `2.12.1+cu132`, TorchAudio `2.11.0+cpu`. Imports passed outside
the read-only sandbox and `torch.cuda.is_initialized()` remained false.
The first sandbox import failed because PyTorch had no writable temporary
directory, not because of a CUDA/model defect. The uv-managed environment has
no pip module; freeze was read through uv without installing pip.

Next review the official-Diffusers transition and platform audio recipe before
modifying dependencies. A small approved candidate trim/source-cache gate is
still required before promotion. If the model environment changes, propose a
targeted generation separately; do not automatically repeat the closed matrix.

## Subsequent official-package transition

Only Diffusers was replaced through uv `--no-deps` with official `0.41.0`.
Actual imports now resolve the host venv's site-packages, not the fork. Torch,
audio/media versions stayed unchanged; imports passed with CUDA hidden and
initialization false. Package consistency passed for 81 installed packages.
The host manifest/lock already selects this release; Linux/WSL CPU TorchAudio
remains an explicit override with reproduction guidance in the host environment
document. No sync or generation was run. New CPU results and CUDA approval must
be recorded separately from the original fork-based model evidence.

The host CPU suite plus DDO VAE-policy/linear-convolution/device-cache and
allocator optional-trim mocks passed: **152 tests and 10 subtests in 36.93 s**.
Execution used the host interpreter, `CUDA_VISIBLE_DEVICES=-1`, host-root and
`modular_diffusers` PYTHONPATH, Python `-B` and disabled pytest cache provider.
No CUDA gate or model generation is included in that result.

## Approved small CUDA gate: PASS

Executed from the examples host with its `.venv-linux` interpreter, official
Diffusers 0.41.0 and candidate SHA256
`5a49e0a6acd672186af116ee31fe1409bd33c605b2846bfc5f9d4ae42d79907c`.
The trim probe ran in a fresh process; only after its success did the three
selected cache contracts run, with pytest `-x -p no:cacheprovider`.

```bash
export DDO_VMM_LIBRARY_PATH=/run/media/eliseu/AI2/ProjetosIA/diffusers-dynamic-offloader/ddo-vmm-allocator/build-linux-cgroup-review-20261012/libddo_vmm.so
.venv-linux/bin/python -B \
  ../diffusers-dynamic-offloader/ddo-vmm-allocator/scripts/vmm_idle_trim_probe.py \
  --library "$DDO_VMM_LIBRARY_PATH" && \
.venv-linux/bin/python -B -m pytest -q -x -p no:cacheprovider \
  ../diffusers-dynamic-offloader/ddo-vmm-allocator/tests/test_source_cache.py::test_persistent_storage_survives_registration_recycling_without_reread \
  ../diffusers-dynamic-offloader/ddo-vmm-allocator/tests/test_source_cache.py::test_immutable_batch_retains_sources_recycles_events_and_guards_close \
  ../diffusers-dynamic-offloader/ddo-vmm-allocator/tests/test_source_cache.py::test_live_tickets_protect_cache_close_and_clear
```

Probe returned `status: PASS`; pytest returned **3 passed in 1.16 s**, no skips,
exit 0. No retry, changed limits, model weights or benchmark ran.

| Native pool observation | Before trim | After trim | Final |
| --- | ---: | ---: | ---: |
| Mapped bytes | 35,651,584 | 2,097,152 | 0 |
| Active requested bytes | 1,024 | 1,024 | 0 |
| Pending bytes | 33,554,432 | 0 | 0 |
| Idle backing bytes | 33,554,432 | 0 | 0 |
| Live allocations | 1 | 1 | 0 |
| Failures / quarantined bytes | 0 / 0 | 0 / 0 | 0 / 0 |

Exact live payload, repeated trim and allocation after trim passed. Native
cumulative eviction wait was 0.000017713 s after the first trim and 0.000023995 s
at final cleanup; these are wait counters, not inference or process timings.
Cache contracts verified retained bytes without reread across registration
recycling, immutable source/batch ownership with async exact-payload copies,
and refusal to close/clear live tickets. This is small-contract acceptance,
not full-model registration capacity, driver-loss recovery or universal support.

The environment changed from the historical fork, so propose one targeted
generation separately before claiming model compatibility of this checkpoint.
Preserved binaries remain unmodified; no default promotion or full benchmark
matrix is authorized by this gate.

## Approved targeted official-Diffusers generation: PASS

One LTX image staged BF16 generation completed with the same candidate hash,
official Diffusers 0.41.0 and host `.venv-linux`. Native encoder and transformer,
resident VAE, real default prompt, 1280x704, seed 43, 8 steps, no PAG; fixed
4 GiB active weights, 1.5 GiB total staging and 24.625 GiB source/registration.
No budget/compute changes, trim, cuBLAS clear or model matrix expansion ran.

The PNG is byte-identical to the preserved Linux resident control, SHA256
`dfbf135a833b5a77f4574a98cebd3be84d19fc569b79de8dd5f18457c52d64c1`.
All eight CUDA timing intervals completed; reported fault/pool/admission/read
failures, temporary fallbacks and quarantined bytes were zero. No monitor
errors occurred. Transformer completion collection reduced 409 live tickets
to zero. Pre-unload source snapshots are not a final allocator-zero proof;
do not extrapolate the small trim probe's final counters to this model run.

Selected environment/checkpoint acceptance is closed; old Linux binary and
fork-based evidence remain preserved. Detailed phase/timing/memory evidence
belongs to sibling examples `experiments/dynamic_offload_results.md`, artifact
directory `outputs/ltx_image_official_linux_cgroup_gate_20261012`.
No video/Qwen rerun, binary promotion, commit or publication occurred. Remaining
work is platform audio reproducibility and the reviewed local checkpoint.

The host subsequently locked Linux/WSL TorchAudio to the explicit CPU index,
preserving Windows selection and CUDA Torch/media indices. `uv lock --check`
and an existing-venv locked/inexact sync dry-run passed; no actual sync or
package change followed. Audio reproducibility is closed at manifest/resolution
scope. Review local commits next; no fresh environment/platform matrix is implied.
