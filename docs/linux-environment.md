# Linux development environment (2026-10-04)

Native Linux setup is separate from the accepted Windows environment/build.
`vmm-dev-linux` is an optional Python 3.12 development profile, not a declaration
that the native Linux cache/transport is complete or safe on CUDA.

## Local setup observed

Ubuntu 26.04.1 LTS, x86_64, NVIDIA RTX 3060 Ti 8192 MiB, driver 595.91.07.
`uv` 0.12.20; Python 3.12.14 in DDO `.venv-linux`. The shared checkout is on
NTFS (`ntfs3`); editable imports and the Linux Python interpreter work there.
Use `--link-mode copy` for uv because its download cache is on another filesystem.
No Windows environment or candidate/default DLL was modified.

The Linux profile pins Torch 2.12.1+cu132, torchvision 0.27.1+cu132,
transformers 5.15.1, accelerate 1.14.0, peft 0.20.0, einops 0.8.2,
safetensors 0.8.0 and the same custom Diffusers revision as the Windows harness.
CUDA tools `cuda-toolkit[nvcc,cccl]==13.2.1` install inside this environment, alongside
the native compiler tools/headers. They do not replace the system NVIDIA driver.
Installed nvcc package version is13.2.78. Linux Torch wheels were verified on
the [official PyTorch index](https://download.pytorch.org/whl/cu132/torch/).
Driver-family compatibility has
[NVIDIA-documented limits](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html);
actual DDO CUDA feature/stream safety still needs its approved gate.

## Reproduce without replacing Windows environments

From the DDO root, only create the venv if it does not already exist:

```bash
uv venv .venv-linux --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-linux uv sync --locked --extra vmm-dev-linux --inexact --link-mode copy
uv pip install --python .venv-linux/bin/python --link-mode copy -e ./ddo-vmm-allocator
```

Always set `UV_PROJECT_ENVIRONMENT` for `uv sync`. `--python` alone chooses an
interpreter but does not prevent uv from replacing the project's `.venv`.
`--inexact` preserves the explicitly installed local allocator, which is not a
mandatory Python dependency of ordinary DDO installs. Both native and sibling
host validation use this DDO environment; no separate LTX interpreter is needed.
Do not install the full sibling project's default environment automatically:
its lock resolves for Linux, but its DDO dependency is an older remote revision.
The local editable DDO must remain authoritative for our native gates.

## System compiler prerequisites — user terminal

The machine initially had no C++ compiler, CMake or Ninja in PATH. These require
administrator authentication; Codex cannot complete interactive sudo here.
Run in your own terminal:

```bash
sudo apt update
sudo apt install --no-install-recommends build-essential cmake ninja-build
```

Do not install `nvidia-cuda-toolkit` from Ubuntu just for this environment: its
observed candidate is12.4, while this optional profile already supplies13.2
headers/nvcc tools. Do not change the working driver or use `apt autoremove` as
part of this setup. The C++ target uses CUDA Driver API, not a Python CPU cache.
The system compiler/linker are still needed even with nvcc wheels installed.

CPU contract build, after system tools are installed:

```bash
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode cpu --jobs 2
```

For the later library build, explicitly locate the environment's toolkit:

```bash
export CUDAToolkit_ROOT="$PWD/.venv-linux/lib/python3.12/site-packages/nvidia/cu13"
"$CUDAToolkit_ROOT/bin/nvcc" --version
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode library --dry-run
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode library --jobs 2
```

Finding headers/compiler is not proof of successful CMake discovery, linkage,
native transport portability or CUDA execution. Those remain separate gates.
The profile does not install/replace a driver or relax memory ceilings.

## Evidence in this preparation

`uv lock --check` passed and Linux profile `uv sync --locked --dry-run --inexact`
reported no changes after installation. Imports confirmed local DDO/allocator,
Torch/version pins and `torch.cuda.is_initialized() == False`. Sixty-six Python
CPU recipe/bootstrap/protected-route tests passed in2.94s. No CUDA allocation,
model inference, `.so` compilation or native C++ test execution occurred here.
An initial bootstrap test collection before editable DDO installation lacked
`ddo_vmm_bootstrap`; adding `PYTHONPATH=src` passed17 tests, and the installed
profile then passed the selected66 without that override.

After adding four platform/profile isolation contracts, the final selected
Python CPU sequence passed70 tests in3.03s, with no skips.

Continue through [Linux handoff](linux-handoff.md); do not infer native Linux
support from a successful dependency installation.

After the user installed system tools, GCC15.2.0/CMake4.2.3/Ninja1.13.2 were
verified. Native Linux storage/ring/cache/batches now compile into the .so;
nine C++ CPU and122 Python CPU contracts passed. CCCL was added to the optional
profile because CMake's toolkit includes it. The user-approved allocator probe
and seven small CUDA contracts then passed outside the sandbox (7 in3.64s,
no skips). [Current evidence](native-linux-transport-gate.md) supersedes the preparation-only status
above; ordinary users and Windows dependencies remain unchanged.

The later mode-aware admission correction passed nine C++ CPU and122 Python
CPU contracts in3.25s; corrected library compiled. Subsequently, the approved
three CUDA cache contracts and single staged real-model generation passed.
Original small CUDA evidence remains tied to the saved pre-correction .so.
Current artifact/limits and executed [staged model gate](native-linux-staged-gate.md)
are recorded separately; no sysctl/swap/driver/environment change was required.

## Local delivery check (2026-10-04; no CUDA rerun)

Final selected Python CPU sequence:122 PASS in3.42s, no skips; existing nine
native CPU executables PASS through CTest in0.04s. `uv lock --check` PASS.
Both Python wheels build successfully with `uv build --wheel`; content checks
confirm bootstrap/public reporting/allocator adapters and optional Linux extra
are included, without experimental reports, environments or mandatory CUDA
development dependencies in an ordinary install.

These are pure-Python wheels: neither bundles a DLL/.so. Native execution still
requires the initialized allocator checkout, its platform build and explicit
library selection as documented above. This check is not binary packaging,
a clean-machine installed-wheel CUDA test or publication approval. Distribution
of platform-specific native artifacts remains a release decision.

The validated .so hash remains unchanged after the delivery checks. Windows
candidate/default DLL hashes also match the accepted checkpoint. No runtime
source changed after the approved staged model gate.

Local allocator checkpoint: `433d3e47ff7a6e7f5635ebedf0acc3c3191f6308`, pinned
by the DDO Linux closure commit. No push or binary publication. Shared-checkout
end-of-line-only changes and unrelated scratch files were excluded from closure.
