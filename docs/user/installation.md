# Installation

DDO and its native allocator are available as experimental `0.1.0rc2` packages
on PyPI. Use the matching versions
below; installing a prerelease does not require building native code.

## Choose your starting point

To run the provided models, start with
[DDO examples setup](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/user/setup.md).
Clone only `ddo-examples`; its environment holds DDO, the allocator and model
dependencies together. You do not need DDO source clones or a second environment.

For your own runner, follow the standalone installation below. Keep Windows,
native Linux and WSL environments separate. You can reuse an existing compatible
environment instead of creating another one.

## Prerequisites

The selected package environment uses Python 3.12 and CUDA-enabled PyTorch
2.12.1+cu132. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and a compatible NVIDIA system driver. Native packages support Windows AMD64
and Linux x86-64 with glibc 2.35 or later and a compatible C++ runtime.
WSL uses the Windows GPU driver; do not install a Linux GPU driver there.

No compiler, CUDA development toolkit, Docker or manual DLL/.so path is needed.
Use uv-managed Python to avoid inheriting incompatible Conda runtime libraries.
Leave `PYTHONPATH` and `DDO_VMM_LIBRARY_PATH` overrides unset.

## Install the standalone packages

Run each command separately and stop if one fails. Choose a new environment
directory. If reusing an environment, skip `uv venv` and set `python` to its
interpreter instead. Explicit interpreter paths mean activation is not required.

Install CUDA-enabled Torch from the PyTorch index, then install DDO, the allocator
and their dependencies from PyPI.

### Windows PowerShell

```powershell
uv venv --managed-python --python 3.12.12 .venv-ddo-rc2
$python = (Resolve-Path '.venv-ddo-rc2/Scripts/python.exe').Path
uv pip install --python $python --link-mode copy --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132'
uv pip install --python $python --link-mode copy --index-url https://pypi.org/simple/ 'diffusers-dynamic-offloader[vmm]==0.1.0rc2' 'ddo-vmm-allocator==0.1.0rc2' 'diffusers==0.41.0' 'accelerate==1.14.0' 'safetensors==0.8.0' 'psutil==7.2.2' 'numpy==2.5.3' 'cuda-core==1.2.0' 'cuda-bindings==13.4.1'
uv pip check --python $python
```

### Native Linux or WSL Bash

```bash
uv venv --managed-python --python 3.12.12 .venv-ddo-rc2
python="$PWD/.venv-ddo-rc2/bin/python"
uv pip install --python "$python" --link-mode copy --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132'
uv pip install --python "$python" --link-mode copy --index-url https://pypi.org/simple/ 'diffusers-dynamic-offloader[vmm]==0.1.0rc2' 'ddo-vmm-allocator==0.1.0rc2' 'diffusers==0.41.0' 'accelerate==1.14.0' 'safetensors==0.8.0' 'psutil==7.2.2' 'numpy==2.5.3' 'cuda-core==1.2.0' 'cuda-bindings==13.4.1'
uv pip check --python "$python"
```

## Verify installation

PowerShell:

```powershell
& $python -I -B -c "import importlib.metadata as m, torch, diffusers_dynamic_offloader as d; print('DDO:', m.version('diffusers-dynamic-offloader'), d.__file__); print('Allocator:', m.version('ddo-vmm-allocator')); print('Torch:', torch.__version__); print('CUDA initialized:', torch.cuda.is_initialized())"
```

Bash:

```bash
"$python" -I -B -c "import importlib.metadata as m, torch, diffusers_dynamic_offloader as d; print('DDO:', m.version('diffusers-dynamic-offloader'), d.__file__); print('Allocator:', m.version('ddo-vmm-allocator')); print('Torch:', torch.__version__); print('CUDA initialized:', torch.cuda.is_initialized())"
```

Expected: version `0.1.0rc2`, installed `site-packages` paths and
`CUDA initialized: False`. Importing the package does not start CUDA; this does
not mean GPU support is missing.

## Check the native allocator

This optional check runs a small GPU operation, not a model benchmark.
`--run-cuda` explicitly authorizes GPU execution; `--help` does not start CUDA.

PowerShell:

```powershell
& $python -I -B -m ddo_native_check --run-cuda
```

Bash:

```bash
"$python" -I -B -m ddo_native_check --run-cuda
```

Expected: the packaged library path and `DDO native CUDA: PASS`. The check
bootstraps a fresh child, installs the allocator before CUDA, and checks two
4 KiB tensors with a 32 MiB allocator cache limit. That is not a total GPU-memory
cap or model-performance validation. Stop and preserve the error if it fails.

## Integration and model dependencies

For your runner, follow [native integration](native-vmm.md) or
[linear integration](full-pipeline.md). Linear and compatibility routes do not
need the native allocator.

DDO installation does not download weights or custom blocks or install all
video, audio and quantization dependencies. Those requirements and runnable
recipes belong to [DDO examples](https://github.com/DEVAIEXP/ddo-examples).
Use its documented versions together in the same environment.

## Source development

Source/editable installation and local-wheel validation are contributor tasks.
See [development installation](../developer/installation.md).
