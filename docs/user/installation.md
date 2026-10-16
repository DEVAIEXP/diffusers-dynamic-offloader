# Installation and experimental support
DDO is a Python offload router. Its normal linear and compatibility routes do not
require the native allocator, compiler or CUDA development toolkit.

## One environment for libraries and runners

DDO, the optional native allocator, Diffusers, model dependencies and runners can
all use the same virtual environment. You do not need a separate environment
just to run DDO. Keep Windows, native Linux and WSL environments separate.

For release validation only, use a new environment to check that the installed
packages work without relying on a development checkout or an old native build.
The wheel examples below show that clean-install check; the same environment can
then run model scripts once their dependencies are installed.

## Choose your starting point

To run the provided models, start with
[DDO examples setup](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/user/setup.md):
clone only that repository, create one environment and install DDO as a package.
No DDO source clone is needed. If you already installed DDO separately, keep that
environment and add the examples' dependencies there; you do not need to start over.

For your own runner, install the wheel pair below without cloning either source
repository. Source development is a separate option, not a user prerequisite.

## Source development

To modify DDO itself, clone its source and initialize the allocator submodule:

```sh
git clone --recurse-submodules https://github.com/DEVAIEXP/diffusers-dynamic-offloader.git
cd diffusers-dynamic-offloader
```

Create or activate your own environment, install compatible CUDA-enabled PyTorch
and Diffusers, then install from the reviewed checkout:
```bash
uv pip install -e .
```
An editable installation does not supply a native DLL/.so. Maintainers build native
candidates separately using the [release procedure](../developer/releases.md).

## Native VMM from a local wheel pair

Use two reviewed wheels from the same candidate set: the platform allocator wheel
and the pure Python DDO wheel, together with their SHA256SUMS and release manifest.
Python 3.12 and Torch 2.12.1+cu132 are the selected delivery-test environment;
broader package metadata is not a promise that every dependency pairing was tested.
The commands below use development candidate `0.1.0rc2`, not an available PyPI
release. Obtain the actual files from the maintainer/local builder; do not substitute
an unrelated package or assume that a GitHub tag already has release assets.

Prerequisites: uv, Python 3.12, a compatible NVIDIA GPU/system driver and a matching
Windows AMD64 or Linux x86-64 wheel. End users installing these wheels do not need
MSVC, CMake, nvcc or Docker. Linux/WSL need compatible system C/C++ runtimes; use
the reviewed Ubuntu 22.04-baseline wheel rather than the newer host-built wheel.
On WSL, the GPU driver comes from Windows; do not install a Linux GPU driver.

Use `uv pip --python <environment-python>` to install the wheels into your chosen
environment. The examples create a new one for release validation. If you already
have a compatible runner environment, reuse its Python and skip environment
creation instead.

Follow the examples repository's user setup to add runner dependencies to this
same environment. Its developer setup explicitly replaces DDO/allocator wheels
with editable sources; do not use that path for wheel validation. Leave
`PYTHONPATH` and `DDO_VMM_LIBRARY_PATH` overrides unset so imports and native
discovery use the installed packages.

### Windows PowerShell

Set `$dist` to the folder containing the reviewed Windows pair. All paths below
are independent of the current PowerShell directory. This example creates a new
environment; when reusing an existing one, set `$environment` to its path and omit
the existence check and `uv venv` command.

```powershell
$dist = (Resolve-Path 'C:/path/to/windows-dist').Path
$environment = Join-Path $PWD '.venv-ddo-rc2'
if (Test-Path -LiteralPath $environment) { throw 'Choose a new environment directory' }
Get-Content (Join-Path $dist 'SHA256SUMS') | ForEach-Object {
    if ($_ -notmatch '^([0-9a-f]{64})  ([A-Za-z0-9_.+-]+\.whl)$') { throw 'Invalid checksum entry' }
    $expected = $Matches[1]
    $wheel = Join-Path $dist $Matches[2]
    if ((Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw "Checksum mismatch: $wheel"
    }
}
uv venv --python 3.12 $environment
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
$python = Join-Path $environment 'Scripts/python.exe'
uv pip install --python $python --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132'
if ($LASTEXITCODE -ne 0) { throw 'Torch installation failed' }
$allocator = Join-Path $dist 'ddo_vmm_allocator-0.1.0rc2-py3-none-win_amd64.whl'
$ddo = Join-Path $dist 'diffusers_dynamic_offloader-0.1.0rc2-py3-none-any.whl'
uv pip install --python $python $allocator "${ddo}[vmm]" 'diffusers==0.41.0'
if ($LASTEXITCODE -ne 0) { throw 'Wheel/dependency installation failed' }
uv pip check --python $python
```

### Native Linux or WSL Bash

Use the same Docker-built Linux wheel bytes on the two targets, but keep their
environments and validation results separate. This example creates a new
environment; when reusing an existing one, set `environment` to its path and omit
`test ! -e` and `uv venv`. Stop if any command fails.

```bash
(
set -eu
dist=$(realpath /path/to/linux-dist)
environment="$PWD/.venv-ddo-rc2"
test ! -e "$environment"
(cd "$dist" && sha256sum -c SHA256SUMS)
uv venv --python 3.12 "$environment"
python="$environment/bin/python"
uv pip install --python "$python" --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132'
uv pip install --python "$python" \
  "$dist/ddo_vmm_allocator-0.1.0rc2-py3-none-linux_x86_64.whl" \
  "$dist/diffusers_dynamic_offloader-0.1.0rc2-py3-none-any.whl[vmm]" \
  'diffusers==0.41.0'
uv pip check --python "$python"
)
```

### Verify installation before inference

Use the environment created or selected above. These commands also work in a new
terminal if you set the environment path first.

Windows PowerShell:

```powershell
$environment = Join-Path $PWD '.venv-ddo-rc2'
$python = Join-Path $environment 'Scripts/python.exe'
& $python -I -B -c "import importlib.metadata as m, torch, diffusers_dynamic_offloader as d; print('DDO:', m.version('diffusers-dynamic-offloader'), d.__file__); print('Allocator:', m.version('ddo-vmm-allocator')); print('Torch:', torch.__version__); print('CUDA initialized:', torch.cuda.is_initialized())"
```

Native Linux or WSL Bash:

```bash
environment="$PWD/.venv-ddo-rc2"
python="$environment/bin/python"
"$python" -I -B -c "import importlib.metadata as m, torch, diffusers_dynamic_offloader as d; print('DDO:', m.version('diffusers-dynamic-offloader'), d.__file__); print('Allocator:', m.version('ddo-vmm-allocator')); print('Torch:', torch.__version__); print('CUDA initialized:', torch.cuda.is_initialized())"
```

If you reused another environment, replace `.venv-ddo-rc2` with its path.

Expected: installed `site-packages` paths, the package versions you installed and
`CUDA initialized: False`. This confirms that the Python packages import without
starting CUDA. It does not mean CUDA is unavailable. Continue with the checks
below to test GPU execution.

### Check CUDA

This check uses ordinary PyTorch allocation, not DDO's native allocator. It creates
two small tensors on GPU 0, checks their values and waits for GPU completion.
No model is loaded. Run each check separately and stop if one fails.

Windows PowerShell, using `$python` from the previous step:

```powershell
& $python -I -B -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.ones(1024, device='cuda:0', dtype=torch.int32); y=x+1; torch.cuda.synchronize(); assert y.sum().item()==2048; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch CUDA: PASS')"
```

Native Linux or WSL Bash:

```bash
"$python" -I -B -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; x=torch.ones(1024, device='cuda:0', dtype=torch.int32); y=x+1; torch.cuda.synchronize(); assert y.sum().item()==2048; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch CUDA: PASS')"
```

Expected: your GPU name and `PyTorch CUDA: PASS`. If this fails, resolve the
PyTorch/driver error before testing native VMM.

### Check the installed native allocator

DDO provides `ddo_native_check`; you do not need to write a test script. The
`--run-cuda` flag explicitly enables GPU execution. `--help` does not start CUDA.

Use the updated DDO wheel containing this check. If Python reports
`No module named ddo_native_check`, reinstall the updated DDO wheel in the same
environment; early local `0.1.0rc2` candidates did not include it. The allocator
wheel does not need to change for this addition.

Windows PowerShell:

```powershell
& $python -I -B -m ddo_native_check --run-cuda
```

Native Linux or WSL Bash:

```bash
"$python" -I -B -m ddo_native_check --run-cuda
```

Expected: a native library path inside the environment's `site-packages` and
`DDO native CUDA: PASS`. Keep diagnostic library-path overrides unset. Bootstrap
finds the packaged DLL/.so, starts a fresh child process and installs the allocator
before GPU allocation. This check requests two 4 KiB tensors and sets the allocator
cache limit to 32 MiB; that limit is not a cap on total GPU memory usage.

Passing confirms basic CUDA execution through the packaged native allocator.
It does not validate model-weight streaming, model output or performance. A
failure is not a reason to increase memory limits or retry automatically; keep
the error for diagnosis. Next, follow [native usage](native-vmm.md) to integrate
your runner, or choose a model guide in
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).

## Integration and model dependencies

For `vmm_fast`, use DDO's bootstrap to start the runner in a child process and
install the allocator before CUDA starts. A compatible allocator wheel supplies
the native library: no compilation or manual DLL/.so path is needed. See
[native usage](native-vmm.md) for integration.

Installing DDO does not download model weights or custom blocks, or install all
video, audio and quantization dependencies. Follow the model guide in
[DDO examples](https://github.com/DEVAIEXP/ddo-examples) for those requirements
and runnable commands. Use its documented dependency versions together.
