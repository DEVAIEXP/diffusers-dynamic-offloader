# Development installation

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
candidates separately using the [release procedure](releases.md).

## Native VMM from a local wheel pair

Use two reviewed wheels from the same candidate set: the platform allocator wheel
and the pure Python DDO wheel, together with their SHA256SUMS and release manifest.
Python 3.12 and Torch 2.12.1+cu132 are the selected delivery-test environment;
broader package metadata is not a promise that every dependency pairing was tested.
For local candidate validation, obtain the files from the builder. Published
PyPI packages use the [user installation guide](../user/installation.md).
The filename below selects the audited manylinux candidate; an unprocessed
builder output has a different tag and requires its own compatibility checks.

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
uv venv --managed-python --python 3.12.12 $environment
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
uv venv --managed-python --python 3.12.12 "$environment"
python="$environment/bin/python"
uv pip install --python "$python" --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132'
uv pip install --python "$python" \
  "$dist/ddo_vmm_allocator-0.1.0rc2-py3-none-manylinux_2_35_x86_64.whl" \
  "$dist/diffusers_dynamic_offloader-0.1.0rc2-py3-none-any.whl[vmm]" \
  'diffusers==0.41.0'
uv pip check --python "$python"
)
```


After installation, use the import and native checks in the
[user installation guide](../user/installation.md#verify-installation).
