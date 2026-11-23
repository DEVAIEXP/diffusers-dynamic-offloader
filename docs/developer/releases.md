# Package delivery

This maintainer guide explains how to prepare and publish DDO packages.
It is not a changelog or a list of local test candidates. For installation,
use the [user installation guide](../user/installation.md); for tested scope,
see [Validation](validation.md).

## Artifacts and ownership

### Published RC5

[DDO 0.1.0rc5](https://pypi.org/project/diffusers-dynamic-offloader/0.1.0rc5/)
and [allocator 0.1.0rc5](https://pypi.org/project/ddo-vmm-allocator/0.1.0rc5/)
were published on November 23, 2026. Downloaded PyPI files were checked against
the original SHA256 values:

| Wheel | SHA256 |
| --- | --- |
| `diffusers_dynamic_offloader-0.1.0rc5-py3-none-any.whl` | `d64e0558944a8630868eab6e4f4a8550c3ff497fe2153b4b84b5fe66cc3c5c77` |
| `ddo_vmm_allocator-0.1.0rc5-py3-none-win_amd64.whl` | `fd35ffbb62c842671b1f5c11d9560745d0af95cef41444e5c80ca85d76d8902e` |
| `ddo_vmm_allocator-0.1.0rc5-py3-none-manylinux_2_35_x86_64.whl` | `e3da8d27017b35bb8c7cb15cb5cd1357208ef5eeef7e6840c6be657cd9726b0f` |

Package sources are DDO `8fa41d82811c74caabb5ab96b00d4ebedf258c30`
and allocator `3f7924ce1b27bae4fae70afb478ae4313cc018b6`.
The Linux wheel was audited and repaired with auditwheel 6.8.2; it requires
glibc 2.35 or newer and leaves `libcuda.so.1` external. No NVIDIA driver is
bundled. Final Windows model acceptance and the remaining Linux/WSL CUDA
validation limit are described in [Validation](validation.md).

### Package contents

DDO provides the pure Python wheel. The allocator provides platform-specific
wheels containing the native library and, where supported, its matching
Torch-linked bridge. Model runners and benchmarks belong to DDO examples.

Build from reviewed source revisions and a pinned allocator gitlink. Record
package versions, source commits, API/ABI, build features, toolchain, dependencies
and SHA256 in the release manifest. NVIDIA driver libraries remain external.
Installation and inference do not compile or download native code.

## Prepare release tools

Use the locked tools from the DDO checkout and a separate environment for each
platform. The examples below target `0.1.0rc5`; select the reviewed CUDA toolkit
and compiler paths for the build machine.

### Native Linux

With uv, CMake, a C++20 compiler, readelf, nm and the reviewed CUDA toolkit:

```bash
uv run --locked --project tools/release python scripts/build_release.py \
  --version 0.1.0rc5 --cuda-toolkit /path/to/reviewed/cu13 --dry-run
```

Review the plan before removing `--dry-run`. A host build inherits the host's
C/C++ ABI requirements. For an Ubuntu 22.04 baseline, use the Docker route below.

### Windows

Use Python 3.12, MSVC 2022 x64 Build Tools, CMake and CUDA 13.2:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$PWD/tools/release/.venv-windows"
uv sync --locked --project tools/release --link-mode copy
& "$env:UV_PROJECT_ENVIRONMENT/Scripts/python.exe" -B scripts/build_release.py `
  --version 0.1.0rc5 `
  --cuda-toolkit 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2' `
  --dumpbin 'C:/path/to/reviewed/MSVC/bin/Hostx64/x64/dumpbin.exe' `
  --dry-run
```

Select the installed dumpbin path and review the plan before removing
`--dry-run`. The allocator wheel targets Windows AMD64; DDO remains pure Python.

### Linux packages built through Docker

From Windows with Docker Desktop providing Linux x86-64 containers:

```powershell
python -B scripts/build_release_docker.py --version 0.1.0rc5 --dry-run
```

The wrapper uses a pinned CUDA 13.2.1 Ubuntu 22.04 image and locked tools.
The candidate build runs without GPU or network access, with two CPUs and a
6 GiB memory/swap ceiling; image preparation obtains build dependencies.
The output must stay within the declared Ubuntu 22.04 ABI baseline.
The initial wheel is tagged `linux_x86_64`, not automatically manylinux.

Validate the resulting Linux package separately on native Linux and WSL.
Do not install a Linux NVIDIA kernel driver inside WSL.

## Validate and publish

1. Use a new candidate directory; never overwrite accepted artifacts. Review
   the native features and compatibility described in [Native builds](builds.md).
2. Verify wheels, `SHA256SUMS` and `release-manifest.json`; retain Docker
   provenance when applicable. Exclude source snapshots, logs, environments
   and model weights from distribution assets.
3. Validate metadata with `python -m twine check --strict <wheel-files>`.
   Published dependencies, including extras, must not contain direct Git/URL
   requirements. Use release tools compatible with the package metadata version.
4. Audit Linux wheels before applying a manylinux tag. Resolve external driver
   linkage without bundling the NVIDIA driver; a filename change is not an audit.
   Record any repair-induced changes and validate the final wheel bytes.
5. Install the exact final pair in clean target environments. Check discovery,
   linkage, compatibility and separately approved GPU/model behavior according
   to [Validation](validation.md). Container CPU success does not certify CUDA.
6. After explicit publication approval, upload the validated bytes and verify
   registry metadata and downloaded hashes. Published files are immutable;
   corrections require a new release.
7. Update release notes, compatibility scope and installation instructions to
   match the actual distributed artifacts.

TestPyPI is separate from PyPI. For TestPyPI checks, install third-party
dependencies from their production indexes, then install the exact DDO/allocator
test packages with `--no-deps`. Keep credentials out of source, logs and docs.
Rebuilding, repairing or changing metadata changes artifact identity; previous
acceptance does not automatically transfer to the changed wheel.

## GitHub workflows

`release-candidates.yml` builds candidates through manual dispatch. It needs
a dedicated trusted self-hosted Windows x64 runner with the `ddo-release` label,
PowerShell 7, Git, uv, Python 3.12, CMake, MSVC, CUDA 13.2 and Docker Desktop.
Configure `DDO_RELEASE_CUDA132` and `DDO_RELEASE_DUMPBIN`. It does not publish,
create tags, run CUDA or download models. Its artifact retention is not a backup.

After a version is published and verified on PyPI, `release-draft.yml` can
prepare a draft prerelease using that version and its original source commit.
Run it separately for DDO and the allocator. It downloads existing wheels and
checks metadata/hashes; it does not build or upload packages to PyPI. Review
assets, source identities and notes before separately authorizing publication
of the draft and its tag.

Earlier published artifact records are retained in
[Validation history](validation-history.md#published-rc3). Local development
candidate records are maintained separately from these public instructions.
