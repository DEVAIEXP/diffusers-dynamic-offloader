# Package delivery

## Ownership and artifacts

The allocator repository owns platform-native DLL/.so wheels. DDO owns its
pure Python wheel and compatible allocator dependency. Examples owns model
runners and benchmarks, not native distribution.

Build from reviewed source revisions with the allocator submodule initialized.
Record source identity, API/ABI, native features, toolchain, dependencies and
SHA256. Install and validate the actual wheel pair, then distribute those
same bytes without rebuilding. NVIDIA driver libraries remain external.
There is no import-time or inference-time build/download.

## One-command native Linux candidate generation

From the DDO root, with uv, CMake, a C++20 compiler, readelf, nm and a reviewed
CUDA toolkit installed:

```bash
uv run --locked --project tools/release python scripts/build_release.py \
  --version 0.1.0rc3 --cuda-toolkit /path/to/reviewed/cu13 --dry-run
```

The version is an example for a new candidate, not a published package claim.
Review the plan and remove `--dry-run` to build. Existing output directories
are refused. Dirty sources are refused unless `--allow-dirty` explicitly
selects a development snapshot; that snapshot is not a clean tagged release.
`--allocator-version` selects a different reviewed package pairing.

The builder uses locked release tools, isolated source snapshots and a new
CPU installation environment. It compiles native CPU contracts, audits ELF
dependencies/symbols/exports, verifies wheels and checks installed discovery.
It does not execute CUDA. Default output:
`.internal/releases/<version>/linux-x86_64/`.

A host-built Linux wheel inherits that host's C/C++ ABI requirements. Building
on a newer distribution does not make it installable on Ubuntu/WSL 22.04.
Use the Docker ABI-baseline recipe for that target; neither wheel is
automatically manylinux-compliant.

## Docker candidate from Windows (Ubuntu 22.04 ABI baseline)

```powershell
python -B scripts/build_release_docker.py --version 0.1.0rc3 --dry-run
```

Docker Desktop must provide Linux x86-64 containers. The recipe pins its CUDA
13.2.1 Ubuntu 22.04 image by digest and its release tools. Image preparation
downloads build dependencies, not models. Candidate execution has no GPU or
network, two CPUs and a 6 GiB memory/swap ceiling. Source is mounted read-only;
output uses a separate directory.

The wrapper invokes the Linux builder, including CPU contracts and installed
discovery. It refuses requirements above GLIBC 2.35, GLIBCXX 3.4.30 or CXXABI
1.3.13. The toolkit link stub is not distributed; the host supplies libcuda.
The wheel tag remains `linux_x86_64`, not manylinux.

Default output: `.internal/releases/<version>/docker-linux-x86_64/`.
Retain Docker provenance with the release manifest. Validate the unchanged
wheel separately on native Linux and WSL; container CPU success is not CUDA
acceptance.

## Windows native candidate generation

Use Python 3.12, uv, CMake >=3.25, MSVC 2022 x64 Build Tools and CUDA 13.2.
Keep Windows release tools separate from Linux environments on shared storage.

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$PWD/tools/release/.venv-windows"
uv sync --locked --project tools/release --link-mode copy
& "$env:UV_PROJECT_ENVIRONMENT/Scripts/python.exe" -B scripts/build_release.py `
  --version 0.1.0rc3 `
  --cuda-toolkit 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2' `
  --dumpbin 'C:/path/to/reviewed/MSVC/bin/Hostx64/x64/dumpbin.exe' `
  --dry-run
```

Select the installed dumpbin path, review the plan and remove `--dry-run`.
The builder compiles native CPU contracts and a Release AMD64 DLL, audits PE
dependencies/exports without loading it and checks clean-wheel discovery.
The allocator wheel is `py3-none-win_amd64`; DDO remains pure Python.

The Windows profile enables span pool, joined spans and allocation diagnostics;
Linux enables span pool without the latter two features. Probes, execution
kernels, source dispatcher and test hooks are disabled in both profiles.
Review explicit features rather than assuming CMake defaults.
Default output: `.internal/releases/<version>/windows-amd64/`.

## Distribution and validation

Each candidate's `dist/` contains both wheels, `SHA256SUMS` and
`release-manifest.json`; Docker adds its provenance record. Source snapshots,
private logs and test environments are not distribution assets.

Follow [installation](../user/installation.md) and [validation](validation.md)
for clean-environment checks. CPU discovery is separate from installed CUDA
validation and from full-model acceptance. Never overwrite accepted binaries
or retag a host build as manylinux without its required audit.

## GitHub candidate workflow

`.github/workflows/release-candidates.yml` uses manual `workflow_dispatch`
with an explicit prerelease version. It invokes the local builders for Windows
CUDA 13.2 and Docker Linux. It does not publish packages, create tags/releases,
run CUDA or download models.

The workflow requires a dedicated trusted self-hosted Windows x64 runner with
the `ddo-release` label, PowerShell 7, Git, uv, Python 3.12, CMake, MSVC, CUDA
13.2 and Docker Desktop. Define `DDO_RELEASE_CUDA132` and
`DDO_RELEASE_DUMPBIN` as the toolkit and dumpbin paths. Windows symlink tests
require developer mode or symlink privilege. Do not use a personal inference
checkout as the runner's disposable workspace.

Only distribution files are retained as Actions artifacts, for 14 days.
Actions retention is not permanent package publication or backup.
Clean CI candidates need their own installed validation; local dirty builds
do not certify them.

## Publication checklist

1. Review source revisions, allocator gitlink, version pairing and required notices.
2. Build candidates and verify manifests/checksums.
3. Validate the exact installed artifacts on the intended targets.
4. Publish those bytes with compatibility and installation instructions.
5. Update user documentation to the actual package-index/release availability.

Package-index upload is not implemented by the candidate workflow. Keep
credentials out of source and private records out of release assets.

## Package-index preparation

Local wheel acceptance and index acceptance are separate gates. Before uploading
to TestPyPI or PyPI:

1. Ensure published dependency metadata contains no direct URL or Git requirements,
   including optional extras. Checkout-only `diffusers-mm` is in the
   `vmm-harness` dependency group, not a published extra. Maintainers who need it
   can install it explicitly with `uv pip install --group vmm-harness` from this
   checkout and their selected environment.
2. Run metadata validation on the newly assembled wheels. Earlier local rc2 DDO
   wheels include the Git requirement and must not be uploaded as-is. Preserve
   them as local validation artifacts; new metadata means a new wheel hash.
   Use `twine==7.0.0` with `packaging==26.3` for the current candidates and run
   `python -m twine check --strict <wheel-files>`. The allocator uses Core
   Metadata 2.5; Twine 6.2.0 in the earlier release-tool environment rejected
   that version. Metadata validation does not upload packages or test CUDA.
3. Audit Linux wheels against an appropriate manylinux policy before assigning
   that tag. The current `linux_x86_64` candidate is for local installation, not
   ready for index publication. An Ubuntu ABI baseline alone does not establish
   manylinux compliance. Resolve external CUDA-driver linkage without bundling
   the system driver; do not merely rename the wheel or suppress audit failures.
4. Keep the native library bytes and their hashes when possible. Record any
   repair-induced change and revalidate the resulting installed artifact.
5. Close source commits and the allocator gitlink, then record the exact wheel
   set and checksums. TestPyPI credentials and project ownership must be configured
   before an explicitly approved upload. Never paste tokens into documentation,
   commands recorded in the repository or chat.

TestPyPI is a separate registry with separate accounts. For installation tests,
obtain third-party dependencies from their documented production indexes, then
install the exact DDO/allocator versions from TestPyPI with `--no-deps`. Do not
resolve arbitrary third-party dependencies from the test registry. Repeat the
installed native check and the selected model recipe on Windows and Linux;
Windows model acceptance does not establish Linux model acceptance.

References: [direct URL restrictions](https://setuptools.pypa.io/en/stable/userguide/dependency_management.html#direct-url-dependencies),
[manylinux policies and audit](https://github.com/pypa/auditwheel), and
[using TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/).
