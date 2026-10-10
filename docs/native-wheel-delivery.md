# Local Windows native wheel delivery

The optional allocator wheel includes the selected existing AMD64 DLL. DDO itself
remains a pure Python wheel; ordinary installations do not acquire this allocator
or native development tools automatically. No binary is compiled at import or
runner startup, and no library path is required for the packaged route.

## Selected local artifacts (2026-10-10)

- `dist/windows/ddo_vmm_allocator-0.1.0-py3-none-win_amd64.whl`
  SHA256 `9bb050bb800007d0fa09ddaf98903ea1007800af545ffe6b750ec5b0fa20e28c`.
- `dist/windows/diffusers_dynamic_offloader-0.1.0-py3-none-any.whl`
  SHA256 `9f7a05d949ab9bf8f6f5048ed6c109bcb5ee39b19d102a0a439388d4428504a6`.
- Embedded DLL SHA256
  `ec5ab4700ee5d171cc06d8e9ec99c549f8d6cc3dfbe7feb2683b93add0e829e6`.

These are local development artifacts, not published releases or a new version
claim. A later assembly may have a different wheel hash while preserving the DLL;
verify its manifest and RECORD rather than treating this wheel hash as universal.
Allocator source/packaging checkpoint is `c925ee1`; the DDO wheel contains the
runtime from `09d101c` (subsequent delivery changes add docs/probes only).
The DLL is the selected joined-span Windows artifact, not a newly compiled default
CMake build. CUDA toolkit evidence is 12.8.93 / MSVC VS2022 x64, span pool and
joined spans ON, source dispatcher/test hooks OFF, allocation diagnostics ON.
This preserves the selected artifact, not a claim that every diagnostic capability
should be enabled at runtime. Python adapter revisions can change independently
of the native library; FP8 added Python payload/ownership adapters, not C++ changes.

## Install into a prepared host environment

From the DDO checkout, substituting the intended host interpreter:

```powershell
uv pip install --python <host-python> `
  dist/windows/diffusers_dynamic_offloader-0.1.0-py3-none-any.whl `
  dist/windows/ddo_vmm_allocator-0.1.0-py3-none-win_amd64.whl
```

Keep the supported CUDA-enabled Torch/Diffusers environment and optional native
runtime dependencies prepared as documented in [native usage](native-vmm.md).
Installing the wheel does not validate an arbitrary Torch/driver combination.
TorchAO remains optional and is needed only for the documented FP8 recipe.
Do not replace the accepted editable benchmark environment solely for this smoke
test. Ordinary linear use needs neither the allocator wheel nor this DLL.

The existing runner bootstrap discovers
`ddo_vmm_allocator/native/win32/ddo_vmm.dll` in the installed package. Explicit
diagnostic overrides and conventional source-checkout builds retain their documented
precedence. This installation does not scan experimental build folders or silently
change a benchmark's selected binary. See [discovery rules](native-vmm.md).

## Reproduce the assembly (maintainers only)

Use an isolated packaging environment with `build` and `hatchling>=1.27` installed.
The opt-in build hook requires both the selected DLL and its expected hash:

```powershell
$env:DDO_VMM_WHEEL_LIBRARY = (Resolve-Path `
  ddo-vmm-allocator/build-joined-spans-cu128-20261006/Release/ddo_vmm.dll).Path
$env:DDO_VMM_WHEEL_SHA256 = 'EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6'
python -m build ddo-vmm-allocator --wheel --outdir dist/windows
Remove-Item Env:DDO_VMM_WHEEL_LIBRARY, Env:DDO_VMM_WHEEL_SHA256
python -m build . --wheel --outdir dist/windows
```

These build variables are maintainer artifact-selection inputs, not runner settings
or end-user configuration. The hook refuses a mismatched hash or a non-AMD64/non-DLL
PE header, snapshots verified bytes, records `artifact.json`, and emits
`py3-none-win_amd64` with `Root-Is-Purelib: false`. It never loads the DLL.
Partial opt-in selection fails; with both variables absent,
the allocator wheel stays `py3-none-any` and contains no DLL. Sdist builds do not
embed the selected binary. Linux native-wheel assembly is not implemented here.

## Executed CPU delivery checks

33 packaging/bootstrap CPU contracts passed in 0.30 s. Native and ordinary wheels
passed archive, platform tag, license, manifest/SHA256 and RECORD integrity checks.
A fresh Python environment installed both wheels without dependencies; an isolated
`-I` process verified package discovery and mocked child launch without source-tree
imports, `PYTHONPATH`, Torch, CUDA or execution of the allocator initializer.

Reusable checks:

```powershell
python scripts/verify_native_delivery_cpu.py `
  dist/windows/ddo_vmm_allocator-0.1.0-py3-none-win_amd64.whl `
  --native-sha256 EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6
python scripts/verify_native_delivery_cpu.py `
  dist/windows/diffusers_dynamic_offloader-0.1.0-py3-none-any.whl
# Run with an isolated interpreter where these wheels are already installed:
python -I -B scripts/probe_installed_native_cpu.py
```

The dependency-free installation isolates package delivery only: it is not a
model-ready environment. No DLL loading, CUDA execution, inference or performance
gate was run from the wheels. Those require a separately approved bounded check;
existing source-checkout evidence is not silently transferred to wheel execution.
No push/publication or history cleanup occurred. Preserve Windows artifacts when
continuing Linux; its BF16 .so acceptance is separate from FP8 and wheel validation.
