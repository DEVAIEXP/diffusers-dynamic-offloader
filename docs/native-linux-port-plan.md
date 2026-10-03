# Native Linux portability after Windows acceptance

## Scope

Windows selected performance/critical robustness is accepted for development;
do not reopen performance tuning or add an exhaustive model/platform matrix.
Preserve its candidate, source/event contracts and protected linear/NVMe routes.
Encoder native ownership is working for the validated text-only case, remains
opt-in, and should use the same public API on Linux rather than a separate runner
implementation. See [support checkpoint](native-support-status.md).

This is a port plan based on local source inspection, not a completed Linux
build or execution. Current host is Windows; no native Linux CUDA environment
has been used for this checkpoint. Do not install a VM/WSL/container or use a
different host without agreement. A real Linux build will produce
`libddo_vmm.so`; renaming/copying the Windows DLL cannot do this.

## Concrete gaps observed

These observations describe the pre-port inspection; the implementation
checkpoint below supersedes the file-source and bootstrap gaps only. Native
cache/ring/admission/batch portability remains incomplete.

- CMake already defines the shared `ddo_vmm` target, uses C++20 and CUDA Driver,
  selects non-MSVC warning flags and links DXGI only on Windows. Compilation on
  Linux still needs verification; this is not evidence of route support.
- `native/src/source_io.h` exposes positioned file/ring/storage operations only
  within `_WIN32`. `ddo_vmm_source.cpp` returns unsupported for non-Windows open,
  validation and read. File identity/generation and retained owners need POSIX
  implementations, not a CPU-Tensor loader/cache.
- `ddo_vmm.cpp` native shard/ring transport and `ddo_vmm_source_cache.cpp`
  admission, storage/registration/trim and batch submission have Windows-only
  branches. Cache creation/batch execution cannot be assumed functional after
  loading a Linux library.
- `src/ddo_vmm_bootstrap.py` accepts an explicit library path but its implicit
  checkout discovery currently searches `build/Release/ddo_vmm.dll`. Linux
  discovery must select the actual `.so` without weakening fresh-process setup.
- System headroom and Windows DXGI checks must not be silently reported as Linux
  observations. Driver memory, native accounting and host/cgroup limits are
  separate quantities. Windows standby purge remains its existing safe no-op
  outside Windows; it must not become a Linux requirement.

## One coherent implementation block

1. Introduce platform primitives in the allocator submodule for immutable regular
   file handles, positioned reads, source metadata validation and native host
   storage. POSIX `open`/`fstat`/`pread` and anonymous mapping are implementation
   candidates; preserve error propagation, partial reads/EINTR handling,
   short-read/range checks and file/worker lifetimes. Platform details must not
   escape into the DDO runner or public ABI.
2. Connect native ring/cache/batches to those primitives on Linux. Retain separate
   committed versus CUDA-registered storage; unregistering does not discard bytes
   or reread files. Preserve aggregate ceilings, DMA/compute events, reusable
   slots, partial-submit ownership and clear/close refusal when unsafe.
3. Add conservative native Linux host/admission observations, including actual
   process/container constraints where applicable. Missing/invalid headroom must
   produce an explicit refusal/diagnostic, not invented available capacity or
   automatic ceiling expansion. Do not emulate WDDM/shared residency on Linux.
4. Verify shared-library exports/link dependencies, CUDA context/stream ownership
   and CMake thread linkage. Keep the accepted feature selection explicit:
   span pool ON, source dispatcher OFF, lifecycle test hooks OFF. Do not change
   default build options or replace the accepted Windows artifact.
5. Make platform-library discovery and loading coherent in DDO/allocator;
   explicit path remains diagnostic/development selection. Bootstrap still
   installs the native PyTorch allocator before CUDA. Document editable install,
   build path and ABI requirements; do not force Linux to install the Windows
   development profile or model-specific host dependencies.

## Delivery and bounded validation

Deliver a Linux build recipe/script plus the `.so` when an agreed native Linux
CUDA build host is available. Record source/submodule revision, compiler,
toolkit/driver and Python/Torch versions and library SHA256. The currently
accepted Windows SHA256 selects its exact local artifact; it cannot identify a
Linux build or prove the mutable checkout reproduces that artifact.

Prefer one small consolidated safety smoke (immutable source/read, bounded
cache registration/DMA, numerical forward and unload) before the user's real
model execution. CPU fixtures can verify POSIX/file/metadata/config paths;
compilation alone is not a CUDA gate. Present exact CUDA commands, hypothesis,
limits and acceptance for approval, stop at the first failure/anomaly, no
automatic retries/full benchmark/model matrix. Do not reinterpret Windows-only
test skips as Linux support or passing safety contracts.

User-driven runs will then identify Linux/model-specific problems for targeted
fixes. Linux stays experimental until that evidence exists; no WSL/macOS,
multi-GPU, arbitrary custom/quantized forward, cuFFT interoperability or poisoned
driver recovery promise. No publication/history cleanup is authorized here.

## File-source/bootstrap implementation checkpoint (2026-10-03)

Added Linux-only `native/src/posix_file_io.h`: regular-file CLOEXEC descriptors,
nonblocking open to reject FIFO/device paths without hanging, positioned chunked
reads with EINTR/partial-read handling, EOF/off_t-range errors. Source C ABI
uses fstat device/inode/size/nanosecond mtime and the existing unique generation
and shared ownership. Retained descriptors conserve their inode even if the
pathname is replaced; reads validate captured metadata before/after. Timestamp
is platform-local, not Windows FILETIME. Caller must keep sources immutable;
POSIX metadata detection does not lock out concurrent external writers or prove
an adversarial payload unchanged after restored timestamps.

Existing Windows handle/open/read/inspect implementation remains in its own
branches; the same common validation/retain/close boundaries are used. No loader
CPU tensors or native cache policy are added to this file primitive. Linux host
storage, registration/admission and DMA/ring/batches are NOT implemented by it.

Bootstrap implicitly discovers `build-linux/libddo_vmm.so`, then
`build/libddo_vmm.so` on Linux; Windows keeps `build/Release/ddo_vmm.dll`.
Other platforms do not discover a Windows DLL automatically. Explicit selection
and fresh-process installation remain unchanged. Matching suffix/discovery does
not establish native support or verify the build's capabilities.

209 Python/ABI CPU regressions PASS in5.76s, including platform discovery with
mocked relaunch, protected presets/profiles, component/source/failure contracts.
Actual ABI null-handle test uses the unchanged existing Windows candidate, not
the new Linux C++; it checks CUDA remains uninitialized. Candidate/default
DLL hashes unchanged. No native compilation, CUDA run, Linux execution or
Windows rebuild/promotion in this checkpoint.

One standalone Linux CPU contract prepared under `native/tests/posix-source/`,
compiling only file-source code without CUDA libraries/toolkit. Covers chunked
and concurrent positioned reads, alias lifetime, pathname replacement/change,
invalid ranges/EOF, regular files versus directory/FIFO and CLOEXEC. It has not
compiled/executed here; Windows has no native Linux compiler/sysroot available.
The single command sequence on the agreed native Linux host will be:

```bash
cmake -S ddo-vmm-allocator/native/tests/posix-source -B ddo-vmm-allocator/build-posix-source -DCMAKE_BUILD_TYPE=Release
cmake --build ddo-vmm-allocator/build-posix-source --parallel
ctest --test-dir ddo-vmm-allocator/build-posix-source --output-on-failure
```

This is CPU only, not a proposed full `.so` build or a CUDA authorization.
Next finish the portable native ring/storage/cache/admission and batch path as
the same Linux block before proposing the bounded CUDA smoke. No additional
full-model performance tuning or broad test matrix is scheduled.
