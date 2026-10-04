# Linux native transport checkpoint (2026-10-04)

Implementation and compilation are complete for the initial Linux source,
ring, persistent cache and immutable batch route. **The approved small CUDA
sequence passed on the original .so**; the later overcommit-mode correction
subsequently passed its three CUDA cache contracts and one selected staged
real-model generation. See [executed model gate](native-linux-staged-gate.md).
This is bounded workload acceptance, not general model support or Windows promotion.

## Native implementation

The existing bounded workers now share positioned source I/O on Windows/Linux.
The adapter's former Windows-only admission guard now allows native Linux;
CPU platform contracts verify identical geometry and refusal on other systems.
Linux uses retained POSIX descriptors and anonymous private mmap storage (no
MAP_NORESERVE, eager full-model copy, CPU loader tensors or Python cache).
The existing native cache/batch state machine, tickets and DMA/compute events
are reused; slot size/worker geometry and configured ceilings did not change.
CUDA registration is separate from committed storage: unregister preserves bytes.
No Linux WDDM emulation, driver reset, model-name policy or budget expansion.

Native Linux admission observes MemAvailable and the actual overcommit mode.
Mode2 also bounds growth by max(CommitLimit - Committed_AS, 0). Modes0/1 use
MemAvailable as a finite admission bound rather than treating CommitLimit as an
enforced kernel ceiling. Unknown/unreadable modes refuse growth. This is not
an OS guarantee of physical commitment. Cgroup v2
memory.max/current/high constraints include visible ancestors and mounted subtree
roots. RLIMIT_AS/DATA bound growth too. Missing/malformed observations refuse
growth; v1 memory-controller environments are deliberately refused in this first
port. Pressure uses the existing low/high hysteresis and eligible-entry trim.
Limits are snapshots, not reservations against competing processes/OOM killers.
Linux can still refuse a Windows full-source capacity under strict commit or
container/process pressure: do not enlarge kernel settings/headroom to bypass it.

The Windows primitives still call VirtualAlloc/VirtualFree/GlobalMemoryStatusEx.
CUDA/event policy and protected Python linear/NVMe paths were not changed.
This host cannot establish rebuilt Windows regression safety; accepted binaries
were preserved and their candidate/default SHA256 values remain unchanged.

## Actual build and CPU evidence

Ubuntu26.04.1 x86_64; GCC15.2.0, CMake4.2.3, Ninja1.13.2; Python3.12.14,
Torch2.12.1+cu132, CUDA toolkit13.2.78. Optional vmm-dev-linux supplies nvcc/CCCL;
the system supplies libcuda.so (driver595.91.07). Shared NTFS checkout supported
build/execution; fixtures use /tmp. No Windows environment was replaced.

Initial CPU configure exposed a CMake test-property directory-scope error, fixed
at the child add_test. Toolkit configure required recognizing the real versioned
libcudart.so.13 and installing CCCL through the optional profile. The first
library compile also exposed malformed -D arguments that set unused DO_* rather
than DDO_* variables. The recipe and its tests now use -DDDO_*; actual CMakeCache
confirms probes OFF, span pool ON, dispatcher OFF and lifecycle hooks OFF.
Those earlier trial builds are not the selected checkpoint.

```bash
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode cpu --jobs 2
export CUDAToolkit_ROOT="$PWD/.venv-linux/lib/python3.12/site-packages/nvidia/cu13"
.venv-linux/bin/python ddo-vmm-allocator/scripts/build_linux.py --mode library --jobs 2
.venv-linux/bin/python -m pytest tests/test_linux_environment_profile.py tests/test_linux_build_recipe.py tests/test_vmm_bootstrap.py tests/test_backend.py tests/test_dynamic_offload_presets.py tests/test_named_profiles.py ddo-vmm-allocator/tests/test_linux_native_source.py -q -x -p no:cacheprovider
DDO_VMM_LIBRARY_PATH="$PWD/ddo-vmm-allocator/build-linux/libddo_vmm.so" .venv-linux/bin/python -m pytest tests/test_linux_environment_profile.py tests/test_linux_build_recipe.py tests/test_vmm_bootstrap.py tests/test_backend.py tests/test_dynamic_offload_presets.py tests/test_named_profiles.py ddo-vmm-allocator/tests/test_linux_native_source.py ddo-vmm-allocator/tests/test_transport_platform_cpu.py ddo-vmm-allocator/tests/test_transport_windows.py ddo-vmm-allocator/tests/test_source_lifecycle_cpu.py tests/test_vmm_component_adapter_cpu.py -q -x -p no:cacheprovider
```

- Nine real C++ CPU contracts PASS (CTest0.03s): seven shared helpers, POSIX
  source and Linux storage/headroom (synthetic ancestor/namespace limits,
  malformed/overflow refusal, real host observation, mmap byte lifetime).
- 73 Python CPU tests PASS in3.09s, no skips. Includes actual Linux .so source
  ABI/read/retained-descriptor/concurrency/invalidation, no CUDA calls in these
  ABI cases, and protected routing/profile/bootstrap/dependency contracts.
- Final expanded CPU sequence:122 PASS in3.22s, no skips. Adds platform/transport,
  component ownership and lifecycle failure-retention/actual null-handle ABI.
  First expanded run stopped with100 passes because the lifecycle ABI test
  requires explicit DDO_VMM_LIBRARY_PATH; the same sequence passed with it set
  to this .so. No CUDA initialization occurred (the ABI test asserts that).
- readelf/nm confirm libcuda.so.1 linkage (not cudart) and exported source,
  source-cache ABI11, batch/ring, native allocator and span-pool symbols.
- Original CUDA-validated library (now preserved as
  build-linux/libddo_vmm.pre-overcommit-mode-20261004.so):
  SHA256 e71288717924c66acb8205284c02eb4089ab9aa09e921515768fb690f3aee503.

DDO parent ddcac71 / allocator base84f17ba with **uncommitted implementation
changes**, not a clean commit reproduction. No source-history rewrite/push.
GCC/glibc build is local, not a universal binary for older distributions.
Existing unused-function warnings in the span-pool build are not runtime proof.

## Overcommit-mode correction — CPU/build PASS, CUDA pending

Real-model preparation observed mode0, about54.5GiB MemAvailable but only20.9GiB
CommitLimit minus Committed_AS. Applying that latter number in all modes would
artificially restrict the integral24.625GiB source +1.5GiB staging +1GiB default
host margin (27.125GiB minimum, excluding other concurrent demands).
The user approved correcting this interpretation before GPU/model execution.
[Kernel accounting documentation](https://docs.kernel.org/mm/overcommit-accounting.html)
identifies mode0 as heuristic, mode1 as always-overcommit and mode2 as strict.

Only the Linux native headroom primitive changed: read/validate the mode on each
observation; enforce remaining commit in mode2; use finite physical admission
in modes0/1, then apply the existing cgroup/RLIMIT bounds. No sysctl, swap, driver,
configured DDO ceiling, reserve, CUDA/event path or Windows primitive changed.
Mode0 mmap can still be denied by the kernel; existing allocation/registration
failure paths remain necessary. This is not unlimited admission or an OOM guarantee.

Expanded native CPU fixtures cover all three modes, strict saturation,
oversubscribed non-strict commit, finite growth/headroom and invalid/unknown
modes; existing cgroup/process/source/storage checks retained. Nine C++ CPU
contracts PASS (CTest0.03s);122 Python CPU PASS in3.25s, no skips. Library compiled
with unchanged feature selection; new SHA256
a87979f8773a010ddc2c781a37f3ef55112ca558a9d93083b972f51246a2b245.
The original e712887... library was saved before rebuilding. Its CUDA result
below does NOT validate this new binary. Both Windows DLL hashes remain unchanged.
Subsequently approved:3 CUDA cache contracts PASS in3.10s, then one staged native
encoder/transformer generation PASS, steady state1.050970s/step. See the executed
[Linux staged model gate](native-linux-staged-gate.md).

## Approved CUDA sequence — PASS (2026-10-04)

The user confirmed this exact sequence. The initial sandboxed probe stopped at
PyTorch's CUDA-availability check, before allocator installation/device allocation.
GPU access was restricted by the sandbox; after permission escalation, the same
probe passed outside it. The seven named CUDA contracts then ran outside the
sandbox and passed: **7 passed in3.64s**, no skips, both commands exit0. No source,
library, parameter or ceiling was changed during the sequence.

Probe observations: Torch2.12.1+cu132;16MiB allocation; after free, zero live
allocations and one16MiB cached allocation; stable address reused. Asynchronous
stream lifetime and nonblocking reuse/renewed consumer fence both verified.
The contracts passed their existing numerical, retained-source/no-reread,
registration ceiling, immutable-batch ownership, live-ticket close/clear refusal
and public buffer/skip-cancellation assertions. They do not include an independent
process-wide peak monitor, poisoned-driver recovery or full-model benchmark.
No model execution, Windows rerun, automatic retry/tuning or capacity expansion
followed the successful gate. Next propose a real-model Linux manifest/command
with explicit host/commit limits and obtain separate approval before execution.

Run from DDO, using the isolated environment and this exact .so. Probe is a fresh
process installing the allocator before CUDA initialization; its temporary cache
is128MiB and allocation16MiB. Only if it passes, run seven small contracts:
persistent registration recycling, immutable-batch ownership/reuse, live-ticket
clear/close protection and four existing public numerical/buffer/cancellation
variants. Public contract ceilings:64MiB source,2MiB staging,6MiB native weight
backing; not a full-model or production budget change.

```bash
export DDO_VMM_LIBRARY_PATH="$PWD/ddo-vmm-allocator/build-linux/libddo_vmm.so"
.venv-linux/bin/python ddo-vmm-allocator/scripts/vmm_torch_allocator_probe.py --library "$DDO_VMM_LIBRARY_PATH" --megabytes 16 --cache-mib 128 --verify-stream-lifetime --verify-nonblocking-reuse
# Check exit/status before starting the following command; stop if it failed.
.venv-linux/bin/python -m pytest ddo-vmm-allocator/tests/test_source_cache.py::test_persistent_storage_survives_registration_recycling_without_reread ddo-vmm-allocator/tests/test_source_cache.py::test_immutable_batch_retains_sources_recycles_events_and_guards_close ddo-vmm-allocator/tests/test_source_cache.py::test_live_tickets_protect_cache_close_and_clear tests/test_vmm_execution_buffer.py -q -x -p no:cacheprovider
```

Hypothesis: the accepted native pipeline preserves Linux storage/registration,
cross-stream order, numerical output and ownership at small fixed capacities.
Acceptance: probe PASS, all seven tests PASS with no skips, existing numerical
tolerances met, no source reread after warm-up, no unexpected temporary fallback,
ceilings honored, canceled/closed owners correctly retired. Stop at first error,
skip or memory anomaly; no retries, capacity escalation or model benchmark.
Source-cache fixture now recognizes Linux and fails rather than hiding absent
CUDA behind a Windows-only skip. Exact numerical assertions remain unchanged.
This passed small gate is not device-loss recovery or full-model performance proof.
