# Validation

## Validation layers

| Layer | What it establishes |
| --- | --- |
| CPU contracts | Routing, configuration, reports, discovery and mocked ownership |
| Native CPU contracts | File/storage behavior and Linux admission observations |
| Small CUDA contracts | Payload correctness, registration reuse, event ownership, close refusal, interruption and idle trim |
| Model validation | Correct output and public reports for a fixed model/workload |
| Installed wheel | Discovery, linkage and separately tested native behavior of the actual packaged bytes |

Compilation and source-checkout tests do not establish installed-wheel acceptance.
A successful model does not certify all routes, quantizers or platforms.

## Installed-package check without CUDA

From the checkout, run `scripts/probe_installed_native_cpu.py` with the installed
environment's Python and `-I -B`. It checks the native library hash and automatic
discovery, then simulates the child-process bootstrap without loading the DLL/.so
or running CUDA. This is a maintainer release check, separate from GPU tests and
complete model runs.

## Selected installed-wheel acceptance

The `0.1.0rc2` allocator/DDO candidate pair has selected installed allocator
smoke-test acceptance on Windows, native Linux and WSL. The unchanged Linux
wheel was used on both Linux targets.

| Target | Native artifact SHA256 | Scope |
| --- | --- | --- |
| Windows AMD64, CUDA 13.2 | `4354a4dce319559c3cbb3533554798d688e9b779f2ca08991f30e28309e69c0b` | Installed discovery and idle-pool trim |
| Linux x86_64 / WSL2, CUDA 13.2 | `ea8f8cc61b2303c4921f1527eae53250833613c83db48b5d2111f1d79309ab59` | Installed discovery, dependency resolution and idle-pool trim |

The reproducible contract is
`ddo-vmm-allocator/scripts/vmm_idle_trim_probe.py --library <installed-library>`.
It preserves a live payload while reclaiming idle backing, permits allocation
after trim and checks final ownership/failure counters. It is not a model or
source-cache performance test.

[Hardware and environments](hardware.md) records the selected test host.
[Package delivery](releases.md) describes candidate generation and identity.
These results do not establish dependency-complete model integration, arbitrary
driver support or manylinux compliance.

## Source-level coverage

Public tests cover policy/precedence, loading/source association, reports,
dense convolution ownership, source-cache/event lifetimes and bounded cleanup.
Saved TorchAO FP8 has separate payload/root-state/interruption contracts;
BF16 success does not imply FP8 success. Refer to
[FP8 support](../user/fp8-weight-only.md) for the supported recipe.

Native Linux admission must preserve overcommit mode, physical/process bounds
and finite cgroup limits. Unknown observations refuse growth. WSL additionally
requires bounded host-registration validation; an observed capacity is not a
universal preset.

Full model comparisons and their exact reproduction conditions are maintained
in [DDO examples](https://github.com/DEVAIEXP/ddo-examples), not duplicated here.

## Acceptance rules

Use [manual validation](manual-validation.md) to record routes, versions, native
identity, output and measurement scope. Missing fields are unavailable, not zero.
Host callbacks, CUDA events and whole-pipeline wall time remain distinct.

Successful inference does not prove recovery from a poisoned CUDA context,
all-platform support, long-run stability or final release of driver memory.
Retain failed-run diagnostics separately from accepted performance results.
