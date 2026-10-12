# Linux native idle-trim candidate (2026-10-11)

Status: isolated compilation, static export inspection and the approved small
CUDA idle-trim ownership gate PASS. The selected staged encoder/transformer image
and five-second video resident/leaf VAE pairs completed with identical delivered
media within each Linux pair. WSL remains unvalidated.
The prior accepted Linux binary is preserved.

## Build identity

- Candidate: `ddo-vmm-allocator/build-linux-idle-trim-20261011/libddo_vmm.so`.
- SHA256: `db63138da2fdcca328a244ff885ca44ecd8b33683825b2ab0f6f39a95b3cba79`.
- Preserved `build-linux/libddo_vmm.so` SHA256, unchanged before/after:
  `a87979f8773a010ddc2c781a37f3ef55112ca558a9d93083b972f51246a2b245`.
- Ubuntu 26.04.1, GCC 15.2.0, CMake 4.2.3, toolkit discovery 13.2.78.
- Source includes uncommitted idle-trim C++ and binding changes. This is not an
  immutable source checkpoint, binary promotion or publication.

The explicit span-pool build retains the previously accepted Linux feature
selection. Joined spans, probes, source dispatcher, source test hooks, allocation
diagnostics and independent execution kernels are OFF. No Windows binary,
protected runtime route, native source/cache ceiling or kernel policy changed.

Approved compilation command, run from the DDO root:

```bash
cmake -S ddo-vmm-allocator -B ddo-vmm-allocator/build-linux-idle-trim-20261011 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/run/media/eliseu/AI2/ProjetosIA/diffusers-dynamic-offloader/.venv-linux/lib/python3.12/site-packages/nvidia/cu13 \
  -DDDO_VMM_BUILD_PROBES=OFF -DDDO_VMM_NATIVE_SPAN_POOL=ON \
  -DDDO_VMM_NATIVE_JOINED_SPANS=OFF -DDDO_VMM_NATIVE_SOURCE_DISPATCHER=OFF \
  -DDDO_VMM_SOURCE_TEST_HOOKS=OFF -DDDO_VMM_ALLOCATION_DIAGNOSTICS=OFF \
  -DDDO_VMM_BUILD_EXECUTION_OPS=OFF
cmake --build ddo-vmm-allocator/build-linux-idle-trim-20261011 --parallel 2
```

Build completed at 100%, with warnings for unused alternative allocator helper
functions/variable; no compilation errors. Static ELF inspection confirms
`ddo_vmm_torch_trim_idle_v1`, pool statistics and Torch malloc/free exports.
NEEDED entries include libcuda.so.1, the C++/GCC runtimes and libc. This is not a
dynamic-load, linkage-at-runtime or CUDA safety gate. No probe was compiled/run.

The primary host Python environment separately passed its eight-file CPU
sequence (52 tests in 6.31 s); those mock/routing contracts do not validate
native event retirement. No native C++ test executable ran in this build step.

## Executed bounded CUDA gate: PASS

Run from the primary examples host in a fresh process, using its `.venv-linux`:

```bash
cd /run/media/eliseu/AI2/ProjetosIA/ddo-examples
.venv-linux/bin/python -B \
  ../diffusers-dynamic-offloader/ddo-vmm-allocator/scripts/vmm_idle_trim_probe.py \
  --library ../diffusers-dynamic-offloader/ddo-vmm-allocator/build-linux-idle-trim-20261011/libddo_vmm.so
```

Hypothesis: the Linux pool can retire pending whole-idle backing only after its
event completes, without touching a live neighbor. The existing probe uses a
32 MiB temporary tensor, a 1024-byte live payload and a 512 MiB pool cache setting;
it loads no model and allocates no persistent source cache. Mapping granularity
and driver baseline mean those payload sizes are not an exact GPU peak bound.

Require preserved exact live bytes, reduced idle mapping, no quarantine/failure,
safe repeated trim, successful allocation after trim, and zero final mapped/live/
pending/quarantined bytes and failures. Stop at the first failure or memory
anomaly; do not run a model, reset the driver, force-free or tune budgets to pass.
The user confirmed this exact gate and it completed with exit 0 and status PASS
on 2026-10-11, using the candidate hash recorded above and the primary host
`.venv-linux`. No additional CUDA contract or model run was added.

| Native pool observation | Before trim | After trim | Final cleanup |
| --- | ---: | ---: | ---: |
| Mapped bytes | 35,651,584 | 2,097,152 | 0 |
| Live requested bytes | 1,024 | 1,024 | 0 |
| Live allocations | 1 | 1 | 0 |
| Pending bytes | 33,554,432 | 0 | 0 |
| Whole-idle backing bytes | 33,554,432 | 0 | 0 |
| Quarantined bytes | 0 | 0 | 0 |
| Failures | 0 | 0 | 0 |

The active byte payload remained exact. Repeated trim, allocation-after-trim
and final cleanup succeeded. Native eviction wait was 0.000015980 s after the
first trim and 0.000021792 s in the final observation. These are native wait
counters, not whole-process or model performance measurements. No force release,
driver reset or budget adjustment occurred. This is bounded pool ownership
acceptance, not driver-loss recovery, source-transport or arbitrary-model support.
The next image/video resident/leaf pairs require their own reviewed commands
and confirmation. The old accepted binary remains unmodified/unpromoted.

The separately approved primary-host Linux image control completed with this
binary, native encoder/transformer and resident VAE, without reported native or
monitor failures. Pipeline metrics, workload, PNG hash and comparison limits
belong to the host's dated entry in `ddo-examples/experiments/dynamic_offload_results.md`.
This is one bounded workload, not general model/platform support. No leaf or
video generation was included in that authorization.

The separately approved leaf image counterpart subsequently completed with
the same candidate and fixed workload. Both delivered Linux PNG hashes match
exactly, with no reported native or monitor errors. Pipeline details and scope
limits remain in the host benchmark evidence; neither image run automatically
invokes native idle trim. Video and WSL are not accepted by this image pair.

The subsequent separately approved five-second video resident/leaf pair also
completed with identical MP4 bytes, no reported native/monitor errors and correct
encoded video/audio structure. Host benchmark evidence and its selected matrix
record full conditions; this adds no arbitrary-model, FP8/SDNQ, WSL or driver-loss
support claim. The prior accepted binary remains unchanged. No additional gate
or benchmark is authorized by this closure.
