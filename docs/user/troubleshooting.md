# Errors and troubleshooting

This page explains the errors DDO itself reports and what to check first. Errors
raised by the model or pipeline are reported by those libraries; DDO does not
rewrite them.

## GPU memory errors in VMM Fast

VMM Fast first tries to satisfy a GPU allocation by reusing and releasing unused
backing. When that recovery cannot free enough memory, DDO raises a
`torch.OutOfMemoryError` with one of these messages:

| Message starts with | What could not be allocated |
| --- | --- |
| `DDO VMM allocator could not obtain <N> MiB of GPU memory on device <D> after recovery (termination=...)` | A PyTorch tensor (activations or intermediate results) |
| `DDO VMM allocator could not obtain <N> MiB of GPU memory for model weights (execution slot)` or `(resident weight range)` | The GPU memory that receives model weights before a layer runs |

`termination` explains why recovery stopped: `no_release` (nothing unused could
be released), `partial_release` (released memory was not enough),
`retry_refused` (the retry after recovery was still refused) or
`quota_consumed`. The weight-memory message keeps the native detail, for
example `cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`.

No tensor or weight storage is published when these errors are raised, so the
CUDA context stays usable: the host can release its components normally and
report the failure.

What to check:

1. **Other applications using GPU memory.** Browsers, games, video players and
   other GPU programs reduce the memory available to the process. A workload
   that runs close to the device limit can fail only while they are open.
2. **Activation size of the workload.** Resolution, sequence length, frame count
   and batch size drive the activation peak. Model-specific options such as
   token chunking reduce it; see the model's own documentation and the
   [DDO examples](https://devaiexp.github.io/ddo-examples/) recipes.
3. **Residency budget.** An explicit transformer budget larger than the profile
   recommendation leaves less room for activations. Prefer a
   [capacity profile](capacity-profiles.md) built on the same machine.

DDO does not retry automatically or grow budgets after these errors.

## `cudaErrorIllegalAddress` after an allocation failure

If a final allocation failure is followed by `CUDA error: an illegal memory
access was encountered` (and component cleanup also fails), the process is not
using DDO's checked allocation boundary. PyTorch's plain pluggable allocator
accepts a null allocation result as tensor storage instead of raising.

The checked boundary is included for the PyTorch build it was compiled against
(currently Windows, PyTorch 2.12.1+cu132, CPython 3.12). On other PyTorch builds
and platforms DDO keeps the plain registration. Check which one a process uses:

```python
from ddo_vmm_allocator.torch_allocator import installed_allocation_boundary

print(installed_allocation_boundary())  # "checked", "unguarded" or None before bootstrap
```

With `unguarded`, the CUDA context cannot be recovered after such an error:
restart the process, and reduce memory pressure as described above.

## Allocation recovery messages

Successful recoveries are quiet by default. Set `DDO_SHOW_RECOVERY_WARNS=1`
before the bootstrap to print them. Each summary line is:

- `DDO allocation recovery: status=recovered ...`: memory was released and the
  allocation succeeded; execution continues.
- `DDO allocation recovery: status=failed ...`: the allocation was not satisfied;
  the error described above follows.

Failed transactions and final errors are printed in either mode. Lines such as
`DDO segmented Driver refusal: operation=cuMemCreate status=2` are low-level
driver reports of the first refusal; they appear before recovery and are not by
themselves an error.

## Bootstrap errors

`CUDA is already initialized; start a fresh process through the DDO VMM
launcher` means Torch, Diffusers or a model library used CUDA before the VMM
bootstrap ran. Call `ensure_vmm_fast_process()` as the first executable action
of the runner, before those imports (see [integration](integration.md)).
