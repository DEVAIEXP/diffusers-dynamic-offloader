"""Validate CUDA managed memory as a PyTorch CUDA tensor on Windows.

CUDA documents that managed allocations on Windows are physically backed by
host/zero-copy memory unless device allocation is explicitly forced.  This
probe allocates such a range through ``cuda-python``, exposes its CUDA-valid
pointer through ``__cuda_array_interface__``, writes the weight bytes from the
CPU, and uses the result in ``torch.nn.functional.linear``.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import time


def _ok(result, operation: str):
    from cuda.bindings import driver

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {result[0]}")
    return result[1:]


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as functional
    from cuda.bindings import driver
    from cuda.core import Device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.dimension <= 0:
        raise ValueError("--dimension must be positive")
    device = torch.device(f"cuda:{args.device}")
    torch.empty(1, device=device)
    Device(args.device).set_current()
    _ok(driver.cuInit(0), "cuInit")
    weight_bytes = args.dimension * args.dimension * torch.empty((), dtype=torch.bfloat16).element_size()
    (pointer,) = _ok(
        driver.cuMemAllocManaged(weight_bytes, driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL),
        "cuMemAllocManaged",
    )
    class _CudaByteArray:
        """Non-owning CUDA array-interface view of a CUDA-valid pointer."""

        def __init__(self, address: int, size_bytes: int) -> None:
            self.__cuda_array_interface__ = {
                "shape": (size_bytes,),
                "strides": None,
                "typestr": "|u1",
                "data": (address, False),
                "version": 3,
            }

    array_view = managed_weight = source = inputs = output = None
    local_weight = None
    try:
        # PyTorch does not currently accept DLPack kDLCUDAManaged (13), but
        # this pointer is CUDA-addressable.  The CUDA array interface carries
        # no false device-type claim and preserves the allocation's ownership
        # in this function until CUDA work has completed.
        array_view = _CudaByteArray(int(pointer), weight_bytes)
        managed_weight = torch.as_tensor(array_view, device=device).view(torch.bfloat16).view(
            args.dimension, args.dimension
        )
        if managed_weight.data_ptr() != int(pointer):
            raise RuntimeError("PyTorch copied the managed allocation instead of wrapping its CUDA address")
        if managed_weight.device != device:
            raise RuntimeError(f"Managed allocation was exposed as {managed_weight.device}, expected {device}")

        source = torch.randn((args.dimension, args.dimension), dtype=torch.bfloat16)
        ctypes.memmove(int(pointer), source.data_ptr(), weight_bytes)
        inputs = torch.randn((1, args.dimension), dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = functional.linear(inputs, managed_weight)
        torch.cuda.synchronize(device)
        managed_linear_seconds = time.perf_counter() - started
        expected = functional.linear(inputs, source.to(device))
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

        # The intended runtime path never executes GEMM over host-backed
        # managed memory.  Measure its direct transfer into ordinary local
        # CUDA storage, which is what a VMM fault/materialization would use.
        local_weight = torch.empty_like(managed_weight, device=device)
        stream = torch.cuda.Stream(device=device)
        started = time.perf_counter()
        _ok(
            driver.cuMemcpyDtoDAsync(
                driver.CUdeviceptr(local_weight.data_ptr()),
                pointer,
                weight_bytes,
                driver.CUstream(stream.cuda_stream),
            ),
            "cuMemcpyDtoDAsync(managed to local)",
        )
        stream.synchronize()
        materialize_seconds = time.perf_counter() - started
        torch.testing.assert_close(local_weight.cpu(), source)
        started = time.perf_counter()
        local_output = functional.linear(inputs, local_weight)
        torch.cuda.synchronize(device)
        local_linear_seconds = time.perf_counter() - started
        torch.testing.assert_close(local_output, expected, rtol=2e-2, atol=2e-2)
        print("Managed CUDA tensor probe: PASS")
        print(f"  device: {torch.cuda.get_device_name(args.device)}")
        print(f"  weight bytes: {weight_bytes / 1024**2:.2f} MiB")
        print(f"  tensor device: {managed_weight.device}")
        print(f"  managed linear: {managed_linear_seconds:.4f}s")
        print(
            f"  managed-to-local: {materialize_seconds:.4f}s "
            f"({weight_bytes / max(materialize_seconds, 1e-9) / 1024**3:.2f} GiB/s)"
        )
        print(f"  local linear: {local_linear_seconds:.4f}s")
    finally:
        local_weight = output = inputs = source = managed_weight = array_view = None
        gc.collect()
        _ok(driver.cuMemFree(pointer), "cuMemFree(managed)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dimension", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
