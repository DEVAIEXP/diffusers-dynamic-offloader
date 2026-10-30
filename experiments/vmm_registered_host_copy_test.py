"""Validate direct registered-host to CUDA-VMM transfers on Windows.

The test registers the existing storage of a CPU PyTorch tensor with
``cuMemHostRegister``.  It then copies that storage directly to a device-backed
CUDA VMM range through ``cuMemcpyHtoDAsync``.  No pinned clone or C/C++ adapter
is used.
"""

from __future__ import annotations

import argparse
import gc
import time


def _ok(result, operation: str):
    from cuda.bindings import driver

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {result[0]}")
    return result[1:]


def run(args: argparse.Namespace) -> None:
    import torch
    from cuda.bindings import driver
    from cuda.core import Buffer, Device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(f"cuda:{args.device}")
    torch.empty(1, device=device)
    Device(args.device).set_current()
    _ok(driver.cuInit(0), "cuInit")

    source = torch.arange(args.megabytes * 1024**2 // 2, dtype=torch.bfloat16)
    source_bytes = source.numel() * source.element_size()
    registered = False
    address = handle = buffer = target = None
    try:
        started = time.perf_counter()
        _ok(
            driver.cuMemHostRegister(source.data_ptr(), source_bytes, driver.CU_MEMHOSTREGISTER_PORTABLE),
            "cuMemHostRegister",
        )
        registered = True
        register_seconds = time.perf_counter() - started

        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = args.device
        prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        (granularity,) = _ok(
            driver.cuMemGetAllocationGranularity(
                prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ),
            "cuMemGetAllocationGranularity",
        )
        size = ((source_bytes + int(granularity) - 1) // int(granularity)) * int(granularity)
        (address,) = _ok(driver.cuMemAddressReserve(size, int(granularity), 0, 0), "cuMemAddressReserve")
        (handle,) = _ok(driver.cuMemCreate(size, prop, 0), "cuMemCreate")
        _ok(driver.cuMemMap(address, size, 0, handle, 0), "cuMemMap")
        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = args.device
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _ok(driver.cuMemSetAccess(address, size, [access], 1), "cuMemSetAccess")

        stream = torch.cuda.Stream(device=device)
        started = time.perf_counter()
        _ok(
            driver.cuMemcpyHtoDAsync(
                address,
                source.data_ptr(),
                source_bytes,
                driver.CUstream(stream.cuda_stream),
            ),
            "cuMemcpyHtoDAsync",
        )
        stream.synchronize()
        copy_seconds = time.perf_counter() - started
        buffer = Buffer.from_handle(address, size, owner=handle)
        target = torch.from_dlpack(buffer)[:source_bytes].view(torch.bfloat16)
        torch.testing.assert_close(target[:4096].cpu(), source[:4096])
        gib_per_second = source_bytes / max(copy_seconds, 1e-9) / 1024**3
        print("Registered-host to VMM copy: PASS")
        print(f"  device: {torch.cuda.get_device_name(args.device)}")
        print(f"  bytes: {source_bytes / 1024**2:.2f} MiB")
        print(f"  host registration: {register_seconds:.4f}s")
        print(f"  direct H2D: {copy_seconds:.4f}s ({gib_per_second:.2f} GiB/s)")
    finally:
        target = buffer = None
        gc.collect()
        if address is not None and handle is not None:
            _ok(driver.cuMemUnmap(address, size), "cuMemUnmap")
            _ok(driver.cuMemRelease(handle), "cuMemRelease")
        if address is not None:
            _ok(driver.cuMemAddressFree(address, size), "cuMemAddressFree")
        if registered:
            _ok(driver.cuMemHostUnregister(source.data_ptr()), "cuMemHostUnregister")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--megabytes", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
