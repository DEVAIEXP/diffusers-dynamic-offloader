"""Probe CUDA VMM physical backing allocated from host memory.

This isolates the Windows/WDDM path where a stable CUDA virtual address is
backed by CUDA's host allocation type instead of device-local VRAM.  It is a
feasibility test only; it does not change DDO runtime behavior.
"""

from __future__ import annotations

import argparse
import gc
import time


def ok(result, operation):
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
    torch.empty(1, device=f"cuda:{args.device}")
    Device(args.device).set_current()
    ok(driver.cuInit(0), "cuInit")
    (device,) = ok(driver.cuDeviceGet(args.device), "cuDeviceGet")
    (supported,) = ok(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            device,
        ),
        "cuDeviceGetAttribute(VMM)",
    )
    if not supported:
        raise RuntimeError("VMM is unsupported")

    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
    prop.location.id = 0
    prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    (granularity,) = ok(
        driver.cuMemGetAllocationGranularity(
            prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        ),
        "cuMemGetAllocationGranularity(host)",
    )
    size = ((args.megabytes * 1024**2 + int(granularity) - 1) // int(granularity)) * int(granularity)
    (address,) = ok(driver.cuMemAddressReserve(size, int(granularity), 0, 0), "cuMemAddressReserve")
    (handle,) = ok(driver.cuMemCreate(size, prop, 0), "cuMemCreate(host)")
    buffer = tensor = None
    try:
        ok(driver.cuMemMap(address, size, 0, handle, 0), "cuMemMap(host)")
        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = args.device
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        ok(driver.cuMemSetAccess(address, size, [access], 1), "cuMemSetAccess(host)")
        buffer = Buffer.from_handle(address, size, owner=handle)
        tensor = torch.from_dlpack(buffer)
        source = torch.full((args.megabytes * 1024**2 // 2,), 0.25, dtype=torch.bfloat16)
        started = time.perf_counter()
        tensor[: source.numel()].view(torch.bfloat16).copy_(source)
        torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - started
        print("VMM host-backing probe: PASS")
        print(f"  device: {torch.cuda.get_device_name(args.device)}")
        print(f"  granularity: {int(granularity) / 1024**2:.2f} MiB")
        print(f"  mapped host bytes: {size / 1024**2:.2f} MiB")
        print(f"  host-backed upload: {elapsed:.4f}s")
    finally:
        tensor = None
        buffer = None
        gc.collect()
        ok(driver.cuMemUnmap(address, size), "cuMemUnmap(host)")
        ok(driver.cuMemRelease(handle), "cuMemRelease(host)")
        ok(driver.cuMemAddressFree(address, size), "cuMemAddressFree(host)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--megabytes", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
