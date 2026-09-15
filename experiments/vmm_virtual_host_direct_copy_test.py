"""Windows feasibility probe for a contiguous registered virtual host buffer.

The buffer is reserved and committed with ``VirtualAlloc``, exposed as a CPU
``torch`` byte tensor, registered once with CUDA, and copied directly into a
device-local CUDA VMM mapping through ``cuMemcpyHtoDAsync``.  It intentionally
does not use a PyTorch pinned clone or one CUDA registration per model tensor.

This is a clean-room probe for the host side of a future VMM transport backend.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import time


MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04


def _ok(result, operation: str):
    from cuda.bindings import driver

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {result[0]}")
    return result[1:]


def _virtual_buffer(size_bytes: int):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong)
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong)
    kernel32.VirtualFree.restype = ctypes.c_int
    address = kernel32.VirtualAlloc(None, size_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
    if not address:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel32, int(address)


def run(args: argparse.Namespace) -> None:
    import torch
    from cuda.bindings import driver
    from cuda.core import Buffer, Device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.megabytes <= 0:
        raise ValueError("--megabytes must be positive")

    size_bytes = args.megabytes * 1024**2
    device = torch.device(f"cuda:{args.device}")
    torch.empty(1, device=device)
    Device(args.device).set_current()
    _ok(driver.cuInit(0), "cuInit")

    kernel32, host_address = _virtual_buffer(size_bytes)
    raw_host = (ctypes.c_ubyte * size_bytes).from_address(host_address)
    host = torch.frombuffer(raw_host, dtype=torch.uint8)
    source = torch.arange(size_bytes, dtype=torch.uint8)
    host.copy_(source)
    registered = False
    address = handle = buffer = target = None
    try:
        started = time.perf_counter()
        _ok(
            driver.cuMemHostRegister(host_address, size_bytes, driver.CU_MEMHOSTREGISTER_PORTABLE),
            "cuMemHostRegister(virtual host buffer)",
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
        mapped_bytes = ((size_bytes + int(granularity) - 1) // int(granularity)) * int(granularity)
        (address,) = _ok(driver.cuMemAddressReserve(mapped_bytes, int(granularity), 0, 0), "cuMemAddressReserve")
        (handle,) = _ok(driver.cuMemCreate(mapped_bytes, prop, 0), "cuMemCreate")
        _ok(driver.cuMemMap(address, mapped_bytes, 0, handle, 0), "cuMemMap")
        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = args.device
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _ok(driver.cuMemSetAccess(address, mapped_bytes, [access], 1), "cuMemSetAccess")

        stream = torch.cuda.Stream(device=device)
        started = time.perf_counter()
        _ok(
            driver.cuMemcpyHtoDAsync(address, host_address, size_bytes, driver.CUstream(stream.cuda_stream)),
            "cuMemcpyHtoDAsync(virtual host buffer)",
        )
        stream.synchronize()
        copy_seconds = time.perf_counter() - started
        buffer = Buffer.from_handle(address, mapped_bytes, owner=handle)
        target = torch.from_dlpack(buffer)[:size_bytes]
        # cuda.core exposes an untyped byte buffer as int8 through DLPack;
        # compare byte representations rather than signed integer values.
        torch.testing.assert_close(target[:4096].view(torch.uint8).cpu(), source[:4096])
        gib_per_second = size_bytes / max(copy_seconds, 1e-9) / 1024**3
        print("Virtual-host registered to VMM copy: PASS")
        print(f"  device: {torch.cuda.get_device_name(args.device)}")
        print(f"  bytes: {size_bytes / 1024**2:.2f} MiB")
        print(f"  host registration: {register_seconds:.4f}s")
        print(f"  direct H2D: {copy_seconds:.4f}s ({gib_per_second:.2f} GiB/s)")
    finally:
        target = buffer = None
        gc.collect()
        if address is not None and handle is not None:
            _ok(driver.cuMemUnmap(address, mapped_bytes), "cuMemUnmap")
            _ok(driver.cuMemRelease(handle), "cuMemRelease")
        if address is not None:
            _ok(driver.cuMemAddressFree(address, mapped_bytes), "cuMemAddressFree")
        if registered:
            _ok(driver.cuMemHostUnregister(host_address), "cuMemHostUnregister")
        host = raw_host = None
        gc.collect()
        if not kernel32.VirtualFree(ctypes.c_void_p(host_address), 0, MEM_RELEASE):
            raise ctypes.WinError(ctypes.get_last_error())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--megabytes", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
