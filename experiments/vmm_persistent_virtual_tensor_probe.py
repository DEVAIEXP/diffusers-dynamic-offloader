"""Check whether a mapped PyTorch VMM tensor survives a safe remap cycle.

PyTorch validates CUDA pointers while importing DLPack, so the first tensor
view is intentionally created with a small physical VMM allocation mapped.
The address is then unmapped and mapped again without touching the tensor in
between.  This tests stable virtual addresses without presenting PyTorch an
invalid pointer.
"""

from __future__ import annotations

import argparse
import gc


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
    if args.megabytes <= 0:
        raise ValueError("--megabytes must be positive")

    payload_bytes = args.megabytes * 1024**2
    device = torch.device(f"cuda:{args.device}")
    torch.empty(1, device=device)
    Device(args.device).set_current()
    _ok(driver.cuInit(0), "cuInit")

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
    mapped_bytes = ((payload_bytes + int(granularity) - 1) // int(granularity)) * int(granularity)
    address = handle = buffer = persistent = None
    mapped = False
    try:
        (address,) = _ok(driver.cuMemAddressReserve(mapped_bytes, int(granularity), 0, 0), "cuMemAddressReserve")
        # A physical page is required during DLPack import because PyTorch
        # validates the CUDA pointer.  The second mapping is what establishes
        # whether the tensor keeps the same virtual address afterwards.
        (handle,) = _ok(driver.cuMemCreate(mapped_bytes, prop, 0), "cuMemCreate(initial)")
        _ok(driver.cuMemMap(address, mapped_bytes, 0, handle, 0), "cuMemMap(initial)")
        mapped = True
        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = args.device
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _ok(driver.cuMemSetAccess(address, mapped_bytes, [access], 1), "cuMemSetAccess(initial)")
        buffer = Buffer.from_handle(address, mapped_bytes, owner=address)
        persistent = torch.from_dlpack(buffer)[:payload_bytes]
        pointer_before_map = persistent.data_ptr()
        if pointer_before_map != int(address):
            raise RuntimeError("PyTorch did not retain the reserved VMM address")

        _ok(driver.cuMemUnmap(address, mapped_bytes), "cuMemUnmap(initial)")
        mapped = False
        _ok(driver.cuMemRelease(handle), "cuMemRelease(initial)")
        handle = None

        (handle,) = _ok(driver.cuMemCreate(mapped_bytes, prop, 0), "cuMemCreate(remap)")
        _ok(driver.cuMemMap(address, mapped_bytes, 0, handle, 0), "cuMemMap(remap)")
        mapped = True
        _ok(driver.cuMemSetAccess(address, mapped_bytes, [access], 1), "cuMemSetAccess(remap)")
        expected = torch.arange(payload_bytes, dtype=torch.uint8, device=device)
        persistent.copy_(expected)
        torch.testing.assert_close(persistent[:4096].view(torch.uint8).cpu(), expected[:4096].cpu())
        if persistent.data_ptr() != pointer_before_map:
            raise RuntimeError("PyTorch changed the persistent VMM address")
        print("Persistent virtual VMM tensor: PASS")
        print(f"  device: {torch.cuda.get_device_name(args.device)}")
        print(f"  bytes: {payload_bytes / 1024**2:.2f} MiB")
        print(f"  address stable: 0x{pointer_before_map:x}")
    finally:
        persistent = buffer = None
        gc.collect()
        if address is not None and handle is not None and mapped:
            _ok(driver.cuMemUnmap(address, mapped_bytes), "cuMemUnmap")
            mapped = False
        if handle is not None:
            _ok(driver.cuMemRelease(handle), "cuMemRelease")
        if address is not None:
            _ok(driver.cuMemAddressFree(address, mapped_bytes), "cuMemAddressFree")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--megabytes", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
