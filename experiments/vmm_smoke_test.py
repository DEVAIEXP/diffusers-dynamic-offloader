"""Validate CUDA VMM mappings with PyTorch on a single CUDA device.

This is a clean-room feasibility probe for a future DDO VMM weight store.  It
does not install a PyTorch allocator, patch a model, or emulate ComfyUI.  It
uses the official ``cuda.bindings.driver`` VMM API to reserve one stable virtual
address, maps a physical allocation into it, exposes that mapping to PyTorch by
way of ``cuda.core.Buffer`` and DLPack, executes ``F.linear``, then unmaps and
remaps a second allocation at the exact same address.

The probe is Windows-first but does not rely on Windows-specific APIs.
"""

from __future__ import annotations

import argparse
import gc
import time


def _require_success(result: tuple[object, ...], operation: str):
    from cuda.bindings import driver

    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {result[0]}")
    return result[1:]


class VmmSlot:
    """One stable CUDA virtual-address slot with replaceable physical backing."""

    def __init__(self, device_id: int, requested_bytes: int) -> None:
        import torch
        from cuda.bindings import driver
        from cuda.core import Device

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable.")

        torch.empty(1, device=f"cuda:{device_id}")
        Device(device_id).set_current()
        _require_success(driver.cuInit(0), "cuInit")
        (device,) = _require_success(driver.cuDeviceGet(device_id), "cuDeviceGet")
        (vmm_supported,) = _require_success(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                device,
            ),
            "cuDeviceGetAttribute(VMM)",
        )
        if not vmm_supported:
            raise RuntimeError(f"CUDA device {device_id} does not support VMM.")

        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_id
        prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        (granularity,) = _require_success(
            driver.cuMemGetAllocationGranularity(
                prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ),
            "cuMemGetAllocationGranularity",
        )
        self.device_id = device_id
        self.prop = prop
        self.granularity = int(granularity)
        self.size = _round_up(requested_bytes, self.granularity)
        (self.address,) = _require_success(
            driver.cuMemAddressReserve(self.size, self.granularity, 0, 0), "cuMemAddressReserve"
        )
        self.handle = None
        self.buffer = None
        self.tensor = None

    def map_tensor(self):
        """Map fresh physical VRAM at the stable address and return uint8 storage."""
        import torch
        from cuda.bindings import driver
        from cuda.core import Buffer

        if self.handle is not None:
            raise RuntimeError("The slot is already mapped.")
        (handle,) = _require_success(driver.cuMemCreate(self.size, self.prop, 0), "cuMemCreate")
        try:
            _require_success(driver.cuMemMap(self.address, self.size, 0, handle, 0), "cuMemMap")
            access = driver.CUmemAccessDesc()
            access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = self.device_id
            access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _require_success(driver.cuMemSetAccess(self.address, self.size, [access], 1), "cuMemSetAccess")
        except BaseException:
            driver.cuMemRelease(handle)
            raise

        self.handle = handle
        self.buffer = Buffer.from_handle(self.address, self.size, owner=self)
        self.tensor = torch.from_dlpack(self.buffer)
        return self.tensor

    def unmap(self) -> None:
        """Drop PyTorch views, then detach physical VRAM while retaining the VA."""
        from cuda.bindings import driver

        if self.handle is None:
            return
        self.tensor = None
        self.buffer = None
        gc.collect()
        _require_success(driver.cuMemUnmap(self.address, self.size), "cuMemUnmap")
        _require_success(driver.cuMemRelease(self.handle), "cuMemRelease")
        self.handle = None

    def close(self) -> None:
        from cuda.bindings import driver

        self.unmap()
        if self.address is not None:
            _require_success(driver.cuMemAddressFree(self.address, self.size), "cuMemAddressFree")
            self.address = None


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _mapped_weight(slot: VmmSlot, source, shape):
    import torch

    storage = slot.map_tensor()
    weight = storage.view(torch.bfloat16)[: source.numel()].view(shape)
    started = time.perf_counter()
    weight.copy_(source, non_blocking=False)
    torch.cuda.synchronize(slot.device_id)
    return weight, time.perf_counter() - started


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as functional

    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(args.seed)
    shape = (args.out_features, args.in_features)
    requested_bytes = args.out_features * args.in_features * torch.tensor([], dtype=torch.bfloat16).element_size()
    source_a = torch.randn(shape, dtype=torch.bfloat16)
    source_b = torch.full(shape, 0.25, dtype=torch.bfloat16)
    inputs = torch.randn((args.batch, args.in_features), device=device, dtype=torch.bfloat16)
    slot = VmmSlot(args.device, requested_bytes)
    stable_address = int(slot.address)

    try:
        weight_a, copy_a_seconds = _mapped_weight(slot, source_a, shape)
        result_a = functional.linear(inputs, weight_a)
        torch.cuda.synchronize(args.device)
        expected_a = functional.linear(inputs, source_a.to(device))
        torch.testing.assert_close(result_a, expected_a, rtol=2e-2, atol=2e-2)
        first_pointer = weight_a.data_ptr()
        del weight_a, result_a, expected_a
        torch.cuda.synchronize(args.device)
        slot.unmap()

        weight_b, copy_b_seconds = _mapped_weight(slot, source_b, shape)
        if weight_b.data_ptr() != first_pointer or int(weight_b.data_ptr()) != stable_address:
            raise AssertionError("VMM remap did not preserve the virtual address.")
        result_b = functional.linear(inputs, weight_b)
        torch.cuda.synchronize(args.device)
        expected_b = functional.linear(inputs, source_b.to(device))
        torch.testing.assert_close(result_b, expected_b, rtol=2e-2, atol=2e-2)
        if torch.equal(result_b, functional.linear(inputs, source_a.to(device))):
            raise AssertionError("Remapped weight did not change the linear result.")

        print("VMM + PyTorch smoke test: PASS")
        print(f"  device: {torch.cuda.get_device_name(args.device)}")
        print(f"  VMM granularity: {slot.granularity / 1024**2:.2f} MiB")
        print(f"  requested/mapped: {requested_bytes / 1024**2:.2f} / {slot.size / 1024**2:.2f} MiB")
        print(f"  stable virtual address: 0x{stable_address:x}")
        print(f"  H2D copy A/B: {copy_a_seconds:.4f}s / {copy_b_seconds:.4f}s")
    finally:
        slot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
