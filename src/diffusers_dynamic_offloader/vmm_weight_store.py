"""Optional CUDA VMM-backed storage for immutable, dense model weights.

The store deliberately owns only a small, explicit part of memory management:
stable virtual-address slots and their physical CUDA mappings.  It does not
replace PyTorch's allocator or expose unmapped storage to a caller.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _cuda_modules() -> tuple[Any, Any, Any]:
    try:
        from cuda.bindings import driver
        from cuda.core import Buffer, Device
    except ImportError as error:  # pragma: no cover - depends on optional extra.
        raise RuntimeError(
            "CUDA VMM support requires the 'vmm' extra: pip install 'diffusers-dynamic-offloader[vmm]'."
        ) from error
    return driver, Buffer, Device


def _require_success(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    driver, _, _ = _cuda_modules()
    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {result[0]}")
    return result[1:]


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass
class VmmWeightStoreStats:
    reserved_bytes: int = 0
    resident_bytes: int = 0
    materializations: int = 0
    evictions: int = 0
    copied_bytes: int = 0


@dataclass(frozen=True)
class PinnedStagingPoolStats:
    """Host-side transport timings for the bounded VMM staging ring."""

    slot_reuse_wait_seconds: float = 0.0
    slot_reuse_waits: int = 0
    source_copy_seconds: float = 0.0
    source_copy_bytes: int = 0
    shard_read_seconds: float = 0.0
    shard_read_bytes: int = 0


class _VmmWeightSlot:
    def __init__(self, key: str, address: Any, size: int) -> None:
        self.key = key
        self.address = address
        self.size = size
        self.handle: Any | None = None
        self.buffer: Any | None = None
        self.storage: torch.Tensor | None = None
        self.active_leases = 0
        self.last_used = 0.0
        self.last_event: Any | None = None

    @property
    def is_mapped(self) -> bool:
        return self.handle is not None


class VmmWeightLease(AbstractContextManager):
    """A scoped, CUDA-ordered view of one materialized immutable weight."""

    def __init__(self, store: VmmWeightStore, slot: _VmmWeightSlot, tensor: torch.Tensor) -> None:
        self._store = store
        self._slot = slot
        self.tensor = tensor

    def __enter__(self) -> torch.Tensor:
        return self.tensor

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._store.release(self._slot.key)


class _WindowsRegisteredVirtualBuffer:
    """One contiguous Windows host allocation registered once with CUDA.

    This deliberately owns a bounded staging slot, not model-weight storage.
    The raw address is valid for direct ``cuMemcpyHtoDAsync`` while the slot's
    CUDA event is outstanding.
    """

    _MEM_COMMIT = 0x1000
    _MEM_RESERVE = 0x2000
    _MEM_RELEASE = 0x8000
    _PAGE_READWRITE = 0x04

    def __init__(self, size_bytes: int) -> None:
        import torch

        if os.name != "nt":
            raise RuntimeError("Registered virtual host buffers are Windows-only")
        self.size_bytes = size_bytes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong)
        self._kernel32.VirtualAlloc.restype = ctypes.c_void_p
        self._kernel32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong)
        self._kernel32.VirtualFree.restype = ctypes.c_int
        address = self._kernel32.VirtualAlloc(
            None, size_bytes, self._MEM_RESERVE | self._MEM_COMMIT, self._PAGE_READWRITE
        )
        if not address:
            raise ctypes.WinError(ctypes.get_last_error())
        self.address = int(address)
        self._raw = (ctypes.c_ubyte * size_bytes).from_address(self.address)
        self.tensor = torch.frombuffer(self._raw, dtype=torch.uint8)
        self._registered = False
        try:
            driver, _, _ = _cuda_modules()
            _require_success(
                driver.cuMemHostRegister(self.address, size_bytes, driver.CU_MEMHOSTREGISTER_PORTABLE),
                "cuMemHostRegister(virtual staging buffer)",
            )
            self._registered = True
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.address == 0:
            return
        if self._registered:
            driver, _, _ = _cuda_modules()
            _require_success(driver.cuMemHostUnregister(self.address), "cuMemHostUnregister(virtual staging buffer)")
            self._registered = False
        self.tensor = None
        self._raw = None
        address = self.address
        self.address = 0
        if not self._kernel32.VirtualFree(ctypes.c_void_p(address), 0, self._MEM_RELEASE):
            raise ctypes.WinError(ctypes.get_last_error())


class PinnedStagingPool:
    """Small ring of bounded pinned buffers for asynchronous weight uploads.

    One buffer makes a next-group prefetch synchronize the upload of the
    current group before CUDA can start computing it.  Two slots let the
    prefetch stream overlap that next upload with current-group compute while
    keeping host-pinned memory explicitly bounded.
    """

    def __init__(
        self,
        size_bytes: int,
        slots: int = 2,
        *,
        registered_virtual_host: bool = False,
        windows_file_read_chunk_bytes: int = 2 * 1024**2,
    ) -> None:
        import torch

        if size_bytes <= 0:
            raise ValueError("Pinned staging buffer size must be positive")
        if slots <= 0:
            raise ValueError("Pinned staging slot count must be positive")
        if windows_file_read_chunk_bytes <= 0:
            raise ValueError("Windows file-read chunk size must be positive")
        self.size_bytes = size_bytes
        self.slots = slots
        self._virtual_buffers: list[_WindowsRegisteredVirtualBuffer] = []
        if registered_virtual_host:
            if os.name != "nt":
                raise RuntimeError("Registered virtual host transport is Windows-only")
            try:
                self._virtual_buffers = [_WindowsRegisteredVirtualBuffer(size_bytes) for _ in range(slots)]
                self._buffers = [buffer.tensor for buffer in self._virtual_buffers]
            except BaseException:
                for buffer in reversed(self._virtual_buffers):
                    buffer.close()
                self._virtual_buffers.clear()
                raise
        else:
            self._buffers = [torch.empty(size_bytes, dtype=torch.uint8, device="cpu", pin_memory=True) for _ in range(slots)]
        self._ready_events: list[Any | None] = [None] * slots
        self._stats = PinnedStagingPoolStats()
        self._next_slot = 0
        self._active_slot: int | None = None
        self._offset = 0
        self._checkpoint_files: dict[Path, Any] = {}
        self._checkpoint_maps: dict[Path, mmap.mmap] = {}
        self._file_lock = threading.Lock()
        self._windows_file_reader = None
        if os.name == "nt":
            from .windows_file_reader import WindowsPinnedFileReader

            self._windows_file_reader = WindowsPinnedFileReader(chunk_bytes=windows_file_read_chunk_bytes)

    def begin_upload(self) -> None:
        """Acquire the next ring slot, waiting only when that slot is reused."""
        slot = self._next_slot
        self._next_slot = (slot + 1) % self.slots
        event = self._ready_events[slot]
        if event is not None:
            start = time.perf_counter()
            event.synchronize()
            self._stats = PinnedStagingPoolStats(
                slot_reuse_wait_seconds=self._stats.slot_reuse_wait_seconds + time.perf_counter() - start,
                slot_reuse_waits=self._stats.slot_reuse_waits + 1,
                source_copy_seconds=self._stats.source_copy_seconds,
                source_copy_bytes=self._stats.source_copy_bytes,
                shard_read_seconds=self._stats.shard_read_seconds,
                shard_read_bytes=self._stats.shard_read_bytes,
            )
        self._active_slot = slot
        self._offset = 0

    @property
    def next_slot_ready(self) -> bool:
        """Whether a speculative upload can acquire its next slot without waiting."""

        if self._active_slot is not None:
            return False
        event = self._ready_events[self._next_slot]
        return event is None or event.query()

    def _active_buffer(self) -> Any:
        if self._active_slot is None:
            raise RuntimeError("Call begin_upload before staging VMM data")
        return self._buffers[self._active_slot]

    def stage(self, source: torch.Tensor) -> torch.Tensor:
        if source.device.type != "cpu" or not source.is_contiguous():
            raise ValueError("Pinned staging accepts contiguous CPU tensors only")
        size_bytes = source.numel() * source.element_size()
        end = self._offset + size_bytes
        if end > self.size_bytes:
            raise RuntimeError(
                f"Pinned staging buffer is too small: need {end / 1024**2:.1f} MiB, "
                f"have {self.size_bytes / 1024**2:.1f} MiB"
            )
        storage = self._active_buffer()[self._offset : end]
        staged = storage.view(dtype=source.dtype).view(source.shape)
        start = time.perf_counter()
        staged.copy_(source)
        self._stats = PinnedStagingPoolStats(
            slot_reuse_wait_seconds=self._stats.slot_reuse_wait_seconds,
            slot_reuse_waits=self._stats.slot_reuse_waits,
            source_copy_seconds=self._stats.source_copy_seconds + time.perf_counter() - start,
            source_copy_bytes=self._stats.source_copy_bytes + size_bytes,
            shard_read_seconds=self._stats.shard_read_seconds,
            shard_read_bytes=self._stats.shard_read_bytes,
        )
        self._offset = end
        return staged

    def stage_file_slice(self, source: torch.Tensor, file_slice: "SafetensorFileSlice") -> torch.Tensor:
        """Read one checkpoint byte range directly into the active pinned slot."""
        source_device = getattr(source, "device", None)
        source_contiguous = getattr(source, "is_contiguous", None)
        if ((source_device is not None and source_device.type != "cpu") or (
            callable(source_contiguous) and not source_contiguous()
        )):
            raise ValueError("Pinned file staging accepts contiguous CPU tensors only")
        size_bytes = source.numel() * source.element_size()
        if size_bytes != file_slice.size_bytes:
            raise ValueError("Safetensor slice size does not match the source tensor")
        end = self._offset + size_bytes
        if end > self.size_bytes:
            raise RuntimeError(
                f"Pinned staging buffer is too small: need {end / 1024**2:.1f} MiB, "
                f"have {self.size_bytes / 1024**2:.1f} MiB"
            )
        storage = self._active_buffer()[self._offset : end]
        self._read_file_slice(file_slice, storage)
        staged = storage.view(dtype=source.dtype).view(source.shape)
        self._offset = end
        return staged

    def stage_file_slices(self, entries: list[tuple[torch.Tensor, "SafetensorFileSlice"]]) -> list[torch.Tensor]:
        """Stage packed slices, joining adjacent ranges in the same shard."""
        if not entries:
            return []
        start = self._offset
        runs: list[tuple[SafetensorFileSlice, int, int]] = []
        staged: list[torch.Tensor] = []
        for source, file_slice in entries:
            source_device = getattr(source, "device", None)
            source_contiguous = getattr(source, "is_contiguous", None)
            if ((source_device is not None and source_device.type != "cpu") or (
                callable(source_contiguous) and not source_contiguous()
            )):
                raise ValueError("Pinned file staging accepts contiguous CPU tensors only")
            size_bytes = source.numel() * source.element_size()
            if size_bytes != file_slice.size_bytes:
                raise ValueError("Safetensor slice size does not match the source tensor")
            end = self._offset + size_bytes
            if end > self.size_bytes:
                raise RuntimeError(
                    f"Pinned staging buffer is too small: need {end / 1024**2:.1f} MiB, "
                    f"have {self.size_bytes / 1024**2:.1f} MiB"
                )
            if (
                runs
                and runs[-1][0].path == file_slice.path
                and runs[-1][0].offset_bytes + runs[-1][2] == file_slice.offset_bytes
                and runs[-1][1] + runs[-1][2] == self._offset
            ):
                previous_slice, destination_offset, run_bytes = runs[-1]
                runs[-1] = (previous_slice, destination_offset, run_bytes + size_bytes)
            else:
                runs.append((file_slice, self._offset, size_bytes))
            staged.append(self._active_buffer()[self._offset : end].view(dtype=source.dtype).view(source.shape))
            self._offset = end
        for file_slice, destination_offset, run_bytes in runs:
            self._read_file_slice(
                SafetensorFileSlice(file_slice.path, file_slice.offset_bytes, run_bytes),
                self._active_buffer()[destination_offset : destination_offset + run_bytes],
            )
        assert self._offset - start == sum(file_slice.size_bytes for _source, file_slice in entries)
        return staged

    def stage_file_slices_async(
        self, entries: list[tuple[torch.Tensor, "SafetensorFileSlice"]]
    ) -> tuple[list[torch.Tensor], list[Any]]:
        """Reserve one slot and start shard reads without blocking its consumer.

        The caller owns the returned futures and must wait for them before
        placing a CUDA DMA on this slot, then call :meth:`mark_upload` as
        usual. Non-Windows transports deliberately retain the synchronous
        path because their mmap fallback is already local and deterministic.
        """

        if self._windows_file_reader is None:
            return self.stage_file_slices(entries), []
        if not entries:
            return [], []
        start = self._offset
        runs: list[tuple[SafetensorFileSlice, int, int]] = []
        staged: list[torch.Tensor] = []
        for source, file_slice in entries:
            size_bytes = source.numel() * source.element_size()
            if size_bytes != file_slice.size_bytes:
                raise ValueError("Safetensor slice size does not match the source tensor")
            end = self._offset + size_bytes
            if end > self.size_bytes:
                raise RuntimeError("Pinned staging buffer is too small for asynchronous file staging")
            if (
                runs
                and runs[-1][0].path == file_slice.path
                and runs[-1][0].offset_bytes + runs[-1][2] == file_slice.offset_bytes
                and runs[-1][1] + runs[-1][2] == self._offset
            ):
                previous_slice, destination_offset, run_bytes = runs[-1]
                runs[-1] = (previous_slice, destination_offset, run_bytes + size_bytes)
            else:
                runs.append((file_slice, self._offset, size_bytes))
            staged.append(self._active_buffer()[self._offset : end].view(dtype=source.dtype).view(source.shape))
            self._offset = end
        futures = [
            self._windows_file_reader.read_at_async(
                file_slice.path,
                file_slice.offset_bytes,
                self._active_buffer()[destination_offset : destination_offset + run_bytes].data_ptr(),
                run_bytes,
            )
            for file_slice, destination_offset, run_bytes in runs
        ]
        assert self._offset - start == sum(file_slice.size_bytes for _source, file_slice in entries)
        return staged, futures

    def _read_file_slice(self, file_slice: "SafetensorFileSlice", storage: torch.Tensor) -> None:
        start = time.perf_counter()
        if self._windows_file_reader is not None:
            self._windows_file_reader.read_at(
                file_slice.path,
                file_slice.offset_bytes,
                storage.data_ptr(),
                file_slice.size_bytes,
            )
            self._record_shard_read(file_slice.size_bytes, time.perf_counter() - start)
            return
        target = memoryview(storage.numpy()).cast("B")
        with self._file_lock:
            mapped = self._checkpoint_maps.get(file_slice.path)
            if mapped is None:
                checkpoint = open(file_slice.path, "rb", buffering=0)  # noqa: SIM115 - mapping owns this handle until close.
                mapped = mmap.mmap(checkpoint.fileno(), 0, access=mmap.ACCESS_READ)
                self._checkpoint_files[file_slice.path] = checkpoint
                self._checkpoint_maps[file_slice.path] = mapped
            end = file_slice.offset_bytes + file_slice.size_bytes
            if end > mapped.size():
                raise OSError("safetensor slice exceeds mapped shard")
            target[:] = mapped[file_slice.offset_bytes:end]
            read = file_slice.size_bytes
        if read != file_slice.size_bytes:
            raise OSError(
                f"Short safetensor read from {file_slice.path!s}: expected {file_slice.size_bytes}, got {read}"
            )
        self._record_shard_read(file_slice.size_bytes, time.perf_counter() - start)

    def _record_shard_read(self, size_bytes: int, seconds: float) -> None:
        self._stats = PinnedStagingPoolStats(
            slot_reuse_wait_seconds=self._stats.slot_reuse_wait_seconds,
            slot_reuse_waits=self._stats.slot_reuse_waits,
            source_copy_seconds=self._stats.source_copy_seconds,
            source_copy_bytes=self._stats.source_copy_bytes,
            shard_read_seconds=self._stats.shard_read_seconds + seconds,
            shard_read_bytes=self._stats.shard_read_bytes + size_bytes,
        )

    @property
    def transport_stats(self) -> PinnedStagingPoolStats:
        """Return a stable snapshot; timings are host-wall-clock measurements."""
        return self._stats

    def mark_upload(self, event: Any) -> None:
        if self._active_slot is None:
            raise RuntimeError("Cannot mark a VMM upload without an active staging slot")
        self._ready_events[self._active_slot] = event
        self._active_slot = None

    @property
    def has_active_upload(self) -> bool:
        """Whether the current materialization borrowed a staging slot.

        A resident VMM group is acquired again on later denoise steps without
        staging another host upload.  Driver-DMA paths likewise bypass the
        pool.  Keeping this state explicit prevents either path from trying to
        retire a slot that it never acquired.
        """
        return self._active_slot is not None

    @property
    def packed_bytes(self) -> torch.Tensor:
        return self._active_buffer()[: self._offset]

    @property
    def uses_windows_overlapped_file_io(self) -> bool:
        return self._windows_file_reader is not None

    @property
    def uses_registered_virtual_host(self) -> bool:
        return bool(self._virtual_buffers)

    def close(self) -> None:
        if self._windows_file_reader is not None:
            self._windows_file_reader.close()
            self._windows_file_reader = None
        with self._file_lock:
            for mapped in self._checkpoint_maps.values():
                mapped.close()
            self._checkpoint_maps.clear()
            for checkpoint in self._checkpoint_files.values():
                checkpoint.close()
            self._checkpoint_files.clear()
        for buffer in reversed(self._virtual_buffers):
            buffer.close()
        self._virtual_buffers.clear()
        self._buffers.clear()


@dataclass(frozen=True)
class SafetensorFileSlice:
    """Byte range of a tensor payload in one local safetensors file."""

    path: Path
    offset_bytes: int
    size_bytes: int
    shape: tuple[int, ...] | None = None
    dtype: str | None = None


class SafetensorSliceRegistry:
    """Maps CPU storage addresses to immutable, shard-local file descriptors.

    The registry deliberately keeps neither a tensor nor a mapped file alive:
    its entries contain only a storage address, byte count, and frozen
    :class:`SafetensorFileSlice`. The transport owns file mappings and must
    close them separately before this registry is discarded.
    """

    def __init__(self) -> None:
        self._slices: dict[tuple[int, int], SafetensorFileSlice] = {}
        self._closed = False

    def bind(self, tensor: torch.Tensor, file_slice: SafetensorFileSlice) -> bool:
        self._require_open()
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            return False
        size_bytes = tensor.numel() * tensor.element_size()
        if size_bytes != file_slice.size_bytes:
            return False
        if file_slice.shape is not None and tuple(tensor.shape) != file_slice.shape:
            return False
        dtype_names = {
            "BOOL": "torch.bool", "U8": "torch.uint8", "I8": "torch.int8",
            "I16": "torch.int16", "I32": "torch.int32", "I64": "torch.int64",
            "F16": "torch.float16", "BF16": "torch.bfloat16",
            "F32": "torch.float32", "F64": "torch.float64",
            "F8_E4M3": "torch.float8_e4m3fn", "F8_E5M2": "torch.float8_e5m2",
        }
        if file_slice.dtype is not None and dtype_names.get(file_slice.dtype) != str(tensor.dtype):
            return False
        key = (tensor.data_ptr(), size_bytes)
        if key in self._slices and self._slices[key] != file_slice:
            return False
        self._slices[key] = file_slice
        return True

    def lookup(self, tensor: torch.Tensor) -> SafetensorFileSlice | None:
        self._require_open()
        return self._slices.get((tensor.data_ptr(), tensor.numel() * tensor.element_size()))

    @property
    def descriptor_count(self) -> int:
        return len(self._slices)

    @property
    def descriptor_bytes(self) -> int:
        return sum(file_slice.size_bytes for file_slice in self._slices.values())

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Drop all descriptors without retaining CPU tensor storage."""
        self._slices.clear()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Safetensor slice registry is closed")


class RegisteredHostWeightRegistry:
    """Bounded registration of immutable CPU weight storage for direct CUDA DMA.

    Unlike ``Tensor.pin_memory()``, registration does not create a second host
    copy.  Each successful registration is retained until the VMM runtime is
    detached, so an upload stream can use ``cuMemcpyHtoDAsync`` with the
    original safetensor/PyTorch storage.
    """

    def __init__(self, device_id: int, *, max_bytes: int) -> None:
        import torch

        driver, _, Device = _cuda_modules()
        torch.empty(1, device=f"cuda:{device_id}")
        Device(device_id).set_current()
        _require_success(driver.cuInit(0), "cuInit")
        (device,) = _require_success(driver.cuDeviceGet(device_id), "cuDeviceGet")
        (readonly_supported,) = _require_success(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
                device,
            ),
            "cuDeviceGetAttribute(READ_ONLY_HOST_REGISTER_SUPPORTED)",
        )
        self._readonly_supported = bool(readonly_supported)
        if max_bytes <= 0:
            raise ValueError("Registered host cache budget must be positive")
        self.max_bytes = int(max_bytes)
        self.registered_bytes = 0
        self.skipped_bytes = 0
        self._registrations: dict[tuple[int, int], tuple[torch.Tensor, bool]] = {}
        self._budget_skips: set[tuple[int, int]] = set()
        self._lock = threading.RLock()

    def register(self, source: torch.Tensor) -> bool:
        """Register ``source`` in place; return false if CUDA rejects it."""
        if source.device.type != "cpu" or not source.is_contiguous():
            return False
        size_bytes = source.numel() * source.element_size()
        if size_bytes <= 0:
            return True
        key = (source.data_ptr(), size_bytes)
        with self._lock:
            if key in self._registrations:
                return True
            if self.registered_bytes + size_bytes > self.max_bytes:
                if key not in self._budget_skips:
                    self._budget_skips.add(key)
                    self.skipped_bytes += size_bytes
                return False
            driver, _, _ = _cuda_modules()
            flags = driver.CU_MEMHOSTREGISTER_PORTABLE
            if self._readonly_supported:
                flags |= driver.CU_MEMHOSTREGISTER_READ_ONLY
            result = driver.cuMemHostRegister(source.data_ptr(), size_bytes, flags)
            if result[0] == driver.CUresult.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
                # Adjacent safetensor views can share an OS page. CUDA rejects
                # the duplicate registration, but the range is already usable
                # for direct copies. Do not unregister a range we do not own.
                self._registrations[key] = (source, False)
                self.registered_bytes += size_bytes
                return True
            if result[0] != driver.CUresult.CUDA_SUCCESS:
                return False
            self._registrations[key] = (source, True)
            self.registered_bytes += size_bytes
            return True

    def contains(self, source: torch.Tensor) -> bool:
        return (source.data_ptr(), source.numel() * source.element_size()) in self._registrations

    def close(self) -> None:
        driver, _, _ = _cuda_modules()
        with self._lock:
            for (pointer, _size), (_source, owned) in self._registrations.items():
                if owned:
                    _require_success(driver.cuMemHostUnregister(pointer), "cuMemHostUnregister")
            self._registrations.clear()
            self._budget_skips.clear()
            self.registered_bytes = 0
            self.skipped_bytes = 0


class VmmWeightStore:
    """Private VMM cache with stable addresses and bounded resident backing.

    Callers must keep a leased tensor only inside its context.  Eviction waits
    for the release event, then unmaps its physical backing while retaining the
    slot's virtual address for a later materialization.
    """

    def __init__(self, device_id: int = 0, max_resident_bytes: int = 0) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable.")
        if max_resident_bytes < 0:
            raise ValueError("max_resident_bytes must be non-negative")
        driver, _, Device = _cuda_modules()
        torch.empty(1, device=f"cuda:{device_id}")
        Device(device_id).set_current()
        _require_success(driver.cuInit(0), "cuInit")
        (device,) = _require_success(driver.cuDeviceGet(device_id), "cuDeviceGet")
        (supported,) = _require_success(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                device,
            ),
            "cuDeviceGetAttribute(VMM)",
        )
        if not supported:
            raise RuntimeError(f"CUDA device {device_id} does not support virtual memory management.")

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
        self.max_resident_bytes = max_resident_bytes
        self.granularity = int(granularity)
        self._prop = prop
        self._slots: dict[str, _VmmWeightSlot] = {}
        self._stats = VmmWeightStoreStats()
        self._lock = threading.RLock()

    @property
    def stats(self) -> VmmWeightStoreStats:
        with self._lock:
            return VmmWeightStoreStats(**self._stats.__dict__)

    def reserve(self, key: str, size_bytes: int) -> None:
        if not key:
            raise ValueError("VMM slot key must not be empty")
        if size_bytes <= 0:
            raise ValueError("VMM slot size must be positive")
        driver, _, _ = _cuda_modules()
        with self._lock:
            existing = self._slots.get(key)
            if existing is not None:
                if existing.size < size_bytes:
                    raise ValueError(f"Existing VMM slot {key!r} is smaller than the requested weight")
                return
            size = _round_up(size_bytes, self.granularity)
            (address,) = _require_success(
                driver.cuMemAddressReserve(size, self.granularity, 0, 0), "cuMemAddressReserve"
            )
            self._slots[key] = _VmmWeightSlot(key, address, size)
            self._stats.reserved_bytes += size

    def acquire(self, key: str, source: torch.Tensor, *, stream: Any | None = None) -> VmmWeightLease:
        """Return a leased CUDA view, copying ``source`` only when first mapped."""
        import torch

        if source.device.type != "cpu":
            raise ValueError("VmmWeightStore source weights must stay on CPU")
        if not source.is_contiguous():
            raise ValueError("VmmWeightStore source weights must be contiguous")
        size_bytes = source.numel() * source.element_size()
        self.reserve(key, size_bytes)
        with self._lock:
            slot = self._slots[key]
            copy_stream = stream or torch.cuda.current_stream(self.device_id)
            if slot.last_event is not None:
                copy_stream.wait_event(slot.last_event)
            if not slot.is_mapped:
                self._ensure_capacity(slot.size)
                self._map(slot)
                typed = slot.storage.view(dtype=source.dtype)[: source.numel()].view(source.shape)
                with torch.cuda.stream(copy_stream):
                    typed.copy_(source, non_blocking=source.is_pinned())
                self._stats.materializations += 1
                self._stats.copied_bytes += size_bytes
            else:
                typed = slot.storage.view(dtype=source.dtype)[: source.numel()].view(source.shape)
            slot.active_leases += 1
            slot.last_used = time.monotonic()
            return VmmWeightLease(self, slot, typed)

    def release(self, key: str) -> None:
        import torch

        with self._lock:
            slot = self._slots[key]
            if slot.active_leases <= 0:
                raise RuntimeError(f"VMM slot {key!r} was released without an active lease")
            slot.active_leases -= 1
            slot.last_used = time.monotonic()
            if slot.active_leases == 0:
                slot.last_event = torch.cuda.Event()
                slot.last_event.record(torch.cuda.current_stream(self.device_id))

    def evict(self, key: str) -> bool:
        with self._lock:
            slot = self._slots[key]
            if not slot.is_mapped:
                return False
            if slot.active_leases:
                return False
            if slot.last_event is not None:
                slot.last_event.synchronize()
            self._unmap(slot)
            return True

    def close(self) -> None:
        driver, _, _ = _cuda_modules()
        with self._lock:
            for slot in list(self._slots.values()):
                if slot.is_mapped:
                    if slot.active_leases:
                        raise RuntimeError(f"Cannot close VMM store with active lease {slot.key!r}")
                    if slot.last_event is not None:
                        slot.last_event.synchronize()
                    self._unmap(slot)
                _require_success(driver.cuMemAddressFree(slot.address, slot.size), "cuMemAddressFree")
            self._slots.clear()

    def _ensure_capacity(self, required_bytes: int) -> None:
        if self.max_resident_bytes <= 0:
            return
        while self._stats.resident_bytes + required_bytes > self.max_resident_bytes:
            candidates = sorted(
                (slot for slot in self._slots.values() if slot.is_mapped and not slot.active_leases),
                key=lambda slot: slot.last_used,
            )
            if not candidates:
                raise RuntimeError("VMM resident budget is too small for the requested weight")
            self.evict(candidates[0].key)

    def _map(self, slot: _VmmWeightSlot) -> None:
        import torch

        driver, Buffer, _ = _cuda_modules()
        (handle,) = _require_success(driver.cuMemCreate(slot.size, self._prop, 0), "cuMemCreate")
        try:
            _require_success(driver.cuMemMap(slot.address, slot.size, 0, handle, 0), "cuMemMap")
            access = driver.CUmemAccessDesc()
            access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = self.device_id
            access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _require_success(driver.cuMemSetAccess(slot.address, slot.size, [access], 1), "cuMemSetAccess")
        except BaseException:
            driver.cuMemRelease(handle)
            raise
        slot.handle = handle
        slot.buffer = Buffer.from_handle(slot.address, slot.size, owner=slot)
        slot.storage = torch.from_dlpack(slot.buffer)
        self._stats.resident_bytes += slot.size

    def _unmap(self, slot: _VmmWeightSlot) -> None:
        import gc

        driver, _, _ = _cuda_modules()
        slot.storage = None
        slot.buffer = None
        gc.collect()
        _require_success(driver.cuMemUnmap(slot.address, slot.size), "cuMemUnmap")
        _require_success(driver.cuMemRelease(slot.handle), "cuMemRelease")
        slot.handle = None
        slot.last_event = None
        self._stats.resident_bytes -= slot.size
        self._stats.evictions += 1
