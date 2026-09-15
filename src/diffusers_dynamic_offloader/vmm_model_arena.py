"""DDO VMM model arena backed by the native C++ allocator.

The DDO layer owns model-agnostic policy: grouping, hooks, prefetch, metrics,
and profile integration.  The native allocator owns CUDA virtual addresses,
physical residency, map/unmap operations, budget guards, and transfer counters.
"""

from __future__ import annotations

import sys
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .vmm_weight_store import PinnedStagingPool, RegisteredHostWeightRegistry, SafetensorSliceRegistry, _round_up

if TYPE_CHECKING:
    import torch


_ARENA_PAGE_BYTES = 32 * 1024**2


@dataclass
class VmmModelArenaStats:
    reserved_bytes: int = 0
    resident_bytes: int = 0
    materializations: int = 0
    evictions: int = 0
    copied_bytes: int = 0
    map_seconds: float = 0.0
    direct_dma_seconds: float = 0.0
    staging_seconds: float = 0.0
    file_slice_staging_seconds: float = 0.0
    file_slice_bytes: int = 0
    temporary_fallbacks: int = 0
    temporary_bytes: int = 0
    persistent_virtual_views: int = 0


@dataclass(frozen=True)
class _ArenaWeightView:
    name: str
    offset: int
    shape: tuple[int, ...]
    dtype: Any
    bytes: int


class _ArenaGroup:
    def __init__(self, key: str, allocation_bytes: int, payload_bytes: int, views: dict[str, _ArenaWeightView]):
        self.key = key
        self.allocation_bytes = allocation_bytes
        self.payload_bytes = payload_bytes
        self.views = views
        self.tensor_views: dict[str, Any] = {}
        self.temporary_views: dict[str, Any] | None = None
        self.active_leases = 0


class VmmModelArenaLease(AbstractContextManager):
    def __init__(self, arena: VmmModelArena, group_key: str) -> None:
        self._arena = arena
        self._group_key = group_key
        self._closed = False

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            self._arena.release(self._group_key)
            self._closed = True


def _allocator_source_root() -> Path:
    base = Path(__file__).resolve().parents[2]
    candidate = base / "ddo-vmm-allocator" / "src"
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parents[3] / "ddo-vmm-allocator" / "src"

def _load_native_allocator(library_path: str | None):
    try:
        from ddo_vmm_allocator.native import load_native_vmm
        from ddo_vmm_allocator.weight_arena import VmmWeightArena
    except ImportError:
        source_root = _allocator_source_root()
        if source_root.exists() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            from ddo_vmm_allocator.native import load_native_vmm
            from ddo_vmm_allocator.weight_arena import VmmWeightArena
        except ImportError as error:
            raise RuntimeError(
                "vmm_runtime requires ddo-vmm-allocator. Install it or run from the DDO source tree."
            ) from error

    resolved_library = library_path
    if resolved_library is None:
        root = _allocator_source_root().parent
        platform_library = root / "build" / "Release" / "ddo_vmm.dll"
        if platform_library.exists():
            resolved_library = str(platform_library)
    return load_native_vmm(resolved_library), VmmWeightArena


class VmmModelArena:
    """Stable model-weight views backed by the native C++ VMM allocator."""

    def __init__(
        self,
        device_id: int,
        max_resident_bytes: int,
        *,
        page_bytes: int = _ARENA_PAGE_BYTES,
        library_path: str | None = None,
        file_slices: SafetensorSliceRegistry | None = None,
        staging_pool: PinnedStagingPool | None = None,
        registered_host: RegisteredHostWeightRegistry | None = None,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable.")
        if max_resident_bytes <= 0:
            raise ValueError("VMM arena resident budget must be positive")
        library, arena_type = _load_native_allocator(library_path)
        self.native_library_path = str(library.path)
        self.device_id = device_id
        self.max_resident_bytes = max_resident_bytes
        self.page_bytes = page_bytes
        self._native = arena_type(
            library,
            resident_budget_bytes=max_resident_bytes,
            device_ordinal=device_id,
            backing="device",
            pin_source=False,
        )
        if file_slices is not None:
            if staging_pool is None:
                raise ValueError("VMM file transport requires a staging pool")
            self._native.configure_file_transport(file_slices, staging_pool)
        if registered_host is not None:
            self._native.configure_registered_host_transport(registered_host)
        self._groups: dict[str, _ArenaGroup] = {}
        self._weight_groups: dict[str, str] = {}
        self._reserved_bytes = 0
        self._persistent_virtual_views = 0
        self._lock = threading.RLock()
        self._closed = False

    @property
    def stats(self) -> VmmModelArenaStats:
        with self._lock:
            stats = self._native.runtime_stats()
            return VmmModelArenaStats(
                reserved_bytes=self._reserved_bytes,
                resident_bytes=int(stats.get("resident_bytes", 0)),
                materializations=int(stats.get("materializations", 0)),
                evictions=int(stats.get("evictions", 0)),
                copied_bytes=int(stats.get("upload_bytes", 0)),
                file_slice_bytes=int(stats.get("file_upload_bytes", 0)),
                temporary_fallbacks=int(stats.get("temporary_groups", 0)),
                temporary_bytes=int(stats.get("temporary_bytes", 0)),
                persistent_virtual_views=self._persistent_virtual_views,
            )

    @property
    def physical_page_bytes(self) -> int:
        return self.page_bytes

    def reserve_groups(self, groups: list[tuple[str, list[tuple[str, torch.Tensor]]]]) -> None:
        with self._lock:
            if self._groups:
                raise RuntimeError("VMM arena groups were already reserved")
            for group_key, weights in groups:
                if not weights:
                    raise ValueError(f"VMM arena group {group_key!r} has no weights")
                offset = 0
                views: dict[str, _ArenaWeightView] = {}
                for weight_name, weight in weights:
                    if weight_name in self._weight_groups:
                        raise ValueError(f"Duplicate VMM arena weight {weight_name!r}")
                    size_bytes = weight.numel() * weight.element_size()
                    views[weight_name] = _ArenaWeightView(
                        weight_name,
                        offset,
                        tuple(weight.shape),
                        weight.dtype,
                        size_bytes,
                    )
                    offset += size_bytes
                allocation_bytes = _round_up(offset, self.page_bytes)
                self._native.reserve(group_key, allocation_bytes)
                tensor_views = self._native.bind_cpu_tensor_group(group_key, weights)
                group = _ArenaGroup(group_key, allocation_bytes, offset, views)
                group.tensor_views = tensor_views
                self._groups[group_key] = group
                self._weight_groups.update(dict.fromkeys(views, group_key))
                self._reserved_bytes += allocation_bytes
                self._persistent_virtual_views += len(views)
            self._native.reset_runtime_stats()

    def configure_managed_backing(self, managed_backing: Any | None) -> None:
        if managed_backing is not None:
            self._native.configure_managed_source(managed_backing)

    def acquire(
        self,
        group_key: str,
        sources: list[tuple[str, torch.Tensor]],
        staging: PinnedStagingPool,
        stream: Any,
        registered_host: RegisteredHostWeightRegistry | None = None,
        file_slices: SafetensorSliceRegistry | None = None,
        managed_backing: Any | None = None,
        allow_eviction: bool = True,
    ) -> VmmModelArenaLease:
        del sources, staging, registered_host, file_slices, managed_backing, allow_eviction
        with self._lock:
            self._require_open()
            group = self._groups[group_key]
            self._native.collect_completed_events(wait=False)
            try:
                self._native.prefetch_loaded_group(group_key, stream)
            except Exception:
                self._native.collect_completed_events(wait=True)
                self._native.prefetch_loaded_group(group_key, stream)

            group.active_leases += 1
            return VmmModelArenaLease(self, group_key)

    def mark_upload(self, group_key: str, stream: Any, staging: PinnedStagingPool) -> Any:
        import torch

        event = torch.cuda.Event()
        event.record(stream)
        if staging.has_active_upload:
            staging.mark_upload(event)
        return event

    def weight(self, weight_name: str) -> torch.Tensor:
        with self._lock:
            self._require_open()
            group = self._groups[self._weight_groups[weight_name]]
            if group.temporary_views is not None and weight_name in group.temporary_views:
                return group.temporary_views[weight_name]
            return group.tensor_views[weight_name]

    def contains_weight(self, weight_name: str) -> bool:
        return weight_name in self._weight_groups

    def release(self, group_key: str) -> None:
        with self._lock:
            self._require_open()
            group = self._groups[group_key]
            if group.active_leases <= 0:
                raise RuntimeError(f"VMM arena group {group_key!r} was released without an active lease")
            group.active_leases -= 1
            if group.active_leases == 0:
                self._native.release_after_current_stream(group_key)
                group.temporary_views = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for group in self._groups.values():
                if group.active_leases:
                    raise RuntimeError(f"Cannot close VMM arena with active group {group.key!r}")
            self._native.collect_completed_events(wait=True)
            self._native.close()
            self._groups.clear()
            self._weight_groups.clear()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("VMM arena is closed")