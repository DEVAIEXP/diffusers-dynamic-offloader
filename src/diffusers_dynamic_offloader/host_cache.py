"""Explicit phase-boundary cleanup of unused PyTorch pinned host allocations."""
from __future__ import annotations

import ctypes
import gc
import os
import time

import torch

from .metrics import DynamicOffloadMetrics


def _host_snapshot() -> dict:
    """Best-effort CPU observations; never query CUDA or change admission."""
    result = {}
    try:
        import psutil

        memory = psutil.Process().memory_info()
        result["process_rss_bytes"] = memory.rss
        if os.name == "nt":
            result["process_private_bytes"] = memory.private
    except Exception as error:
        result["process_observation_error"] = str(error)
    if os.name == "nt":
        try:
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                    (name, ctypes.c_uint64) for name in (
                        "total_phys", "available_phys", "total_commit", "available_commit",
                        "total_virtual", "available_virtual", "extended",
                    )
                ]
            function = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
            function.argtypes = [ctypes.POINTER(MemoryStatus)]
            function.restype = ctypes.c_int
            memory = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
            if not function(ctypes.byref(memory)):
                raise ctypes.WinError(ctypes.get_last_error())
            result.update(available_physical_bytes=memory.available_phys,
                          available_process_commit_bytes=memory.available_commit)
        except Exception as error:
            result["system_observation_error"] = str(error)
    return result


def _allocator_snapshot() -> dict:
    stats_function = getattr(torch.cuda, "host_memory_stats", None)
    if not callable(stats_function):
        return {"status": "unavailable"}
    try:
        stats = stats_function()
        allocated = stats.get("allocated_bytes.current")
        active = stats.get("active_bytes.current")
        valid_allocated = type(allocated) is int and allocated >= 0
        consistent = valid_allocated and type(active) is int and 0 <= active <= allocated
        return {"status": "available" if valid_allocated else "unavailable",
                "allocated_bytes": allocated if valid_allocated else None,
                "reported_active_bytes": active,
                "active_bytes": active if consistent else None,
                "active_counter_consistent": bool(consistent)}
    except Exception as error:
        return {"status": "unavailable", "error": str(error)}


def release_pytorch_host_cache(*, enabled: bool = False, phase: str = "", device=None) -> DynamicOffloadMetrics:
    """Release unused PyTorch host blocks at an explicitly opted-in phase boundary.

    Call after previous components are detached and their caller-owned references
    dropped. Completes device transfers once, never per denoise step. Does not
    initialize CUDA, clear native source storage, free live tensors or set budgets.
    Transfer completion and release errors propagate: failure is not success.
    """
    started = time.perf_counter()
    observation = {"phase": phase, "status": "disabled", "released": False}
    if enabled:
        if not torch.cuda.is_initialized():
            observation["status"] = "cuda_not_initialized"
        else:
            empty_cache = getattr(torch._C, "_accelerator_emptyHostCache", None)
            if not callable(empty_cache):
                observation["status"] = "unsupported"
            else:
                gc.collect()
                torch.cuda.synchronize(device)
                observation["before"] = _allocator_snapshot()
                observation["host_before"] = _host_snapshot()
                empty_cache()
                observation["after"] = _allocator_snapshot()
                observation["host_after"] = _host_snapshot()
                before = observation["before"].get("allocated_bytes")
                after = observation["after"].get("allocated_bytes")
                observation["allocator_bytes_delta"] = (
                    before - after if before is not None and after is not None else None
                )
                observation.update(status="released", released=True)
    observation["elapsed_seconds"] = time.perf_counter() - started
    return DynamicOffloadMetrics(route="pytorch_host_cache_release",
                                configuration={"enabled": bool(enabled)},
                                setup={"host_cache_release": observation})
