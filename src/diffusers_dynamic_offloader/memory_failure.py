"""Failure-only host diagnostics; no CUDA calls, admission or recovery policy."""
from __future__ import annotations

import ctypes
import os


def _windows_commit_snapshot():
    if os.name != "nt":
        return {"status": "unavailable", "reason": "Windows commit observation only"}
    class PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32)] + [
            (name, ctypes.c_size_t) for name in (
                "commit", "limit", "peak", "physical", "available", "cache",
                "kernel", "paged", "nonpaged", "page_size")
        ] + [(name, ctypes.c_uint32) for name in ("handles", "processes", "threads")]
    info = PerformanceInfo(cb=ctypes.sizeof(PerformanceInfo))
    query = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    query.argtypes = [ctypes.POINTER(PerformanceInfo), ctypes.c_uint32]
    query.restype = ctypes.c_int
    if not query(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if not info.page_size or not info.limit or info.commit > info.limit:
        raise ValueError("inconsistent system commit observation")
    return {"status": "observed", "commit_bytes": info.commit * info.page_size,
            "limit_bytes": info.limit * info.page_size,
            "available_bytes": (info.limit - info.commit) * info.page_size,
            "physical_available_bytes": info.available * info.page_size}


def diagnose_memory_failure(error: BaseException, *, observation=None) -> dict:
    """Explain host exhaustion without relabeling an arbitrary CUDA fault as OOM.

    Optional detached observations support offline analysis. The 256 MiB low
    headroom label is diagnostic only: it never stops work or changes budgets.
    A failure-time query cannot recover a previous minimum or prove causality.
    """
    chain = []
    current = error
    while current is not None and len(chain) < 16 and all(current is not item for item in chain):
        chain.append(current)
        current = current.__cause__ if current.__cause__ is not None else (
            None if current.__suppress_context__ else current.__context__)
    explicit = any(getattr(item, "winerror", None) == 1455 for item in chain)
    host_oom = explicit or any(isinstance(item, MemoryError) for item in chain)
    try:
        snapshot = dict(_windows_commit_snapshot() if observation is None else observation)
        available = snapshot.get("available_bytes")
        if snapshot.get("status") == "observed" and (type(available) is not int or available < 0):
            snapshot = {"status": "unavailable", "reason": "invalid commit headroom"}
        low = (snapshot.get("status") == "observed" and type(available) is int
               and 0 <= available <= 256 * 1024**2)
    except Exception as query_error:
        snapshot = {"status": "unavailable", "reason": str(query_error)}
        low = False
    message = None
    classification = "unclassified"
    if explicit:
        classification = "windows_commit_allocation_failed"
        message = "Windows refused a memory allocation because the system commit limit was reached."
    elif host_oom:
        classification = "host_allocation_failed"
        message = "A host memory allocation failed; the specific memory limit is not established."
    elif low:
        classification = "failure_with_low_system_commit_headroom"
        message = ("The operation failed while Windows system commit headroom was critically low. "
                   "This observation does not prove the cause of the original error.")
    if message is not None:
        if snapshot.get("status") == "observed":
            message += (f" Remaining commit: {snapshot['available_bytes'] / 1024**2:.2f} MiB; "
                        "available physical RAM is a separate metric.")
        message += (" Close memory-intensive applications or use a smaller workload before a new run. "
                    "The original error is preserved; no automatic retry, CUDA reset or unsafe cleanup is performed.")
    return {"schema": 1, "classification": classification, "message": message,
            "host": snapshot, "scope": "failure-time observation; no causal attribution"}
