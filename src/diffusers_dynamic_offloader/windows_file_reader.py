"""Windows overlapped file reads into already pinned host memory.

This is a small DDO transport primitive for the optional VMM runtime.
It uses only documented Win32 I/O through ``ctypes``; CUDA ownership remains
with the cuda-python/PyTorch caller.  Non-Windows callers deliberately use the
ordinary Python file fallback.
"""

from __future__ import annotations

import ctypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


_CHUNK_BYTES = 2 * 1024**2
_WORKERS = 8


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", ctypes.c_uint32),
        ("OffsetHigh", ctypes.c_uint32),
        ("hEvent", ctypes.c_void_p),
    ]


class WindowsPinnedFileReader:
    """Concurrent positional reads into raw addresses of pinned CPU tensors."""

    def __init__(self, *, workers: int = _WORKERS, chunk_bytes: int = _CHUNK_BYTES) -> None:
        if os.name != "nt":
            raise RuntimeError("WindowsPinnedFileReader is only available on Windows")
        self.chunk_bytes = chunk_bytes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        self._kernel32.CreateFileW.restype = ctypes.c_void_p
        self._kernel32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(_OVERLAPPED)]
        self._kernel32.ReadFile.restype = ctypes.c_int
        self._kernel32.GetOverlappedResult.argtypes = [ctypes.c_void_p, ctypes.POINTER(_OVERLAPPED), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
        self._kernel32.GetOverlappedResult.restype = ctypes.c_int
        self._kernel32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
        self._kernel32.CreateEventW.restype = ctypes.c_void_p
        self._kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._kernel32.CloseHandle.restype = ctypes.c_int
        self._handles: dict[Path, int] = {}
        self._handles_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ddo-file-read")

    def _handle(self, path: Path) -> int:
        with self._handles_lock:
            existing = self._handles.get(path)
            if existing is not None:
                return existing
            # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS.
            handle = self._kernel32.CreateFileW(str(path), 0x80000000, 0x00000001, None, 3, 0x40000000 | 0x10000000, None)
            if handle in (-1, 0, 0xFFFFFFFFFFFFFFFF, None, ctypes.c_void_p(-1).value):
                raise ctypes.WinError(ctypes.get_last_error())
            self._handles[path] = int(handle)
            return int(handle)

    def _read_chunk(self, handle: int, offset: int, destination: int, size: int) -> None:
        event = self._kernel32.CreateEventW(None, True, False, None)
        if not event:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            overlapped = _OVERLAPPED(Offset=offset & 0xFFFFFFFF, OffsetHigh=offset >> 32, hEvent=event)
            ok = self._kernel32.ReadFile(handle, destination, size, None, ctypes.byref(overlapped))
            if not ok and ctypes.get_last_error() != 997:  # ERROR_IO_PENDING
                raise ctypes.WinError(ctypes.get_last_error())
            read = ctypes.c_uint32()
            if not self._kernel32.GetOverlappedResult(handle, ctypes.byref(overlapped), ctypes.byref(read), True):
                raise ctypes.WinError(ctypes.get_last_error())
            if read.value != size:
                raise OSError(f"Short overlapped read: expected {size}, got {read.value}")
        finally:
            self._kernel32.CloseHandle(event)

    def read_at(self, path: Path, offset: int, destination: int, size: int) -> None:
        handle = self._handle(path)
        futures = []
        for done in range(0, size, self.chunk_bytes):
            chunk = min(self.chunk_bytes, size - done)
            futures.append(self._executor.submit(self._read_chunk, handle, offset + done, destination + done, chunk))
        for future in futures:
            future.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        with self._handles_lock:
            for handle in self._handles.values():
                self._kernel32.CloseHandle(handle)
            self._handles.clear()
