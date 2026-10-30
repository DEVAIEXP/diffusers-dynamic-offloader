"""Small direct allocator ordering gate; execute only after CUDA approval."""
import ctypes as c
import os
import sys

import pytest


@pytest.mark.vmm
@pytest.mark.parametrize("route", ["same_stream", "cross_stream", "legacy_stream"])
@pytest.mark.parametrize("cache_mib", [4, 0])
def test_cached_native_backing_preserves_last_writer_order(monkeypatch, route, cache_mib):
    if sys.platform != "win32" or not os.environ.get("DDO_VMM_LIBRARY_PATH"):
        pytest.skip("requires Windows and an explicitly selected native DLL")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("DDO_VMM_TORCH_CACHE_MIB", str(cache_mib))
    # These are C ABI allocator entry points, not a global PyTorch allocator
    # replacement. Only one small native allocation is leased at a time.
    dll = c.CDLL(os.environ["DDO_VMM_LIBRARY_PATH"])
    allocate = dll.ddo_vmm_torch_malloc
    allocate.argtypes = [c.c_size_t, c.c_int, c.c_void_p]
    allocate.restype = c.c_void_p
    release = dll.ddo_vmm_torch_free
    release.argtypes = [c.c_void_p, c.c_size_t, c.c_int, c.c_void_p]
    release.restype = None
    driver = c.WinDLL("nvcuda.dll")
    fill = driver.cuMemsetD32Async
    fill.argtypes = [c.c_uint64, c.c_uint, c.c_size_t, c.c_void_p]
    fill.restype = c.c_int
    read = driver.cuMemcpyDtoH_v2
    read.argtypes = [c.c_void_p, c.c_uint64, c.c_size_t]
    read.restype = c.c_int
    device = torch.cuda.current_device()
    producer = torch.cuda.default_stream(device) if route == "legacy_stream" else torch.cuda.Stream(device)
    consumer = torch.cuda.Stream(device) if route == "cross_stream" else producer
    size = 256
    pointer = None
    stream = producer
    completion = torch.cuda.Event()
    try:
        pointer = allocate(size, device, producer.cuda_stream)
        assert pointer
        with torch.cuda.stream(producer):
            # Keep the old writer in flight while native free caches its fence.
            torch.cuda._sleep(250_000_000)
            assert fill(pointer, 3, size // 4, producer.cuda_stream) == 0
            completion.record(producer)
        original = pointer
        release(pointer, size, device, producer.cuda_stream)
        pointer = None
        assert not completion.query(), "free waited for in-flight work during cache trimming"
        stream = consumer
        pointer = allocate(size, device, consumer.cuda_stream)
        assert pointer == original, "contract must exercise actual cache reuse"
        assert fill(pointer, 7, size // 4, consumer.cuda_stream) == 0
        consumer.synchronize()
        producer.synchronize()
        result = (c.c_uint * (size // 4))()
        assert read(result, pointer, size) == 0
        assert list(result) == [7] * (size // 4)
    finally:
        # Test boundaries only; no borrowed pointer is released with live work.
        producer.synchronize()
        consumer.synchronize()
        if pointer:
            release(pointer, size, device, stream.cuda_stream)
