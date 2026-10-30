"""Upload plus fence failure diagnostics; fake backends only, no CUDA calls."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ddo_vmm_allocator.model_arena_adapter import NativeExecutionBuffer, NativeExecutionBuffers
from ddo_vmm_allocator.native import NativeVmmError


@pytest.mark.parametrize("kind", ["buffer", "pool"])
@pytest.mark.parametrize("original_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_type", [NativeVmmError, KeyboardInterrupt])
def test_upload_keeps_first_error_and_refuses_unsafe_close(monkeypatch, kind, original_type, cleanup_type):
    original = original_type("original upload failure")
    cleanup = cleanup_type("fence record failure")
    calls = []
    def upload(*args):
        calls.append("upload")
        raise original
    store = SimpleNamespace(_upload_stream=object(), _native_shard_transport=SimpleNamespace(upload_batch=upload))
    if kind == "buffer":
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
        buffer = NativeExecutionBuffer.__new__(NativeExecutionBuffer)
        buffer.store, buffer._materialized, buffer._unsafe_fence = store, True, False
        buffer._views = [object()]
        def record(stream):
            calls.append("fence")
            raise cleanup
        buffer.native = SimpleNamespace(wait_compute=lambda stream: None, record_compute=record)
    else:
        buffer = NativeExecutionBuffers.__new__(NativeExecutionBuffers)
        buffer.store = store
        arena = object()
        buffer._layouts = {(0, ()): ([object()], [])}
        state = {"unsafe": False}
        def release(slot, stream):
            calls.append("fence")
            state["unsafe"] = True
            raise cleanup
        def close():
            assert state["unsafe"]
            raise NativeVmmError("completion is unproven")
        buffer.pool = SimpleNamespace(acquire=lambda stream: (0, arena), release=release, close=close)
    with pytest.raises(original_type, match="original upload failure") as caught:
        buffer.upload([])
    assert caught.value is original
    assert calls == ["upload", "fence"]
    assert any("fence record failure" in note for note in original.__notes__)
    with pytest.raises(NativeVmmError, match="completion is unproven"):
        buffer.close()
    if kind == "buffer":
        assert buffer._unsafe_fence and buffer._views
    else:
        assert buffer._layouts


def test_successful_upload_still_raises_fence_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    buffer = NativeExecutionBuffer.__new__(NativeExecutionBuffer)
    buffer._materialized, buffer._unsafe_fence, buffer._views = True, False, [object()]
    buffer.store = SimpleNamespace(_upload_stream=object(), _native_shard_transport=SimpleNamespace(upload_batch=lambda *args: object()))
    failure = NativeVmmError("fence record failure")
    def record(stream):
        raise failure
    buffer.native = SimpleNamespace(wait_compute=lambda stream: None, record_compute=record)
    with pytest.raises(NativeVmmError) as caught:
        buffer.upload([])
    assert caught.value is failure and buffer._unsafe_fence
