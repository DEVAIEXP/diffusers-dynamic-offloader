"""Approved-only missing-current-context rejection, ownership and retry gate."""
import os
from pathlib import Path

import pytest

from ddo_vmm_allocator.native import load_native_vmm
from ddo_vmm_allocator.source import NativeSourceError
from ddo_vmm_allocator.tensor_view import cuda_uint8_tensor_from_address
from vmm_test_helpers import detached_context_for_probe

pytestmark = pytest.mark.vmm
PAGE = 2 * 2**20


def test_missing_context_preserves_owners_then_upload_and_cleanup_recover(tmp_path):
    import torch
    from cuda.bindings import driver

    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    path = os.environ.get("DDO_VMM_LIBRARY_PATH")
    assert path, "select isolated lifecycle OFF library explicitly"
    lib = load_native_vmm(Path(path))
    assert not hasattr(lib._library, "ddo_vmm_source_test_failures")
    torch.empty(1, device="cuda:0")
    source_path = tmp_path / "context.bin"
    source_path.write_bytes(b"\x35" * PAGE)
    with lib.open_source_file(source_path) as source, lib.create_source_cache(
        slot_bytes=PAGE, slots=1, persistent_buffer=True,
        registered_limit_bytes=PAGE, process_limit_bytes=3 * PAGE,
    ) as cache, lib.create_model_arena(PAGE, page_size_bytes=PAGE) as arena:
        arena.prepare_view_range(0, PAGE)
        view, view_owner = cuda_uint8_tensor_from_address(
            arena.info.address, PAGE, owner=arena, backing="device")
        arena.release_view_range(0, PAGE)
        arena.fault_range(0, PAGE)
        ticket = None
        try:
            stream = torch.cuda.current_stream()
            ticket = cache.request(source, 0, PAGE)
            assert ticket is not None
            ticket.copy_to_arena_async(arena, stream=stream)
            original_ticket, original_cache = ticket._handle.value, cache._handle.value
            # DMA may still be live. Do not query Torch while detached, reset a
            # device, destroy a context, or inject a real driver fault.
            with detached_context_for_probe(driver):
                for operation in (ticket.close, cache.clear, cache.close):
                    with pytest.raises(NativeSourceError, match="owning CUDA context") as error:
                        operation()
                    assert error.value.status == 3
                    assert ticket._handle.value == original_ticket
                    assert cache._handle.value == original_cache
                    assert ticket._arena is arena
            ticket.close()
            assert ticket._handle.value is None and ticket._arena is None
            assert torch.all(view == 53).item()
            state = cache.snapshot()
            assert state["live_tickets"] == state["transfer_owned_events"] == 0
            reads = state["source_read_bytes"]
            assert reads == PAGE

            # A fresh ticket proves continued use after rejection, not merely
            # that a failed owner can be destroyed. Native bytes are reused.
            ticket = cache.request(source, 0, PAGE)
            assert ticket is not None
            view.zero_()
            ticket.copy_to_arena_async(arena, stream=stream)
            ticket.close()
            assert torch.all(view == 53).item()
            state = cache.snapshot()
            assert state["source_read_bytes"] == reads
            assert state["live_tickets"] == state["transfer_owned_events"] == 0
            cache.clear()
            assert cache.snapshot()["committed_bytes"] == cache.snapshot()["registered_bytes"] == 0
        finally:
            if ticket is not None:
                ticket.close()
            arena.unpin_range(0, PAGE)
            del view, view_owner
    source_path.unlink()
