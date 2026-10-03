"""Partial prefetch cancellation retry contracts, without CUDA calls."""
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator.model_arena_adapter import ModelArenaModuleAdapter, _ModuleBinding
from ddo_vmm_allocator.native import NativeVmmError


def make_adapter():
    released, waits = [], []
    store = SimpleNamespace(
        unpin=released.append,
        _upload_stream=SimpleNamespace(synchronize=lambda: waits.append(1)),
        _runtime_stats={"gpu_prefetch_cancelled_units": 0},
    )
    adapter = ModelArenaModuleAdapter(store)
    bindings = [object(), object()]
    state = _ModuleBinding("unit", object(), bindings, {}, prefetched=True)
    adapter._execution_units = [state]
    return adapter, state, released, waits


@pytest.mark.parametrize("error_type", [NativeVmmError, KeyboardInterrupt])
@pytest.mark.parametrize("failed_position", [0, 1])
def test_partial_cancel_retry_does_not_repeat_successful_unpins(error_type, failed_position):
    adapter, state, released, waits = make_adapter()
    def unpin(binding):
        if binding is state.tensors[failed_position]:
            raise error_type("cancel unpin failed")
        released.append(binding)
    adapter.store.unpin = unpin
    with pytest.raises(error_type, match="cancel unpin failed"):
        adapter._cancel_gpu_prefetches()
    assert state.prefetched
    assert state.prefetch_cancelled_bindings == failed_position
    assert released == state.tensors[:failed_position]
    assert adapter.store._runtime_stats["gpu_prefetch_cancelled_units"] == 0
    adapter.store.unpin = released.append
    adapter._cancel_gpu_prefetches()
    assert released == state.tensors
    assert waits == [1, 1]
    assert not state.prefetched and state.prefetch_cancelled_bindings == 0
    assert adapter.store._runtime_stats["gpu_prefetch_cancelled_units"] == 1
    adapter._cancel_gpu_prefetches()
    assert released == state.tensors and waits == [1, 1]


def test_failed_upload_fence_preserves_all_prefetched_owners():
    adapter, state, released, waits = make_adapter()
    def fail():
        raise NativeVmmError("upload wait failed")
    adapter.store._upload_stream.synchronize = fail
    with pytest.raises(NativeVmmError, match="upload wait failed"):
        adapter._cancel_gpu_prefetches()
    assert state.prefetched and state.prefetch_cancelled_bindings == 0
    assert released == []
    adapter.store._upload_stream.synchronize = lambda: waits.append(1)
    adapter._cancel_gpu_prefetches()
    assert released == state.tensors and not state.prefetched


@pytest.mark.parametrize("error_type", [NativeVmmError, KeyboardInterrupt])
def test_native_execution_lease_cancel_failure_retains_lease(error_type):
    adapter, state, released, waits = make_adapter()
    lease, cancelled = object(), []
    state.execution_buffer_prefetch = lease
    def fail(value):
        assert value is lease
        raise error_type("lease cancellation failed")
    adapter.store.execution_buffer = SimpleNamespace(cancel=fail)
    with pytest.raises(error_type, match="lease cancellation failed"):
        adapter._cancel_gpu_prefetches()
    assert state.prefetched and state.execution_buffer_prefetch is lease
    assert released == waits == []
    adapter.store.execution_buffer.cancel = cancelled.append
    adapter._cancel_gpu_prefetches()
    assert cancelled == [lease] and state.execution_buffer_prefetch is None
    assert not state.prefetched and released == waits == []


def test_later_cancel_failure_does_not_reopen_earlier_completed_unit():
    adapter, state, released, _waits = make_adapter()
    second = _ModuleBinding("next", object(), [object()], {}, prefetched=True)
    adapter._execution_units.append(second)
    def unpin(binding):
        if binding is second.tensors[0]:
            raise NativeVmmError("second unit failed")
        released.append(binding)
    adapter.store.unpin = unpin
    with pytest.raises(NativeVmmError, match="second unit failed"):
        adapter._cancel_gpu_prefetches()
    assert not state.prefetched and second.prefetched
    assert adapter.store._runtime_stats["gpu_prefetch_cancelled_units"] == 1
    adapter.store.unpin = released.append
    adapter._cancel_gpu_prefetches()
    assert released == state.tensors + second.tensors
    assert adapter.store._runtime_stats["gpu_prefetch_cancelled_units"] == 2
