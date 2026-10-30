"""Fake-event failure retention; no CUDA availability query or native calls."""
from types import SimpleNamespace

import pytest

from ddo_vmm_allocator.model_arena_adapter import ModelArenaModuleAdapter
from ddo_vmm_allocator.native import NativeVmmError


class Event:
    def __init__(self, failure=None, ready=False):
        self.failure, self.ready = failure, ready

    def query(self):
        if self.failure == "query":
            raise NativeVmmError("query failed")
        return self.ready

    def synchronize(self):
        if self.failure == "wait":
            raise NativeVmmError("wait failed")
        self.ready = True


def adapter_with_event(event):
    released, waits = [], []
    store = SimpleNamespace(_debug_faults=False, unpin=released.append,
                            record_capacity_event_wait=lambda: waits.append(1))
    adapter = ModelArenaModuleAdapter(store)
    bindings = [object(), object()]
    adapter._pending_unpins = [(event, bindings)]
    return adapter, bindings, released, waits


def retire(adapter, route):
    if route == "capacity":
        return adapter._retire_oldest_pending_unpin()
    return adapter._collect_temporary_lifetimes(wait=True)


@pytest.mark.parametrize("route", ["capacity", "collect"])
@pytest.mark.parametrize("failure", ["query", "wait"])
def test_event_failure_retains_pending_owners_and_allows_retry(route, failure):
    event = Event(failure)
    adapter, bindings, released, waits = adapter_with_event(event)
    with pytest.raises(NativeVmmError, match=failure + " failed"):
        retire(adapter, route)
    assert adapter._pending_unpins == [(event, bindings)]
    assert released == waits == []
    event.failure = None
    retire(adapter, route)
    assert adapter._pending_unpins == [] and released == bindings
    assert waits == ([1] if route == "capacity" else [])


@pytest.mark.parametrize("route", ["capacity", "collect"])
@pytest.mark.parametrize("error_type", [NativeVmmError, KeyboardInterrupt])
def test_partial_unpin_failure_retains_only_unreleased_bindings(route, error_type):
    event = Event(ready=True)
    adapter, bindings, released, waits = adapter_with_event(event)
    def unpin(binding):
        if binding is bindings[1]:
            raise error_type("unpin interrupted")
        released.append(binding)
    adapter.store.unpin = unpin
    with pytest.raises(error_type, match="unpin interrupted"):
        retire(adapter, route)
    assert adapter._pending_unpins == [(event, [bindings[1]])]
    assert released == [bindings[0]] and waits == []
    adapter.store.unpin = released.append
    retire(adapter, route)
    assert adapter._pending_unpins == [] and released == bindings


def test_nonblocking_collection_preserves_incomplete_event():
    event = Event(ready=False)
    adapter, bindings, released, waits = adapter_with_event(event)
    adapter._collect_temporary_lifetimes()
    assert adapter._pending_unpins == [(event, bindings)]
    assert released == waits == []


def test_collection_commits_prior_completed_entry_before_later_failure():
    event = Event(ready=True)
    adapter, bindings, released, _waits = adapter_with_event(event)
    failed = Event("query")
    remaining = [object()]
    adapter._pending_unpins.append((failed, remaining))
    with pytest.raises(NativeVmmError, match="query failed"):
        adapter._collect_temporary_lifetimes()
    assert released == bindings
    assert adapter._pending_unpins == [(failed, remaining)]
