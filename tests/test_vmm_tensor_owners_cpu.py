import gc
import weakref
from unittest.mock import patch

import pytest
import torch

from ddo_vmm_allocator import capture_model_tensor_owners


class Event:
    def __init__(self, ready=False):
        self.ready = ready
        self.error = None

    def query(self):
        if self.error:
            raise self.error
        return self.ready


def shared_model():
    root = torch.nn.Module()
    root.embedding = torch.nn.Embedding(5, 3)
    root.head = torch.nn.Linear(3, 5, bias=False)
    root.head.weight = root.embedding.weight
    root.register_buffer("scale", torch.ones(1), persistent=False)
    root.embedding.register_buffer("scale_alias", root.scale, persistent=False)
    return root


def test_capture_and_publication_preserve_all_parameter_and_buffer_aliases():
    root = shared_model()
    groups = capture_model_tensor_owners(root)
    assert len(groups) == 2
    parameter, buffer = groups
    original = weakref.ref(root.head.weight)
    view = root.head.weight.detach().clone()
    with patch("torch.Tensor.to", side_effect=AssertionError("copy")), \
         patch("torch.Tensor.data_ptr", side_effect=AssertionError("pointer")), \
         patch("torch.cuda.init", side_effect=AssertionError("CUDA")):
        published = parameter.publish(view)
        published_buffer = buffer.publish(torch.full((1,), 2.0))
    assert root.head.weight is root.embedding.weight is published
    assert root.scale is root.embedding.scale_alias is published_buffer
    assert "scale" in root._non_persistent_buffers_set
    assert "scale_alias" in root.embedding._non_persistent_buffers_set
    gc.collect()
    assert original() is None
    with pytest.raises(RuntimeError, match="before publication"):
        _ = parameter.loader_tensor
    ids = torch.tensor([0, 4])
    expected = torch.nn.functional.linear(torch.nn.functional.embedding(ids, view), view)
    actual = root.head(root.embedding(ids))
    torch.testing.assert_close(actual, expected)


def test_pending_or_failed_event_retains_aliases_and_event_until_retry():
    root = shared_model()
    group = capture_model_tensor_owners(root)[0]
    published = group.publish(root.head.weight.detach().clone())
    event = Event()
    assert group.retire(event) is False
    assert root.head.weight is root.embedding.weight is published
    with pytest.raises(RuntimeError, match="replace a retained"):
        group.retire(Event(True))
    event.error = RuntimeError("query failed")
    with pytest.raises(RuntimeError, match="query failed"):
        group.retire(event)
    assert group._published is published
    event.error, event.ready = None, True
    assert group.retire(event)
    assert group.state == "retired"
    assert root.head.weight is root.embedding.weight
    assert root.head.weight.device.type == "meta"
    assert group._published is None
    assert group.retire(event)  # Idempotent; no new event query.


@pytest.mark.parametrize("kind_index", [0, 1])
def test_retirement_requires_completion(kind_index):
    root = shared_model()
    group = capture_model_tensor_owners(root)[kind_index]
    group.publish(group.loader_tensor.detach().clone())
    with pytest.raises(ValueError, match="completion event"):
        group.retire(None)
    assert group.state == "published"


@pytest.mark.parametrize("change", ["shape", "dtype", "meta", "noncontiguous"])
def test_invalid_view_does_not_change_module(change):
    root = shared_model()
    group = capture_model_tensor_owners(root)[0]
    original = root.head.weight
    view = torch.ones(5, 3)
    if change == "shape":
        view = view.reshape(3, 5)
    elif change == "dtype":
        view = view.double()
    elif change == "meta":
        view = torch.empty(5, 3, device="meta")
    else:
        view = torch.ones(3, 5).t()
    with pytest.raises(ValueError):
        group.publish(view)
    assert root.head.weight is root.embedding.weight is original
    assert group.state == "prepared"


def test_changed_owner_is_not_silently_overwritten():
    root = shared_model()
    group = capture_model_tensor_owners(root)[0]
    root.head.weight = torch.nn.Parameter(torch.ones(5, 3))
    with pytest.raises(RuntimeError, match="changed before publication"):
        group.publish(torch.ones(5, 3))


def test_external_change_blocks_retirement_and_retains_published_view():
    root = shared_model()
    group = capture_model_tensor_owners(root)[0]
    published = group.publish(torch.ones(5, 3))
    root.head.weight = torch.nn.Parameter(torch.zeros(5, 3))
    with pytest.raises(RuntimeError, match="changed before retirement"):
        group.retire(Event(True))
    assert group._published is published


def test_shared_module_paths_do_not_duplicate_physical_owner_assignments():
    root = torch.nn.Module()
    root.one = torch.nn.Linear(2, 2, bias=False)
    root.two = root.one
    group, = capture_model_tensor_owners(root)
    assert group.snapshot()["owner_count"] == 1
    assert group.snapshot()["names"] == ["one.weight", "two.weight"]


def test_meta_capture_does_not_access_payload_or_storage():
    with torch.device("meta"):
        root = shared_model()
    with patch("torch.Tensor.data_ptr", side_effect=AssertionError("pointer")), \
         patch("torch.Tensor.to", side_effect=AssertionError("copy")):
        groups = capture_model_tensor_owners(root)
    assert [group.nbytes for group in groups] == [60, 4]


def test_nonpersistent_buffer_retirement_preserves_alias_and_registration_kind():
    root = shared_model()
    group = capture_model_tensor_owners(root)[1]
    group.publish(torch.zeros(1))
    assert group.retire(Event(True))
    assert root.scale is root.embedding.scale_alias
    assert root.scale.device.type == "meta"
    assert "scale" not in root.state_dict()
    assert "embedding.scale_alias" not in root.state_dict()


def test_deleted_owner_is_not_treated_as_an_uncreated_placeholder():
    root = shared_model()
    group = capture_model_tensor_owners(root)[0]
    published = group.publish(torch.ones(5, 3))
    root.head._parameters.pop("weight")
    with pytest.raises(RuntimeError, match="changed before retirement"):
        group.retire(Event(True))
    assert group._published is published


def test_parametrized_modules_are_refused_without_mutation():
    root = torch.nn.Linear(3, 2, bias=False)
    torch.nn.utils.parametrize.register_parametrization(root, "weight", torch.nn.Identity())
    original = root.parametrizations.weight.original
    with pytest.raises(TypeError, match="Parametrized"):
        capture_model_tensor_owners(root)
    assert root.parametrizations.weight.original is original


class FailingTable(dict):
    fail = True

    def __setitem__(self, key, value):
        if self.fail:
            raise RuntimeError("assignment failed")
        return super().__setitem__(key, value)


def test_partial_publication_retains_original_and_can_roll_back():
    root = shared_model()
    group = capture_model_tensor_owners(root)[0]
    original = root.head.weight
    root.head._parameters = FailingTable(root.head._parameters)
    with pytest.raises(RuntimeError, match="assignment failed"):
        group.publish(torch.ones(5, 3))
    assert group.state == "publication_failed"
    assert group._original is original
    root.head._parameters.fail = False
    group.rollback_failed_publication()
    assert root.head.weight is root.embedding.weight is original
    assert group.state == "prepared"
    group.publish(torch.ones(5, 3))
    with pytest.raises(RuntimeError, match="Only failed publication"):
        group.rollback_failed_publication()


def test_partial_retirement_retains_view_and_allows_same_event_retry():
    root = shared_model()
    group = capture_model_tensor_owners(root)[0]
    published = group.publish(torch.ones(5, 3))
    event = Event(True)
    root.head._parameters = FailingTable(root.head._parameters)
    with pytest.raises(RuntimeError, match="assignment failed"):
        group.retire(event)
    assert group.state == "retiring"
    assert group._published is published
    assert root.embedding.weight.device.type == "meta"
    root.head._parameters.fail = False
    assert group.retire(event)
    assert root.head.weight is root.embedding.weight
