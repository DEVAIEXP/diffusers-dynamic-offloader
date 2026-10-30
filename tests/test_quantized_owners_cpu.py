"""Optional CPU ownership state-machine tests, not CUDA event acceptance."""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization import Float8WeightOnlyConfig, quantize_
from ddo_vmm_allocator.quantized_owners import (
    _capture_float8_component_owners_cpu, capture_float8_weight_owners_cpu,
)


class Completion:
    done = False
    def query(self):
        return self.done


@pytest.fixture(autouse=True)
def no_cuda():
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        yield
    assert not torch.cuda.is_initialized()


def setup_owner():
    root = torch.nn.Sequential(torch.nn.Linear(32, 16, bias=False, dtype=torch.bfloat16),
                               torch.nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)).eval()
    root.requires_grad_(False)
    quantize_(root, Float8WeightOnlyConfig(set_inductor_config=False))
    root[1].weight = root[0].weight
    group = capture_float8_weight_owners_cpu(root, root[0].weight)
    payloads = {"qdata": root[0].weight.qdata.clone(), "scale": root[0].weight.scale.clone()}
    return root, group, payloads


def test_aliases_backend_dispatch_and_final_event_retirement():
    root, group, payloads = setup_owner()
    forward = type(root[0]).forward
    old = group.loader_tensor
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    expected = root[0](inputs)
    published = group.publish_payload_views(payloads)
    assert type(root[0]).forward is forward
    assert root[0].weight is root[1].weight is published
    assert published.qdata.data_ptr() == payloads["qdata"].data_ptr()
    assert published.scale.data_ptr() == payloads["scale"].data_ptr()
    assert group._original is None and group.snapshot()["bytes"] == 576
    torch.testing.assert_close(root[0](inputs), expected, rtol=0, atol=0)
    event = Completion()
    assert not group.retire(event)
    assert root[0].weight is published and group._published is published
    with pytest.raises(RuntimeError, match="replace"):
        group.retire(Completion())
    event.done = True
    assert group.retire(event)
    assert root[0].weight is root[1].weight and root[0].weight.device.type == "meta"
    assert type(root[0].weight) is type(old)
    assert root[0].weight.qdata.device.type == "meta" and root[0].weight.scale.device.type == "meta"
    assert group._published is None


def test_missing_payload_and_changed_owner_refuse_before_publication():
    root, group, payloads = setup_owner()
    old = root[0].weight
    with pytest.raises(ValueError):
        group.publish_payload_views({"qdata": payloads["qdata"]})
    assert root[0].weight is old and group.state == "prepared"
    root[1].weight = torch.nn.Parameter(torch.zeros(16, 32, dtype=torch.bfloat16), requires_grad=False)
    with pytest.raises(RuntimeError, match="owner changed"):
        group.publish_payload_views(payloads)
    assert root[0].weight is old


def test_failed_publication_retains_both_sides_for_rollback():
    root, group, payloads = setup_owner()
    old = root[0].weight
    class FailingTable(dict):
        fail = True
        def __setitem__(self, key, value):
            if self.fail:
                raise RuntimeError("assignment failed")
            return super().__setitem__(key, value)
    table = FailingTable(root[1]._parameters)
    root[1]._parameters = table
    with pytest.raises(RuntimeError, match="assignment failed"):
        group.publish_payload_views(payloads)
    assert group.state == "publication_failed" and group._original is old
    assert group._published is not None and root[0].weight is group._published
    table.fail = False
    group.rollback_failed_publication()
    assert root[0].weight is root[1].weight is old and group.state == "prepared"


def test_failed_event_query_retains_views_and_same_event():
    root, group, payloads = setup_owner()
    published = group.publish_payload_views(payloads)
    event = Completion()
    with patch.object(event, "query", side_effect=RuntimeError("event failure")):
        with pytest.raises(RuntimeError, match="event failure"):
            group.retire(event)
    assert group._completion is event and root[0].weight is published
    event.done = True
    assert group.retire(event)


def test_distinct_wrapper_shared_storage_refused():
    root, _group, _payloads = setup_owner()
    root[1].weight = torch.nn.Parameter(root[0].weight.detach(), requires_grad=False)
    with pytest.raises(ValueError, match="share composite"):
        capture_float8_weight_owners_cpu(root, root[0].weight)


def test_bulk_discovery_traverses_once_and_preserves_aliases():
    root, reference, _payloads = setup_owner()
    root.again = root[0]
    root.register_buffer("constant", torch.ones(2))
    with patch.object(root, "named_modules", wraps=root.named_modules) as traversal:
        groups = _capture_float8_component_owners_cpu(root)
    assert traversal.call_count == 1
    actual = groups[id(root[0].weight)]
    assert actual.weight_spec == reference.weight_spec
    assert len(actual.owners) == 2
    assert set(actual.owners[0][2]) == {"0.weight", "again.weight"}
    assert actual.owners[1][2] == ("1.weight",)
    assert groups[id(root.constant)].kind == "buffer"


@pytest.mark.parametrize("alias", ["wrapper", "parameter", "buffer"])
def test_bulk_discovery_refuses_distinct_payload_storage_aliases(alias):
    root, _group, _payloads = setup_owner()
    if alias == "wrapper":
        root[1].weight = torch.nn.Parameter(root[0].weight.detach(), requires_grad=False)
    elif alias == "parameter":
        root.alias = torch.nn.Parameter(root[0].weight.qdata, requires_grad=False)
    else:
        root.register_buffer("alias", root[0].weight.scale)
    original = root[0].weight
    with pytest.raises(ValueError, match="share composite"):
        _capture_float8_component_owners_cpu(root)
    assert root[0].weight is original
