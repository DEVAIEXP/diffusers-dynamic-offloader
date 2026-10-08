"""CPU ownership/geometry contracts; native views and completion are doubles."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ddo_vmm_allocator.execution_auxiliary import (
    ExecutionAuxiliaryAdapter, discover_execution_auxiliary_layout as _discover,
)
from ddo_vmm_allocator.model_arena_adapter import ModelArenaModuleAdapter, _layout_view_spans
from diffusers_dynamic_offloader import DynamicOffloadConfig, resolve_dynamic_offload_backend


def discover_execution_auxiliary_layout(root, pattern, registry):
    # Tiny ownership fixtures explicitly disable the existing small route.
    return _discover(root, pattern, registry, small_tensor_threshold_bytes=0)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.table = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.alias = self.table

    def forward(self, value):
        return self.projection(value) + self.table + self.alias


def prepared():
    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList([Block(), Block()])
    root.eval()
    registry = SimpleNamespace(lookup=lambda tensor: SimpleNamespace(nbytes=tensor.numel() * tensor.element_size()))
    return root, registry


def test_discovery_aliases_unique_and_alignment_matches_attachment():
    root, registry = prepared()
    with patch("torch.cuda.init", side_effect=AssertionError("CUDA must not initialize")):
        layout = discover_execution_auxiliary_layout(root, ".*", registry)
    assert len(layout[3]) == 2
    assert len(layout[0]) == 4
    assert layout[2] == 784  # Four aligned parameters, not six alias names.
    spans = _layout_view_spans(layout[0], layout[1], layout[3])
    assert spans[-1][0] + spans[-1][1] == layout[2]
    assert root.blocks[0].table is root.blocks[0].alias


@pytest.mark.parametrize("defect,match", [
    ("cross_unit", "one non-Linear"), ("external", "one non-Linear"),
    ("storage", "storage aliases"), ("missing_source", "immutable file"),
    ("noncontiguous", "contiguous"), ("training", "eval mode"),
])
def test_preflight_refuses_unsafe_inputs_without_mutation(defect, match):
    root, registry = prepared()
    if defect == "cross_unit":
        root.blocks[1].table = root.blocks[0].table
    elif defect == "external":
        root.outside = root.blocks[0].table
    elif defect == "storage":
        root.blocks[0].alias = torch.nn.Parameter(root.blocks[0].table.detach())
    elif defect == "missing_source":
        registry = None
    elif defect == "noncontiguous":
        root.blocks[0].table = torch.nn.Parameter(torch.ones(4, 4).t())
        root.blocks[0].alias = root.blocks[0].table
    elif defect == "training":
        root.train()
    original = tuple(root.parameters())
    with pytest.raises(ValueError, match=match):
        discover_execution_auxiliary_layout(root, ".*", registry)
    assert all(old is new for old, new in zip(original, root.parameters()))


def test_auxiliary_publication_keeps_dense_queue_and_event_retirement(monkeypatch):
    root, registry = prepared()
    expected = [block(torch.ones(1, 4)).detach() for block in root.blocks]
    layout = discover_execution_auxiliary_layout(root, ".*", registry)
    store = SimpleNamespace(
        virtual_bytes=layout[2], page_bytes=512, _file_slices=registry,
        _staging_pool=None, _native_shard_transport=SimpleNamespace(prepare_sources=lambda units: None),
        _host_registry=None, _streaming_enabled=False,
        _source_stats=dict(file_backed_bindings=0, file_backed_bytes=0,
                           cpu_source_bindings=0, cpu_source_bytes=0),
        view_setup=lambda spans: nullcontext(),
        stable_tensor=lambda offset, source: (source.detach().clone(), None),
    )
    adapter = ExecutionAuxiliaryAdapter(store, tuple(layout[3].values()))
    installed = []
    monkeypatch.setattr(ModelArenaModuleAdapter, "_register_execution_unit", lambda self, state: installed.append(state))
    # CPU doubles isolate publication; this is not native/CUDA evidence.
    monkeypatch.setattr(adapter, "_publish_auxiliary_view", lambda group, view, descriptor: group.publish(view))
    event = SimpleNamespace(query=lambda: True, synchronize=lambda: None)
    monkeypatch.setattr(adapter, "_completion_event", lambda: event)
    report = adapter.attach_linear_modules(root, ".*", _layout=layout)
    assert report.attached_tensors == 4
    assert len(installed) == 2
    assert installed[0].prefetch_next is installed[1]
    assert all(unit.guard_reentry for unit in installed)
    for index, block in enumerate(root.blocks):
        assert block.table is block.alias
        torch.testing.assert_close(block(torch.ones(1, 4)), expected[index])
        assert len(installed[index].tensors) == 2
    groups = tuple(layout[3].values())
    assert all(group._original is None for group in groups)
    assert all(not isinstance(unit.tensors[-1].source, torch.Tensor) for unit in installed)
    adapter.close()
    assert all(group.state == "retired" for group in groups)
    assert all(block.table is block.alias and block.table.device.type == "meta" for block in root.blocks)


def test_budget_default_and_protected_routes():
    assert DynamicOffloadConfig().vmm_execution_auxiliary_ownership is False
    for backend, preset in (("linear", "auto"), ("vmm", "nvme_baseline")):
        with pytest.raises(ValueError, match="exclusive"):
            resolve_dynamic_offload_backend(backend, preset).build_config(vmm_execution_auxiliary_ownership=True)
    config = resolve_dynamic_offload_backend("vmm", "auto").build_config(
        vmm_execution_auxiliary_ownership=True, vmm_resident_weight_budget_gb=2)
    assert config.vmm_resident_weight_budget_gb == 2


def test_production_publication_rejects_cpu_views():
    root, registry = prepared()
    layout = discover_execution_auxiliary_layout(root, ".*", registry)
    group = next(iter(layout[3].values()))
    adapter = ExecutionAuxiliaryAdapter(SimpleNamespace(), (group,))
    with pytest.raises(RuntimeError, match="CUDA native view"):
        adapter._publish_auxiliary_view(group, group.loader_tensor.detach(), registry.lookup(group.loader_tensor))
    assert group.state == "prepared"


def test_retirement_wait_failure_keeps_alias_ownership(monkeypatch):
    root, registry = prepared()
    layout = discover_execution_auxiliary_layout(root, ".*", registry)
    group = next(iter(layout[3].values()))
    group.publish(group.loader_tensor.detach().clone())
    adapter = ExecutionAuxiliaryAdapter(SimpleNamespace(), (group,))
    from ddo_vmm_allocator.model_arena_adapter import ModelArenaModuleAdapter
    monkeypatch.setattr(ModelArenaModuleAdapter, "close", lambda self: None)
    def refuse():
        raise RuntimeError("completion unproven")
    event = SimpleNamespace(query=lambda: True, synchronize=refuse)
    monkeypatch.setattr(adapter, "_completion_event", lambda: event)
    with pytest.raises(RuntimeError, match="completion unproven"):
        adapter.close()
    assert group.state == "published"
    assert root.blocks[0].table is root.blocks[0].alias
    event.synchronize = lambda: None
    adapter.close()
    assert group.state == "retired"


def test_preserves_existing_small_parameters_at_inclusive_threshold():
    root, registry = prepared()
    root.blocks[0].table = torch.nn.Parameter(torch.ones(4096))
    root.blocks[0].alias = root.blocks[0].table
    root.blocks[1].table = torch.nn.Parameter(torch.ones(4097))
    root.blocks[1].alias = root.blocks[1].table
    small = root.blocks[0].table
    lookups = []
    registry.lookup = lambda tensor: lookups.append(tensor.numel()) or SimpleNamespace()
    layout = _discover(root, ".*", registry)  # Existing default is 16 KiB.
    assert len(layout[3]) == 1
    assert next(iter(layout[3].values())).nbytes == 16388
    assert lookups == [4097]
    assert root.blocks[0].table is small and root.blocks[0].alias is small
    assert root.blocks[0].table.device.type == "cpu"
    assert all(path != 'blocks.0' for path, _ in layout[0])


@pytest.mark.parametrize('threshold', [-1, True, 1.5])
def test_invalid_existing_threshold_refuses_before_mutation(threshold):
    root, registry = prepared()
    original = root.blocks[0].table
    with pytest.raises(ValueError, match='threshold'):
        _discover(root, ".*", registry, small_tensor_threshold_bytes=threshold)
    assert root.blocks[0].table is original
