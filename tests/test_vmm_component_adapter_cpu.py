from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import gc
import weakref

import pytest
import torch
from safetensors.torch import save_file

from ddo_vmm_allocator.component_arena_adapter import ComponentArenaAdapter, discover_component_layout
from ddo_vmm_allocator.model_arena_adapter import (
    ModelArenaError, ModelArenaModuleAdapter, _ModuleBinding,
    _component_output_borrows_weights, attach_model_linear_modules,
)
from diffusers_dynamic_offloader import add_dynamic_offload_arguments, resolve_dynamic_offload_backend
from diffusers_dynamic_offloader import DynamicOffloadApplyResult, DynamicOffloadConfig, DynamicOffloadHook
from diffusers_dynamic_offloader.dynamic_offload import _attach_safetensor_file_slices


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 3)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3)) for _ in range(2)])
        self.head = torch.nn.Linear(3, 5, bias=False)
        self.head.weight = self.embedding.weight
        self.register_buffer("scale", torch.ones(1), persistent=False)

    def forward(self, ids):
        value = self.embedding(ids)
        for block in self.blocks:
            value = block(value)
        return self.head(value) * self.scale


def prepared(tmp_path):
    root = Model().eval()
    path = tmp_path / "weights.safetensors"
    save_file({name: value.detach() for name, value in root.named_parameters()}, str(path))
    _attach_safetensor_file_slices(root, [path])
    layout = discover_component_layout(root, root._ddo_safetensor_slices, buffer_limit_bytes=1024)
    return root, layout


def fake_adapter(layout, monkeypatch):
    store = SimpleNamespace(
        view_setup=lambda spans: nullcontext(), virtual_bytes=layout.payload_bytes, page_bytes=512,
        _streaming_enabled=False, _native_shard_transport=SimpleNamespace(prepare_sources=lambda units: None),
        _source_stats={"file_backed_bindings": 0, "file_backed_bytes": 0},
    )
    adapter = ComponentArenaAdapter(store)
    # No simulated CUDA claim: these fakes replace view/copy/event boundaries.
    monkeypatch.setattr(adapter, "_native_view", lambda group, source, offset: (source.detach().clone(), None))
    monkeypatch.setattr(adapter, "_move_buffer", lambda source: source.clone())
    installed = []
    monkeypatch.setattr(adapter, "_register_execution_unit", installed.append)
    event = SimpleNamespace(query=lambda: True, synchronize=lambda: None)
    monkeypatch.setattr(adapter, "_completion_event", lambda: event)
    return adapter, installed


def test_generic_layout_includes_norms_and_shared_native_range_once(tmp_path):
    root, layout = prepared(tmp_path)
    assert len(layout.parameters) == 9
    assert [name for name, _, _ in layout.units] == ["embedding", "head", "blocks.0", "blocks.1"]
    assert layout.units[0][2] == layout.units[1][2]
    assert len(layout.units[2][2]) == 4  # Linear and normalization.
    assert layout.buffer_bytes == 4


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_component_convolutions_preserve_decode_method_and_retire_owners(tmp_path, monkeypatch, dimensions):
    """CPU publication contract, not CUDA convolution or memory acceptance."""
    convolution = getattr(torch.nn, f"Conv{dimensions}d")

    class DecoderComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = convolution(2, 2, 1)
            self.blocks = torch.nn.ModuleList([
                torch.nn.Sequential(convolution(2, 2, 3, padding=1), torch.nn.GroupNorm(1, 2))
            ])

        def forward(self, value):
            raise AssertionError("decode must not call the component's forward")

        def decode(self, value):
            value = self.projection(value)
            for block in self.blocks:
                value = block(value)
            return value

    root = DecoderComponent().eval()
    sample = torch.randn((1, 2) + (4,) * dimensions)
    expected = root.decode(sample).detach()
    original_forward = type(root.projection).forward
    source = weakref.ref(root.projection.weight)
    path = tmp_path / "convolutions.safetensors"
    save_file({name: value.detach() for name, value in root.named_parameters()}, str(path))
    _attach_safetensor_file_slices(root, [path])
    layout = discover_component_layout(root, root._ddo_safetensor_slices, buffer_limit_bytes=0)
    assert [name for name, _, _ in layout.units] == ["projection", "blocks.0"]
    assert len(layout.parameters) == 6
    adapter, installed = fake_adapter(layout, monkeypatch)
    with patch("torch.cuda.init", side_effect=AssertionError("CUDA")):
        report = adapter.attach_component(layout)
        torch.testing.assert_close(root.decode(sample), expected, rtol=0, atol=0)
    gc.collect()
    assert source() is None
    assert type(root.projection).forward is original_forward
    assert report.attached_tensors == 6
    assert len(installed) == 2
    assert all(not isinstance(binding.source, torch.Tensor) for unit in installed for binding in unit.tensors)
    adapter.close()
    assert all(parameter.device.type == "meta" for parameter in root.parameters())
    root._ddo_safetensor_slices.close()


def test_publication_connects_one_binding_to_multiple_units_without_cpu_loader_retention(tmp_path, monkeypatch):
    root, layout = prepared(tmp_path)
    expected = root(torch.tensor([0, 4])).detach()
    old = weakref.ref(root.embedding.weight)
    adapter, installed = fake_adapter(layout, monkeypatch)
    with patch("torch.cuda.init", side_effect=AssertionError("CUDA")):
        report = adapter.attach_component(layout)
    gc.collect()
    assert old() is None
    assert root.embedding.weight is root.head.weight
    assert installed[0].tensors[0] is installed[1].tensors[0]
    assert all(unit.guard_reentry and unit.prefetch_next is None for unit in installed)
    assert report.attached_tensors == 9
    assert adapter.store._source_stats["file_backed_bindings"] == 9
    assert all(not isinstance(binding.source, torch.Tensor) for unit in installed for binding in unit.tensors)
    torch.testing.assert_close(root(torch.tensor([0, 4])), expected)
    adapter.close()
    assert root.embedding.weight is root.head.weight
    assert all(parameter.device.type == "meta" for parameter in root.parameters())
    assert root.scale.device.type == "meta"


def test_buffer_ceiling_and_missing_sources_fail_before_mutation(tmp_path):
    root, _ = prepared(tmp_path)
    original = root.embedding.weight
    with pytest.raises(ValueError, match="buffers exceed"):
        discover_component_layout(root, root._ddo_safetensor_slices, buffer_limit_bytes=0)
    with pytest.raises(ValueError, match="no immutable file source"):
        discover_component_layout(root, None, buffer_limit_bytes=1024)
    assert root.embedding.weight is root.head.weight is original


def test_overlapping_parameter_owner_units_are_refused(tmp_path):
    root, _ = prepared(tmp_path)
    root.register_parameter("direct", torch.nn.Parameter(torch.ones(2)))
    path = tmp_path / "overlap.safetensors"
    save_file({name: value.detach() for name, value in root.named_parameters()}, str(path))
    root._ddo_safetensor_slices.close()
    del root._ddo_safetensor_slices
    _attach_safetensor_file_slices(root, [path])
    with pytest.raises(ValueError, match="overlapping"):
        discover_component_layout(root, root._ddo_safetensor_slices, buffer_limit_bytes=1024)


def test_parameter_parent_encloses_child_units_without_architecture_conditions(tmp_path):
    root = torch.nn.Module()
    root.projector = torch.nn.Module()
    root.projector.register_parameter("projection", torch.nn.Parameter(torch.ones(3, 3)))
    root.projector.norm = torch.nn.LayerNorm(3)
    root.eval()
    path = tmp_path / "parent.safetensors"
    save_file({name: value.detach() for name, value in root.named_parameters()}, str(path))
    _attach_safetensor_file_slices(root, [path])
    layout = discover_component_layout(root, root._ddo_safetensor_slices, buffer_limit_bytes=0)
    assert len(layout.parameters) == 3
    assert len(layout.units) == 1
    name, module, indices = layout.units[0]
    assert name == "projector" and module is root.projector
    assert set(indices) == {0, 1, 2}


def test_retirement_failure_retains_view_owners_and_retries(tmp_path, monkeypatch):
    root, layout = prepared(tmp_path)
    adapter, _ = fake_adapter(layout, monkeypatch)
    adapter.attach_component(layout)
    published = root.head.weight
    event = SimpleNamespace(query=lambda: False, synchronize=lambda: None)
    monkeypatch.setattr(adapter, "_completion_event", lambda: event)
    with pytest.raises(ModelArenaError, match="completion is unproven"):
        adapter.close()
    assert root.head.weight is root.embedding.weight is published
    assert adapter._ownership_groups
    event.query = lambda: True
    adapter.close()
    assert not adapter._ownership_groups


def test_active_execution_unit_blocks_close_without_removing_ownership(tmp_path, monkeypatch):
    root, layout = prepared(tmp_path)
    adapter, installed = fake_adapter(layout, monkeypatch)
    adapter.attach_component(layout)
    installed[0].active = True
    with pytest.raises(ModelArenaError, match="active execution unit"):
        adapter.close()
    assert adapter._ownership_groups
    assert root.embedding.weight.device.type == "cpu"  # CPU fake, not CUDA.
    installed[0].active = False
    adapter.close()


def test_component_dispatch_does_not_enter_legacy_discovery(monkeypatch):
    import ddo_vmm_allocator.component_arena_adapter as component
    calls = []
    monkeypatch.setattr(component, "attach_component_parameters", lambda *args, **kwargs: calls.append(kwargs) or ("report", "store"))
    assert attach_model_linear_modules(torch.nn.Linear(2, 2), None, module_pattern=".*",
                                      resident_budget_bytes=1024, component_tensor_ownership=True) == ("report", "store")
    assert calls[0]["resident_budget_bytes"] == 1024
    with pytest.raises(ValueError, match="full-component selection"):
        attach_model_linear_modules(None, None, module_pattern="blocks", resident_budget_bytes=1024,
                                    component_tensor_ownership=True)


def test_public_arguments_remain_opt_in_and_protected_routes_reject_ownership():
    parser = add_dynamic_offload_arguments(argparse.ArgumentParser())
    default = resolve_dynamic_offload_backend().build_config_from_args(parser.parse_args([]))
    assert not default.vmm_component_tensor_ownership
    args = parser.parse_args(["--ddo-backend", "vmm", "--ddo-vmm-component-tensor-ownership",
                              "--ddo-vmm-native-transport", "--ddo-vmm-component-buffer-limit-mb", "1"])
    config = resolve_dynamic_offload_backend("vmm").build_config_from_args(args)
    assert config.vmm_component_tensor_ownership and config.vmm_native_transport
    for backend, preset in (("linear", "auto"), ("vmm", "nvme_baseline")):
        with pytest.raises(ValueError, match="exclusive"):
            resolve_dynamic_offload_backend(backend, preset).build_config(vmm_component_tensor_ownership=True)


def test_component_gate_explicit_route_bypasses_encoder_group_preset(monkeypatch):
    import diffusers_dynamic_offloader.dynamic_offload as api

    root = Model().eval()
    config = DynamicOffloadConfig(execution_mode="vmm_fast_runtime",
                                 vmm_component_tensor_ownership=True, vmm_native_transport=True)
    calls = []
    def dynamic(module, **kwargs):
        calls.append(kwargs)
        return DynamicOffloadApplyResult(module, hook=object())
    monkeypatch.setattr(api, "enable_dynamic_offload", dynamic)
    monkeypatch.setattr(api, "enable_diffusers_group_offload", lambda *args, **kwargs: pytest.fail("group route"))
    with patch("torch.cuda.init", side_effect=AssertionError("CUDA")):
        result = api.enable_offload(root, component="text_encoder", route="dynamic_offload",
                                    use_environment=False, config=config)
    assert result.route == "dynamic_offload" and result.hook is not None
    assert calls[0]["config"] is config and calls[0]["component"] == "text_encoder"


def test_component_reentry_guard_does_not_retire_the_enclosing_call(monkeypatch):
    class Recursive(torch.nn.Module):
        def forward(self, value, depth=0):
            return self(value, depth=1) if depth == 0 else value

    root = Recursive()
    store = SimpleNamespace(_native_shard_transport=None, _device_ordinal=0,
                            fault_and_copy_group_if_needed=lambda bindings: [], _runtime_stats={})
    adapter = ModelArenaModuleAdapter(store)
    state = _ModuleBinding("recursive", root, [], {}, guard_reentry=True)
    records = []
    event = SimpleNamespace(query=lambda: True, record=lambda stream: records.append(stream))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "CPU-fake-stream")
    adapter._register_execution_unit(state)
    with pytest.raises(ModelArenaError, match="reentrant"):
        root(torch.ones(1))
    assert records == ["CPU-fake-stream"]
    assert state.active is False
    adapter.close()
    assert not root._forward_hooks and not root._forward_pre_hooks


def test_component_output_weight_views_are_detected_without_rejecting_normal_outputs():
    root = torch.nn.Linear(3, 2, bias=False)
    binding = SimpleNamespace(owner=root, parameter_name="weight")
    assert _component_output_borrows_weights({"nested": [root.weight[:, 1:]]}, [binding])
    assert not _component_output_borrows_weights(root(torch.ones(1, 3)), [binding])


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_component_boundary_fences_interrupt_resumes_and_restores_forward(error_type, monkeypatch):
    class Interruptible(torch.nn.Module):
        interruption = error_type

        def forward(self, value, *, multiplier=1):
            if self.interruption:
                raise self.interruption("intentional CPU boundary interruption")
            return value * multiplier

    root = Interruptible()
    store = SimpleNamespace(_native_shard_transport=None, _device_ordinal=0,
                            fault_and_copy_group_if_needed=lambda bindings: [], _runtime_stats={})
    adapter = ModelArenaModuleAdapter(store)
    state = _ModuleBinding("unit", root, [], {}, guard_reentry=True)
    records = []
    event = SimpleNamespace(query=lambda: True, record=lambda stream: records.append(stream))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "CPU-fake-stream")
    adapter._register_execution_unit(state)
    with pytest.raises(error_type, match="intentional CPU boundary"):
        root(torch.ones(1), multiplier=2)
    assert not state.active and records == ["CPU-fake-stream"]
    root.interruption = None
    torch.testing.assert_close(root(torch.ones(1), multiplier=2), torch.full((1,), 2.))
    assert not state.active and len(records) == 2
    adapter.close()
    assert "forward" not in root.__dict__
    assert not root._forward_hooks and not root._forward_pre_hooks


def test_component_capacity_is_available_through_public_metrics(tmp_path):
    root, layout = prepared(tmp_path)
    hook = SimpleNamespace(state=SimpleNamespace(as_dict=lambda: {}), vmm_runtime_stats=lambda: None,
                           _vmm_fast_adapter=SimpleNamespace(component_report=layout.snapshot()))
    report = DynamicOffloadApplyResult(root, hook=hook).metrics_snapshot()
    assert report.as_dict()["runtime"]["component_ownership"]["parameter_groups"] == 9
    assert "experimental_component_ownership" in report.format_profile()


def test_loaded_component_preflight_is_public_detached_and_cuda_free(tmp_path):
    from diffusers_dynamic_offloader import DynamicOffloadLoadResult
    root, layout = prepared(tmp_path)
    original = root.embedding.weight
    result = DynamicOffloadLoadResult(root)
    with patch("torch.cuda.init", side_effect=AssertionError("CUDA")), patch(
            "torch.cuda.is_available", side_effect=AssertionError("CUDA availability")):
        report = result.inspect_vmm_component(component="encoder", buffer_limit_bytes=1024)
    capacity = report.as_dict()["setup"]["component_capacity"]
    assert capacity["parameter_groups"] == 9
    assert capacity["execution_slot_total_payload_bytes"] == 2 * capacity["execution_slot_payload_bytes"]
    assert capacity["native_admission_validated"] is False
    assert root.embedding.weight is root.head.weight is original
    capacity["units"][0]["name"] = "changed"
    assert report.as_dict()["setup"]["component_capacity"]["units"][0]["name"] != "changed"
    with pytest.raises(ValueError, match="nonempty"):
        result.inspect_vmm_component(component=" ")


def test_component_auxiliary_placement_is_owned_by_adapter_not_legacy_copy_loop(monkeypatch):
    hook = DynamicOffloadHook(DynamicOffloadConfig(vmm_component_tensor_ownership=True))
    monkeypatch.setattr(hook, "_move_root_local_tensors_to_device", lambda root: pytest.fail("legacy copy loop"))
    hook._prepare_vmm_fast_auxiliary_tensors(torch.nn.Linear(3, 2))


def test_training_mode_is_refused_before_native_placement(tmp_path):
    root, _ = prepared(tmp_path)
    root.train()
    with pytest.raises(ValueError, match="eval mode"):
        discover_component_layout(root, root._ddo_safetensor_slices, buffer_limit_bytes=1024)


def test_embedding_inplace_renormalization_is_not_treated_as_immutable(tmp_path):
    root, _ = prepared(tmp_path)
    root.embedding.max_norm = 1.0
    with pytest.raises(ValueError, match="renormalization"):
        discover_component_layout(root, root._ddo_safetensor_slices, buffer_limit_bytes=1024)
