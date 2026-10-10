"""CPU geometry/publication doubles for FP8 on the existing native arena."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import sys

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization import Float8WeightOnlyConfig, quantize_
from ddo_vmm_allocator.quantized_arena_adapter import (
    Float8ComponentArenaAdapter, discover_float8_component_layout,
)
from ddo_vmm_allocator.model_arena_adapter import _CompletionSet, ModelArenaModuleAdapter, ModelArenaError
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry


@pytest.fixture(autouse=True)
def no_cuda():
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        yield
    assert not torch.cuda.is_initialized()


class Completion:
    def __init__(self, done=True):
        self.done = done
    def query(self):
        return self.done
    def synchronize(self):
        self.done = True


class CpuAdapter(Float8ComponentArenaAdapter):
    # Doubles supply CPU views, not real CUDA/native acceptance evidence.
    def _native_view(self, group, source, offset):
        return source.clone(), None
    def _register_execution_unit(self, state):
        self._finish_composite_publication()
    def _completion_event(self):
        return Completion()


def prepared(tmp_path):
    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList([torch.nn.Linear(32, 16, dtype=torch.bfloat16)])
    root.eval().requires_grad_(False)
    quantize_(root, Float8WeightOnlyConfig(set_inductor_config=False))
    registry = SafetensorSliceRegistry()
    weight = root.blocks[0].weight
    offset = 256
    for tensor, dtype in ((weight.qdata, "F8_E4M3"), (weight.scale, "F32"), (root.blocks[0].bias, "BF16")):
        size = tensor.numel() * tensor.element_size()
        assert registry.bind(tensor, SafetensorFileSlice((tmp_path / "weights.safetensors").resolve(),
                                                      offset, size, tuple(tensor.shape), dtype))
        offset += size
    return root, registry


def test_physical_geometry_and_existing_unit_contains_data_scale_bias(tmp_path):
    root, registry = prepared(tmp_path)
    layout, composites = discover_float8_component_layout(root, registry)
    assert len(composites) == 1
    assert [group.nbytes for group, _offset, _slice in layout.parameters] == [512, 64, 32]
    assert [offset for _group, offset, _slice in layout.parameters] == [0, 512, 768]
    assert layout.payload_bytes == 800
    assert len(layout.units) == 1
    name, module, indices = layout.units[0]
    assert name == "blocks.0" and module is root.blocks[0] and indices == (0, 1, 2)


def test_live_physical_bindings_follow_backend_wrapper_and_restore(tmp_path):
    root, registry = prepared(tmp_path)
    original = root.blocks[0].weight
    forward = type(root.blocks[0]).forward
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    reference = root.blocks[0](inputs)
    layout, composites = discover_float8_component_layout(root, registry)
    store = SimpleNamespace(virtual_bytes=800, page_bytes=256, _streaming_enabled=False,
                            _native_shard_transport=SimpleNamespace(prepare_sources=lambda units: None),
                            _source_stats={"file_backed_bindings": 0, "file_backed_bytes": 0},
                            view_setup=lambda spans: nullcontext())
    adapter = CpuAdapter(store, composites)
    adapter.attach_component(layout)
    assert type(root.blocks[0]).forward is forward
    assert type(root.blocks[0].weight) is type(original)
    assert composites[0].logical_group._original is None
    bindings = adapter._execution_units[0].tensors
    assert [binding.nbytes for binding in bindings] == [512, 64, 32]
    assert all(binding.file_slice is not None for binding in bindings)
    assert store._source_stats["file_backed_bytes"] == 608
    torch.testing.assert_close(root.blocks[0](inputs), reference, rtol=0, atol=0)
    # Same physical `.data` operation used by native slots and fallback hooks.
    for binding in bindings:
        temporary = binding.stable.clone()
        binding.owner._parameters[binding.parameter_name].data = temporary
        assert binding.virtual_address == temporary.data_ptr()
    torch.testing.assert_close(root.blocks[0](inputs), reference, rtol=0, atol=0)
    for binding in bindings:
        binding.owner._parameters[binding.parameter_name].data = binding.stable
    torch.testing.assert_close(root.blocks[0](inputs), reference, rtol=0, atol=0)
    adapter.close()
    assert root.blocks[0].weight.qdata.device.type == "meta"
    assert root.blocks[0].weight.scale.device.type == "meta"
    assert root.blocks[0].bias.device.type == "meta"


def test_missing_scale_refuses_before_module_mutation(tmp_path):
    root, registry = prepared(tmp_path)
    original = root.blocks[0].weight
    registry._slices.pop((original.scale.data_ptr(), original.scale.numel() * original.scale.element_size()))
    with pytest.raises(ValueError, match="immutable"):
        discover_float8_component_layout(root, registry)
    assert root.blocks[0].weight is original


class RootedComponent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(32, 16, dtype=torch.bfloat16)])

    def forward(self, value):
        return self.blocks[0](value * self.gain) * self.gain


def rooted_prepared(tmp_path):
    root = RootedComponent().eval().requires_grad_(False)
    quantize_(root, Float8WeightOnlyConfig(set_inductor_config=False))
    registry = SafetensorSliceRegistry()
    offset = 256
    for tensor, dtype in ((root.gain, "BF16"), (root.blocks[0].weight.qdata, "F8_E4M3"),
                          (root.blocks[0].weight.scale, "F32"), (root.blocks[0].bias, "BF16")):
        size = tensor.numel() * tensor.element_size()
        assert registry.bind(tensor, SafetensorFileSlice((tmp_path / "rooted.safetensors").resolve(),
                                                      offset, size, tuple(tensor.shape), dtype))
        offset += size
    return root, registry


def test_root_state_geometry_is_bounded_and_dense_overlap_guard_unchanged(tmp_path):
    from ddo_vmm_allocator.component_arena_adapter import discover_component_layout
    root, registry = rooted_prepared(tmp_path)
    layout, _ = discover_float8_component_layout(root, registry, buffer_limit_bytes=2)
    assert len(layout.units) == 1 and layout.units[0][0] == "blocks.0"
    assert len(layout.component_lifetime_parameters) == 1
    group, descriptor = layout.component_lifetime_parameters[0]
    assert group.kind == "parameter" and descriptor.size_bytes == 2
    assert layout.snapshot()["component_lifetime_parameter_bytes"] == 2
    assert layout.capacity_snapshot()["unique_parameter_bytes"] == 610
    with pytest.raises(ValueError, match="overlapping"):
        discover_component_layout(root, registry, buffer_limit_bytes=2, _prepared_groups=layout.groups)
    with pytest.raises(ValueError, match="explicit component limit"):
        discover_float8_component_layout(root, registry, buffer_limit_bytes=1)
    registry._slices.pop((root.gain.data_ptr(), 2))
    with pytest.raises(ValueError, match="immutable file source"):
        discover_float8_component_layout(root, registry)


def test_root_state_alias_into_child_is_refused(tmp_path):
    root, registry = rooted_prepared(tmp_path)
    root.blocks[0].bias = root.gain
    with pytest.raises(ValueError, match="root-only"):
        discover_float8_component_layout(root, registry)


@pytest.mark.parametrize("page,expected_charge", [(0, 256), (256, 256), (300, 512)])
def test_root_native_capacity_is_subtracted_from_existing_total_ceiling(tmp_path, page, expected_charge):
    from ddo_vmm_allocator.quantized_arena_adapter import _component_lifetime_store_kwargs
    root, registry = rooted_prepared(tmp_path)
    layout, _ = discover_float8_component_layout(root, registry)
    library = SimpleNamespace(device_granularity=lambda device: 256)
    kwargs = {"page_size_bytes": page, "resident_budget_bytes": 2048}
    adjusted, charge = _component_lifetime_store_kwargs(layout, library, kwargs)
    assert charge == expected_charge
    assert adjusted["resident_budget_bytes"] + charge == kwargs["resident_budget_bytes"]
    assert kwargs["resident_budget_bytes"] == 2048
    with pytest.raises(ValueError, match="existing weight ceiling"):
        _component_lifetime_store_kwargs(layout, library, {**kwargs, "resident_budget_bytes": charge})


class CpuRootBacking:
    def __init__(self, size):
        self.capacity_bytes = size
        self.uploads = 0
        self.fences = 0
        self.closed = False

    def upload(self, bindings):
        self.uploads += 1
        self.bindings = bindings
        return [(binding, binding.owner._parameters[binding.parameter_name].detach().clone())
                for binding in bindings], Completion()

    def finish_compute(self, stream):
        self.fences += 1

    def close(self):
        self.closed = True
        self.bindings = []


class CpuRootAdapter(CpuAdapter):
    def _make_component_lifetime_buffer(self, size):
        return CpuRootBacking(size)

    def _fence_component_lifetime(self):
        self._component_lifetime_buffer.finish_compute(None)


def test_root_state_publishes_once_keeps_parameter_semantics_and_fenced_retirement(tmp_path):
    import weakref
    import gc

    root, registry = rooted_prepared(tmp_path)
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    reference = root(inputs)
    original = weakref.ref(root.gain)
    forward = type(root).forward
    layout, composites = discover_float8_component_layout(root, registry)
    store = SimpleNamespace(virtual_bytes=layout.payload_bytes, page_bytes=256, _streaming_enabled=False,
                            _native_shard_transport=SimpleNamespace(prepare_sources=lambda units: None),
                            _source_stats={"file_backed_bindings": 0, "file_backed_bytes": 0},
                            view_setup=lambda spans: nullcontext(), wait_for_compute=lambda ready: None)
    adapter = CpuRootAdapter(store, composites)
    adapter.attach_component(layout)
    backing = adapter._component_lifetime_buffer
    assert type(root.gain) is torch.nn.Parameter and "gain" not in root._buffers
    assert type(root).forward is forward
    gc.collect()
    assert original() is None and layout.component_lifetime_parameters[0][0]._original is None
    assert not isinstance(backing.bindings[0].source, torch.Tensor)
    for _ in range(3):
        torch.testing.assert_close(root(inputs), reference, rtol=0, atol=0)
    assert backing.uploads == 1
    assert store._source_stats["file_backed_bindings"] == 4
    assert store._source_stats["file_backed_bytes"] == 610
    event = SimpleNamespace(query=lambda: False, synchronize=lambda: None)
    adapter._completion_event = lambda: event
    with pytest.raises(ModelArenaError, match="completion is unproven"):
        adapter.close()
    assert root.gain.device.type == "cpu" and not backing.closed
    event.query = lambda: True
    adapter.close()
    assert root.gain.device.type == "meta" and root.blocks[0].weight.qdata.device.type == "meta"
    assert backing.closed and backing.fences == 2
    assert adapter._component_lifetime_buffer is None


def test_partial_root_upload_failure_retains_backing_until_close_proves_completion(tmp_path):
    root, registry = rooted_prepared(tmp_path)
    original = root.gain
    layout, composites = discover_float8_component_layout(root, registry)
    store = SimpleNamespace(virtual_bytes=layout.payload_bytes, page_bytes=256, _streaming_enabled=False,
                            _native_shard_transport=SimpleNamespace(prepare_sources=lambda units: None),
                            _source_stats={"file_backed_bindings": 0, "file_backed_bytes": 0},
                            view_setup=lambda spans: nullcontext(), wait_for_compute=lambda ready: None)
    adapter = CpuRootAdapter(store, composites)
    backing = CpuRootBacking(256)
    adapter._make_component_lifetime_buffer = lambda size: backing
    backing.upload = lambda bindings: (_ for _ in ()).throw(RuntimeError("partial root DMA"))
    with pytest.raises(RuntimeError, match="partial root DMA"):
        adapter.attach_component(layout)
    assert root.gain is original and not backing.closed
    adapter._fence_component_lifetime = lambda: (_ for _ in ()).throw(RuntimeError("unproven root fence"))
    with pytest.raises(RuntimeError, match="unproven root fence"):
        adapter.close()
    assert not backing.closed and adapter._ownership_groups
    adapter._fence_component_lifetime = lambda: backing.finish_compute(None)
    adapter.close()
    assert backing.closed and root.gain is original
    assert root.blocks[0].bias.device.type == "meta"


def test_root_state_uses_real_single_native_buffer_transport_and_fence_contract(tmp_path):
    from ddo_vmm_allocator.model_arena_adapter import NativeExecutionBuffer, ModelArenaWeightStore
    from ddo_vmm_allocator import model_arena_adapter as native_module

    root, registry = rooted_prepared(tmp_path)
    layout, composites = discover_float8_component_layout(root, registry)
    calls = []
    native = SimpleNamespace(info=SimpleNamespace(address=4096, resident_size=256),
                             wait_compute=lambda stream: calls.append("wait_upload"),
                             fault_range=lambda offset, size: calls.append(("fault", size)),
                             record_compute=lambda stream: calls.append("fence"),
                             drain_compute=lambda: calls.append("drain"))
    def close_native():
        assert root.gain.device.type == "meta"
        calls.append("close")
    native.close = close_native
    def create(size, **kwargs):
        assert size == 256 and kwargs["resident_budget_bytes"] == 256
        calls.append("create_single_arena")
        return native
    def upload(bindings, arena, stream):
        assert arena is native and len(bindings) == 1
        for binding in bindings:
            assert binding.file_slice is not None and not isinstance(binding.source, torch.Tensor)
            binding.stable.copy_(binding.owner._parameters[binding.parameter_name])
        calls.append("immutable_upload")
        return Completion()
    store = SimpleNamespace(virtual_bytes=layout.payload_bytes, page_bytes=256, _streaming_enabled=False,
                            _native_shard_transport=SimpleNamespace(prepare_sources=lambda units: None, upload_batch=upload),
                            _source_stats={"file_backed_bindings": 0, "file_backed_bytes": 0},
                            _runtime_stats={key: 0 for key in ("dma_enqueue_seconds", "dma_enqueue_bytes",
                                "upload_bytes", "uploads", "execution_buffer_uploads", "execution_buffer_bytes")},
                            view_setup=lambda spans: nullcontext(), wait_for_compute=lambda ready: None,
                            _library=SimpleNamespace(create_model_arena=create), _device_ordinal=0,
                            _backing="device", _upload_stream=object())
    adapter = CpuRootAdapter(store, composites)
    adapter._make_component_lifetime_buffer = lambda size: Float8ComponentArenaAdapter._make_component_lifetime_buffer(adapter, size)
    with patch.object(torch.cuda, "stream", lambda stream: nullcontext()), \
            patch.object(native_module, "cuda_uint8_tensor_from_address",
                         lambda address, size, **kwargs: (torch.empty(size, dtype=torch.uint8), native)):
        adapter.attach_component(layout)
    assert isinstance(adapter._component_lifetime_buffer, NativeExecutionBuffer)
    assert calls.count("immutable_upload") == 1 and ("fault", 256) in calls
    assert store._component_lifetime_buffer is adapter._component_lifetime_buffer
    # Exercise the actual public counter aggregation with detached native doubles.
    store.native = SimpleNamespace(info=SimpleNamespace(resident_size=512, fault_ceiling=768,
                                   virtual_size=1024, page_size=256), residency_counters={})
    store.execution_buffer = None
    store._staging_pool = None
    transport = store._native_shard_transport
    store._native_shard_transport = None
    metrics = ModelArenaWeightStore.runtime_stats(store)
    assert metrics["component_lifetime_resident_bytes"] == 256
    assert metrics["total_weight_resident_bytes"] == 768
    store._native_shard_transport = transport
    adapter.close()
    assert calls[-2:] == ["drain", "close"]
    assert store._component_lifetime_buffer is None


def test_cross_module_wrapper_alias_refuses_initial_native_gate(tmp_path):
    root, registry = prepared(tmp_path)
    extra = torch.nn.Linear(32, 16, bias=False, dtype=torch.bfloat16).eval()
    extra.weight = root.blocks[0].weight
    root.blocks.append(extra)
    with pytest.raises(ValueError, match="unique"):
        discover_float8_component_layout(root, registry)


def test_partial_upload_completion_is_not_compute_completion():
    compute, upload = Completion(True), Completion(False)
    completion = _CompletionSet((compute, upload))
    assert not completion.query()
    upload.done = True
    assert completion.query()


def test_fresh_bootstrap_import_does_not_import_adapters_or_torchao():
    command = """
import sys
import ddo_vmm_allocator.torch_allocator
assert 'torch' not in sys.modules
assert 'diffusers' not in sys.modules
assert 'torchao' not in sys.modules
assert 'ddo_vmm_allocator.model_arena_adapter' not in sys.modules
import torch
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", command], check=True)


@pytest.mark.parametrize("failed_compute_fence", [False, True])
def test_real_hook_partial_fallback_retains_upload_ownership(tmp_path, failed_compute_fence):
    root, registry = prepared(tmp_path)
    layout, composites = discover_float8_component_layout(root, registry)
    store = SimpleNamespace(virtual_bytes=800, page_bytes=256, _streaming_enabled=False,
                            _native_shard_transport=SimpleNamespace(prepare_sources=lambda units: None),
                            _source_stats={"file_backed_bindings": 0, "file_backed_bytes": 0},
                            view_setup=lambda spans: nullcontext(), _device_ordinal=0,
                            _debug_faults=False)
    adapter = CpuAdapter(store, composites)
    adapter.attach_component(layout)
    state = adapter._execution_units[0]
    state.composite_payload_ownership = True
    store._native_shard_transport = None
    store.record_fault_failure = lambda error: None
    store.record_temporary = lambda size: None
    uploads = []
    def temporary(binding):
        event = Completion(False)
        uploads.append(event)
        return binding.stable.clone(), event
    def wait(event):
        if len(uploads) == 2:
            raise RuntimeError("wait failed")
    def fault(bindings):
        raise ModelArenaError("materialization failed")
    store.temporary_cuda_source, store.wait_for_compute = temporary, wait
    adapter._fault_execution_unit_with_capacity_retry = fault
    class ComputeEvent(Completion):
        def __init__(self, **kwargs):
            super().__init__(True)
        def record(self, stream):
            if failed_compute_fence:
                raise RuntimeError("fence failed")
    stream = SimpleNamespace(cuda_stream=0)
    with patch.object(torch.cuda, "Event", ComputeEvent), \
         patch.object(torch.cuda, "current_stream", return_value=stream):
        ModelArenaModuleAdapter._register_execution_unit(adapter, state)
        with pytest.raises(RuntimeError, match="wait failed"):
            root.blocks[0](torch.randn(4, 32, dtype=torch.bfloat16))
    assert len(uploads) == 2
    if failed_compute_fence:
        assert state.active and len(state.temporary_views) == 2
        with pytest.raises(ModelArenaError, match="active"):
            adapter.close()
    else:
        assert not state.active and len(adapter._temporary_lifetimes) == 1
        adapter._collect_temporary_lifetimes()
        assert len(adapter._temporary_lifetimes) == 1
        for event in uploads:
            event.done = True
        adapter._collect_temporary_lifetimes()
        assert not adapter._temporary_lifetimes
        adapter.close()
