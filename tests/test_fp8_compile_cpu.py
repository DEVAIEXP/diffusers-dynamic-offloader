"""CPU graph/ownership compatibility only; not CUDA Inductor acceptance."""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization import Float8WeightOnlyConfig, quantize_
from torch._dynamo.backends.registry import lookup_backend
from ddo_vmm_allocator.quantized_owners import capture_float8_weight_owners_cpu
from diffusers_dynamic_offloader.float8_compile import configure_float8_linear_compile


@pytest.fixture(autouse=True)
def no_cuda():
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        yield
    assert not torch.cuda.is_initialized()


class Completion:
    def query(self):
        return True


@pytest.mark.parametrize("replacement", ["payload_data", "wrapper_publication"])
def test_compiled_backend_forward_tracks_current_payloads(replacement):
    torch.manual_seed(43)
    model = torch.nn.Sequential(torch.nn.Linear(32, 16, dtype=torch.bfloat16)).eval()
    model.requires_grad_(False)
    quantize_(model, Float8WeightOnlyConfig(set_inductor_config=False))
    layer = model[0]
    forward = type(layer).forward
    group = capture_float8_weight_owners_cpu(model, layer.weight)
    first_payloads = {name: getattr(layer.weight, name).clone() for name in ("qdata", "scale")}
    if replacement == "payload_data":
        group.publish_payload_views(first_payloads)
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    backend = lookup_backend("aot_eager")
    graphs = []

    def record_graph(graph, example_inputs):
        graphs.append(graph)
        return backend(graph, example_inputs)

    compiled = torch.compile(layer.forward, backend=record_graph, fullgraph=True, dynamic=False)
    reference = layer(inputs)
    torch.testing.assert_close(compiled(inputs), reference, rtol=0, atol=0)
    assert graphs, "The backend must receive a graph, not silently run eager"
    assert type(layer).forward is forward
    original_scale = layer.weight.scale.clone()
    replacement_payloads = {"qdata": layer.weight.qdata.clone(), "scale": original_scale * 0.25}
    if replacement == "payload_data":
        layer.weight.qdata.data = replacement_payloads["qdata"]
        layer.weight.scale.data = replacement_payloads["scale"]
    else:
        group.publish_payload_views(replacement_payloads)
    changed = layer(inputs)
    assert not torch.equal(changed, reference), "The fixture must detect stale weight capture"
    torch.testing.assert_close(compiled(inputs), changed, rtol=0, atol=0)
    torch.testing.assert_close(compiled(inputs), changed, rtol=0, atol=0)
    assert type(layer).forward is forward
    assert group.retire(Completion())
    assert layer.weight.qdata.device.type == "meta"
    assert layer.weight.scale.device.type == "meta"
    del compiled, graphs
    torch._dynamo.reset()


def test_compile_configuration_is_explicit_and_transactional():
    model = torch.nn.Sequential(torch.nn.Linear(32, 16, dtype=torch.bfloat16)).eval()
    model.requires_grad_(False)
    quantize_(model, Float8WeightOnlyConfig(set_inductor_config=False))
    model.append(torch.nn.Linear(16, 8, dtype=torch.bfloat16).eval())
    with patch.object(torch, "compile", side_effect=lambda fn, **kwargs: lambda *a: fn(*a)) as compiler:
        report, handle = configure_float8_linear_compile(model)
    assert compiler.call_count == report["compiled_linear_count"] == 1
    assert compiler.call_args.kwargs == {
        "backend": "inductor", "fullgraph": True, "dynamic": False, "options": report["options"]}
    assert report["options"]["pattern_matcher"] is False
    assert report["options"]["epilogue_fusion"] is False
    assert "forward" not in vars(model[1])
    compiled = model[0].forward
    model[0].forward = lambda *args: None
    with pytest.raises(RuntimeError, match="Retire"):
        handle.remove()
    assert len(handle.bindings) == 1
    model[0].forward = compiled
    handle.remove()
    assert "forward" not in vars(model[0])
    handle.remove()


def test_compile_setup_failure_restores_preceding_forwards():
    model = torch.nn.Sequential(torch.nn.Linear(32, 16, dtype=torch.bfloat16),
                               torch.nn.Linear(16, 8, dtype=torch.bfloat16)).eval()
    model.requires_grad_(False)
    quantize_(model, Float8WeightOnlyConfig(set_inductor_config=False))
    with patch.object(torch, "compile", side_effect=[lambda value: value, RuntimeError("compile failure")]):
        with pytest.raises(RuntimeError, match="compile failure"):
            configure_float8_linear_compile(model)
    assert all("forward" not in vars(module) for module in model)


def test_more_than_eight_geometries_use_isolated_caches_and_current_weights():
    torch._dynamo.reset()
    limit = torch._dynamo.config.recompile_limit
    model = torch.nn.ModuleList([
        torch.nn.Linear(32, 16 + i, dtype=torch.bfloat16) for i in range(12)
    ]).eval().requires_grad_(False)
    quantize_(model, Float8WeightOnlyConfig(set_inductor_config=False))
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    references = [layer(inputs) for layer in model]
    original_compile = torch.compile
    graphs = []
    backend = lookup_backend("aot_eager")

    def record(graph, example_inputs):
        graphs.append(graph)
        return backend(graph, example_inputs)

    def compile_cpu(fn, **kwargs):
        return original_compile(fn, backend=record, fullgraph=True, dynamic=False)

    with patch.object(torch, "compile", side_effect=compile_cpu):
        report, handle = configure_float8_linear_compile(model)
    try:
        assert report["compiled_geometry_count"] == 12
        for layer, reference in zip(model, references):
            torch.testing.assert_close(layer(inputs), reference, rtol=0, atol=0)
            layer.weight.scale.data = layer.weight.scale.clone() * 0.25
            changed = torch.nn.Linear.forward(layer, inputs)
            assert not torch.equal(changed, reference)
            torch.testing.assert_close(layer(input=inputs), changed, rtol=0, atol=0)
        assert len(graphs) == 12
        assert torch._dynamo.config.recompile_limit == limit
    finally:
        handle.remove()
        torch._dynamo.reset()
