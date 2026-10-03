"""CPU/meta-only setup accounting; no CUDA initialization or device queries."""

import pytest
import torch

from diffusers_dynamic_offloader import dynamic_offload as ddo


def make_hook(mode="vmm_fast_runtime"):
    return ddo.DynamicOffloadHook(ddo.DynamicOffloadConfig(
        execution_mode=mode, execution_device="meta", offload_device="cpu",
        small_tensor_threshold_bytes=16, verbose=False,
    ))


def make_model():
    model = torch.nn.Module()
    model.register_parameter("root_weight", torch.nn.Parameter(torch.ones(8)))
    model.add_module("aux", torch.nn.Linear(3, 4))
    model.add_module("native_view", torch.nn.Linear(3, 4, device="meta"))
    other = torch.nn.Module()
    other.register_parameter("small", torch.nn.Parameter(torch.ones(2)))
    other.register_parameter("large", torch.nn.Parameter(torch.ones(8)))
    other.register_buffer("small_buffer", torch.ones(2))
    model.add_module("other", other)
    return model


def old_auxiliary_placement(hook, model):
    hook._move_root_local_tensors_to_device(model)
    for name, child in model.named_modules():
        if isinstance(child, torch.nn.Linear) and child.weight.device.type == "cpu":
            hook._move_module_tensors_to_execution_device(child)
        elif name and not isinstance(child, torch.nn.Linear):
            hook._move_small_local_tensors_to_device(child)


def test_auxiliary_placement_matches_previous_loop_and_counts_actual_bytes(monkeypatch):
    # PyTorch forbids Parameter.data replacement across CPU/meta tensor types.
    # Observe identical copy requests while keeping storage CPU-only.
    calls = []
    original_to = torch.Tensor.to

    def mock_to(tensor, *args, **kwargs):
        if args and args[0] == torch.device("meta"):
            calls.append((tuple(tensor.shape), tensor.dtype, args, kwargs))
            return tensor.clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", mock_to)
    previous, current = make_model(), make_model()
    old_auxiliary_placement(make_hook(), previous)
    previous_calls = list(calls)
    calls.clear()
    hook = make_hook()
    hook._prepare_vmm_fast_auxiliary_tensors(current)
    assert calls == previous_calls
    expected = [(name, tuple(t.shape), t.dtype, t.device) for name, t in
                (*previous.named_parameters(), *previous.named_buffers())]
    actual = [(name, tuple(t.shape), t.dtype, t.device) for name, t in
              (*current.named_parameters(), *current.named_buffers())]
    assert actual == expected
    assert current.other.large.device.type == "cpu"
    assert hook.state.setup_bytes_by_action["vmm_fast_auxiliary_linears_to_device"] == 64
    assert hook.state.setup_bytes_by_action["root_tensors_to_device"] == 32
    assert hook.state.setup_bytes_by_action["small_parameters_to_device"] == 8
    assert hook.state.setup_bytes_by_action["small_buffers_to_device"] == 8
    assert "vmm_fast_auxiliary_linears_to_device" in hook.state.as_dict()["setup_runtime"]


def test_already_placed_auxiliary_modules_are_not_moved_or_counted():
    model = torch.nn.Module()
    model.add_module("placed", torch.nn.Linear(3, 4, device="meta"))
    hook = make_hook()
    hook._prepare_vmm_fast_auxiliary_tensors(model)
    assert not hook.state.setup_seconds_by_action


@pytest.mark.parametrize("mode,prepare_method", [
    ("vmm_fast_runtime", "_prepare_vmm_fast_runtime"),
    ("vmm_runtime", "_prepare_vmm_runtime"),
    ("linear_runtime", "_prepare_linear_runtime"),
])
def test_metadata_accounting_is_exclusive_to_fast_route(monkeypatch, mode, prepare_method):
    hook = make_hook(mode)
    model = torch.nn.Linear(3, 4)
    state = ddo.DynamicOffloadState()
    monkeypatch.setattr(ddo, "build_dynamic_offload_plan", lambda *args: state)
    monkeypatch.setattr(hook, prepare_method, lambda model: None)
    hook.initialize_hook(model)
    assert ("vmm_fast_plan_metadata" in state.setup_seconds_by_action) == (mode == "vmm_fast_runtime")


def test_auxiliary_copy_scope_does_not_add_or_defer_operations(monkeypatch):
    hook = make_hook()
    model = torch.nn.Module()
    model.add_module("aux", torch.nn.Linear(3, 4))
    calls = []
    monkeypatch.setattr(hook, "_move_root_local_tensors_to_device", lambda model: calls.append("root"))
    monkeypatch.setattr(hook, "_move_module_tensors_to_execution_device", lambda model: calls.append("copy"))
    ticks = iter((1.0, 1.25))
    monkeypatch.setattr(ddo.time, "perf_counter", lambda: next(ticks))
    hook._prepare_vmm_fast_auxiliary_tensors(model)
    assert calls == ["root", "copy"]
    assert hook.state.setup_seconds_by_action["vmm_fast_auxiliary_linears_to_device"] == 0.25
