"""Opt-in component integration gate; execute only with explicit approval."""

import copy
import os
import weakref

import pytest
import torch
from safetensors.torch import save_file, load_file

from diffusers_dynamic_offloader import (
    DynamicOffloadConfig, enable_offload, from_pretrained_with_dynamic_offload,
    remove_dynamic_offload,
)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(32, 32, dtype=torch.bfloat16)
        self.norm = torch.nn.LayerNorm(32, dtype=torch.bfloat16)
        self.interruption = None

    def forward(self, value):
        value = self.projection(value)
        if self.interruption:
            raise self.interruption("intentional component interruption")
        return self.norm(value)


class Component(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 32, dtype=torch.bfloat16)
        self.blocks = torch.nn.ModuleList([Block(), Block()])
        self.head = torch.nn.Linear(32, 17, bias=False, dtype=torch.bfloat16)
        self.head.weight = self.embedding.weight
        self.register_buffer("scale", torch.ones((), dtype=torch.bfloat16), persistent=False)

    def forward(self, ids):
        value = self.embedding(ids)
        for block in self.blocks:
            value = block(value)
        return self.head(value) * self.scale


class ParameterParent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Parameter(torch.randn(32, 32, dtype=torch.bfloat16) / (32 ** 0.5))
        self.norm = torch.nn.LayerNorm(32, dtype=torch.bfloat16)
        self.interruption = None

    def forward(self, value):
        value = value @ self.projection
        if self.interruption:
            raise self.interruption("intentional component interruption")
        return self.norm(value)


class NestedComponent(Component):
    def __init__(self):
        super().__init__()
        self.projector = ParameterParent()

    def forward(self, ids):
        value = self.projector(self.embedding(ids))
        for block in self.blocks:
            value = block(value)
        return self.head(value) * self.scale


def _run_component_gate(error_type, tmp_path, *, component_type=Component,
                        expected_groups=9, interruption_path="blocks.0"):
    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    assert os.environ.get("DDO_VMM_LIBRARY_PATH"), "Select the isolated lifecycle OFF library"
    torch.manual_seed(43)
    original = component_type().eval()
    inputs = [torch.tensor([0]), torch.tensor([[0, 4, 16, 2]]), torch.arange(9) % 17]
    reference = copy.deepcopy(original).cuda()
    with torch.no_grad():
        expected = [reference(ids.cuda()).cpu() for ids in inputs]
    del reference
    path = tmp_path / "model.safetensors"
    save_file({name: value.detach() for name, value in original.named_parameters()}, str(path))
    del original

    def loader(directory):
        model = component_type().eval()
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=False)
        return model

    loaded = from_pretrained_with_dynamic_offload(tmp_path, model_loader=loader)
    model = loaded.module
    registry = model._ddo_safetensor_slices  # Library contract, not host usage.
    assert loaded.metrics_snapshot().as_dict()["setup"]["weight_source"]["bound"] == expected_groups
    interrupted_unit = model.get_submodule(interruption_path)
    old_weight = weakref.ref(model.embedding.weight)
    result = enable_offload(model, component="text_encoder", route="dynamic_offload", use_environment=False, config=DynamicOffloadConfig(
        execution_mode="vmm_fast_runtime", vmm_native_transport=True,
        vmm_component_tensor_ownership=True, vmm_component_buffer_limit_mb=1,
        vmm_resident_weight_budget_gb=0.07, vmm_staging_buffer_gb=2 / 1024,
        vmm_page_mb=2, vmm_source_cache_gb=64 / 1024, vmm_source_registered_gb=64 / 1024,
    ))
    try:
        assert result.route == "dynamic_offload" and result.hook is not None
        assert old_weight() is None
        assert model.embedding.weight is model.head.weight
        assert all(parameter.device.type == "cuda" for parameter in model.parameters())
        assert model.scale.device.type == "cuda"
        first_reads = None
        for repeat in range(4):
            for ids, target in zip(inputs, expected, strict=True):
                with torch.no_grad():
                    actual = model(ids.cuda()).cpu()
                torch.testing.assert_close(actual, target, rtol=0.016, atol=0.016)
                assert model.embedding.weight is model.head.weight
                report = result.metrics_snapshot(collect_completed=True).as_dict()
                stats = report["runtime"]["vmm"]
                assert stats["fast_cpu_source_bindings"] == 0
                assert stats["fast_file_backed_bindings"] == expected_groups
                assert report["runtime"]["component_ownership"]["buffer_bytes"] == 2
                assert report["runtime"]["component_ownership"]["execution_units"] == (5 if expected_groups == 12 else 4)
                assert report["runtime"]["completion_collection"]["tickets_after"] == 0
                assert stats["fast_source_cache_transfer_owned_events"] == 0
                assert stats["fast_source_cache_committed_bytes"] <= 64 * 2**20
                assert stats["fast_source_cache_registered_bytes"] <= 64 * 2**20
                for key in ("fast_fault_failures", "fast_temporary_tensors",
                            "fast_source_cache_admission_denials", "fast_source_cache_bypasses"):
                    assert stats[key] == 0
                reads = stats["fast_source_cache_source_read_bytes"]
                first_reads = reads if first_reads is None else first_reads
                assert reads == first_reads
            interrupted_unit.interruption = error_type
            with torch.no_grad(), pytest.raises(error_type, match="intentional component interruption"):
                model(inputs[0].cuda())
            interrupted_unit.interruption = None
        # Resume numerically, then interrupt immediately before unload: no
        # intermediate metrics collection/readback may hide close ordering.
        with torch.no_grad():
            actual = model(inputs[-1].cuda()).cpu()
        torch.testing.assert_close(actual, expected[-1], rtol=0.016, atol=0.016)
        interrupted_unit.interruption = error_type
        with torch.no_grad(), pytest.raises(error_type):
            model(inputs[0].cuda())
    finally:
        remove_dynamic_offload(model)
    assert registry.closed
    assert model.embedding.weight is model.head.weight
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert model.scale.device.type == "meta"
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    path.unlink()


@pytest.mark.vmm
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_component_shared_weights_norms_buffers_reuse_cancel_unload(error_type, tmp_path):
    _run_component_gate(error_type, tmp_path)


@pytest.mark.vmm
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_nested_parameter_parent_numerical_reuse_interrupt_unload(error_type, tmp_path):
    _run_component_gate(error_type, tmp_path, component_type=NestedComponent,
                        expected_groups=12, interruption_path="projector")
