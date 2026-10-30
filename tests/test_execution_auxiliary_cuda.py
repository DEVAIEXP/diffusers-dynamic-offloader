"""Small explicit-approval gate for auxiliary native source/view lifetimes."""
import copy
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from diffusers_dynamic_offloader import (
    DynamicOffloadConfig, enable_offload, from_pretrained_with_dynamic_offload,
    remove_dynamic_offload,
)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(128, 128, dtype=torch.bfloat16)
        self.table = torch.nn.Parameter(torch.randn(128, 128, dtype=torch.bfloat16))
        self.alias = self.table
        self.small = torch.nn.Parameter(torch.ones(128, dtype=torch.bfloat16))
        self.interruption = None

    def forward(self, value):
        value = self.projection(value)
        if self.interruption is not None:
            raise self.interruption("intentional auxiliary interruption")
        return value + self.table[0] + self.alias[1] + self.small


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


@pytest.mark.vmm
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_native_auxiliary_alias_reuse_interrupt_unload(error_type, tmp_path):
    if os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Windows CUDA required")
    assert os.environ.get("DDO_VMM_LIBRARY_PATH")
    torch.manual_seed(43)
    original = Model().eval()
    reference = copy.deepcopy(original).cuda()
    value = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        target = reference(value).cpu()
    del reference
    path = tmp_path / "model.safetensors"
    save_file({name: parameter.detach() for name, parameter in original.named_parameters()}, str(path))
    del original

    def loader(directory):
        model = Model().eval()
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=False)
        return model

    model = from_pretrained_with_dynamic_offload(tmp_path, model_loader=loader).module
    registry = model._ddo_safetensor_slices
    small_identities = [id(block.small) for block in model.blocks]
    result = enable_offload(model, component="transformer", route="dynamic_offload", use_environment=False,
                            config=DynamicOffloadConfig(
        execution_mode="vmm_fast_runtime", vmm_native_transport=True,
        vmm_execution_auxiliary_ownership=True,
        vmm_resident_weight_budget_gb=0.07, vmm_staging_buffer_gb=2 / 1024,
        vmm_page_mb=2, vmm_source_cache_gb=64 / 1024, vmm_source_registered_gb=64 / 1024,
    ))
    try:
        reads = None
        for _ in range(4):
            with torch.no_grad():
                actual = model(value).cpu()
            torch.testing.assert_close(actual, target, rtol=0.016, atol=0.016)
            assert all(block.table is block.alias and block.table.device.type == "cuda" for block in model.blocks)
            assert [id(block.small) for block in model.blocks] == small_identities
            assert all(block.small.device.type == "cuda" for block in model.blocks)
            report = result.metrics_snapshot(collect_completed=True).as_dict()
            stats = report["runtime"]["vmm"]
            assert stats["fast_file_backed_bindings"] == 6
            assert stats["fast_cpu_source_bindings"] == 0
            assert report["runtime"]["completion_collection"]["tickets_after"] == 0
            for key in ("fast_fault_failures", "fast_temporary_tensors",
                        "fast_source_cache_admission_denials", "fast_source_cache_bypasses",
                        "fast_source_cache_transfer_owned_events"):
                assert stats[key] == 0
            current = stats["fast_source_cache_source_read_bytes"]
            reads = current if reads is None else reads
            assert current == reads
            model.blocks[0].interruption = error_type
            with torch.no_grad(), pytest.raises(error_type, match="intentional auxiliary interruption"):
                model(value)
            model.blocks[0].interruption = None
        model.blocks[0].interruption = error_type
        with torch.no_grad(), pytest.raises(error_type):
            model(value)
    finally:
        remove_dynamic_offload(model)
    assert registry.closed
    assert all(block.table is block.alias and block.table.device.type == "meta" for block in model.blocks)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    path.unlink()
