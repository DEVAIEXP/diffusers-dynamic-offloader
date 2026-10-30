import unittest

import torch
import torch.nn as nn

from diffusers_dynamic_offloader import DynamicOffloadConfig, apply_dynamic_offload, remove_dynamic_offload


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16)

    def forward(self, value):
        return self.linear(value)


class _SequentialBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block(), _Block()])

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GroupedVmmRuntimeTests(unittest.TestCase):
    def test_module_list_groups_hold_weights_for_the_whole_block(self):
        torch.manual_seed(7)
        model = _SequentialBlocks()
        reference = _SequentialBlocks()
        reference.load_state_dict(model.state_dict())
        inputs = torch.randn(2, 1024, dtype=torch.bfloat16)
        expected = reference(inputs)

        hook = apply_dynamic_offload(
            model,
            DynamicOffloadConfig(
                execution_mode="vmm_runtime",
                vmm_resident_weight_budget_gb=0.07,
                vmm_staging_buffer_gb=0.003,
                show_profile=False,
            ),
        )
        try:
            actual = model(inputs.cuda()).cpu()
            torch.cuda.synchronize()
            self.assertTrue(torch.allclose(actual, expected, rtol=0.03, atol=0.03))
            self.assertEqual(hook.state.planner_decisions["vmm_execution_group_count"], 3)
            self.assertEqual(hook.state.planner_decisions["vmm_physical_page_mb"], 32.0)
            self.assertEqual(hook.vmm_runtime_stats()["arena_materializations"], 3)
        finally:
            remove_dynamic_offload(model)
