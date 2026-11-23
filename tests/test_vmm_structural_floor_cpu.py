"""Page-aware structural floor of the native VMM arena (CPU)."""
from diffusers_dynamic_offloader.profiles import (
    VMM_FLOOR_ARENA_HEADROOM_BYTES, recommend_vmm_resident_budget, vmm_structural_floor_bytes,
    vmm_component_structural_floor,
)

MIB = 1024**2


def native_admits(budget, slot, page, slots=2):
    """Mirror of native execution-pool admission (page-aligned slots, one spare page)."""
    capacity = -(-slot // page) * page
    return capacity * slots < (budget // page) * page and (budget // page) > (capacity * slots) // page


def test_small_pages_keep_the_established_floor():
    slot = 304087040  # LTX 2.3 block_child slot, already 2 MiB aligned.
    assert vmm_structural_floor_bytes(slot, 2 * MIB) == 2 * slot + VMM_FLOOR_ARENA_HEADROOM_BYTES
    assert vmm_structural_floor_bytes(slot, None) == 2 * slot + VMM_FLOOR_ARENA_HEADROOM_BYTES


def test_large_pages_reserve_alignment_and_a_spare_page():
    page = 32 * MIB
    for slot in (290 * MIB, 300 * MIB + 2 * MIB, 31 * MIB, 32 * MIB, 33 * MIB):
        floor = vmm_structural_floor_bytes(slot, page)
        assert floor % page == 0 and native_admits(floor, slot, page), slot
        # The old 2 x slot + 64 MiB floor is refused for unaligned slots.
    assert not native_admits(2 * 302 * MIB + VMM_FLOOR_ARENA_HEADROOM_BYTES, 302 * MIB, page)


def test_recommendation_floor_uses_the_observed_page():
    observation = {"device_budget_bytes": 8 * 1024 * MIB, "peak_usage_bytes": 2 * 1024 * MIB,
                   "total_weight_resident_bytes": 640 * MIB, "execution_slots_bytes": 640 * MIB,
                   "page_bytes": 32 * MIB}
    floor = recommend_vmm_resident_budget(observation, 256 * MIB)["floor_bytes"]
    assert floor == 640 * MIB + 64 * MIB and native_admits(floor, 320 * MIB, 32 * MIB)
    observation["page_bytes"] = 128 * MIB
    assert recommend_vmm_resident_budget(observation, 256 * MIB)["floor_bytes"] == 640 * MIB + 128 * MIB


def test_component_floor_uses_actual_slot_layout_and_root_charge(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    layout = SimpleNamespace(
        capacity_snapshot=lambda: {"execution_slot_payload_bytes": 387323904},
        component_lifetime_parameters=((SimpleNamespace(nbytes=8192), None),
                                       (SimpleNamespace(nbytes=16384), None)))
    def discover(module, sources, **kwargs):
        calls.append((module, sources, kwargs))
        return layout, ()
    monkeypatch.setitem(sys.modules, "ddo_vmm_allocator.quantized_arena_adapter",
                        SimpleNamespace(discover_float8_component_layout=discover))
    floor, geometry = vmm_component_structural_floor('module', 'sources', float8_weight_only=True,
                                                     page_bytes=2 * MIB)
    assert floor == 806 * MIB
    assert geometry['component_lifetime_payload_bytes'] == 24576
    assert geometry['layout'] == 'float8_component'
    assert calls == [('module', 'sources', {'buffer_limit_bytes': MIB})]
    assert native_admits(floor - 2 * MIB, 387323904, 2 * MIB)
    assert not native_admits(vmm_structural_floor_bytes(304087040, 2 * MIB) - 2 * MIB,
                             387323904, 2 * MIB)


def test_plain_component_floor_uses_component_discovery(monkeypatch):
    import sys
    from types import SimpleNamespace

    layout = SimpleNamespace(capacity_snapshot=lambda: {'execution_slot_payload_bytes': 5 * MIB},
                             component_lifetime_parameters=())
    monkeypatch.setitem(sys.modules, 'ddo_vmm_allocator.component_arena_adapter',
                        SimpleNamespace(discover_component_layout=lambda *a, **k: layout))
    floor, geometry = vmm_component_structural_floor(None, None, page_bytes=2 * MIB)
    assert floor == 76 * MIB and geometry['layout'] == 'component'


def test_hook_keeps_linear_floor_and_selects_component_floor_before_native_attach(monkeypatch):
    import torch
    import pytest
    from types import SimpleNamespace
    from diffusers_dynamic_offloader import DynamicOffloadConfig
    from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadHook
    from diffusers_dynamic_offloader import profiles
    import ddo_vmm_allocator.model_arena_adapter as adapter
    import ddo_vmm_allocator.quantized_arena_adapter as quantized
    import diffusers_dynamic_offloader.vmm_model_arena as arena

    class BeforeNativeAttach(RuntimeError):
        pass
    def stop(*args):
        raise BeforeNativeAttach('CPU boundary before loading native allocator')
    monkeypatch.setattr(arena, '_load_native_allocator', stop)
    estimates = []
    monkeypatch.setattr(adapter, 'estimate_execution_slot_bytes', lambda *a: estimates.append('linear') or MIB)
    monkeypatch.setattr(profiles, 'vmm_component_structural_floor', lambda *a, **k: (
        806 * MIB, {'layout': 'float8_component', 'floor_bytes': 806 * MIB}))
    monkeypatch.setattr(quantized, 'discover_float8_component_layout', lambda *a, **k: (None, ()))
    model = torch.nn.Linear(2, 2).eval()
    for component, budget, expected in [(False, 0, 68), (True, 0, 806), (True, 6 / 1024, 6)]:
        config = DynamicOffloadConfig(execution_mode='vmm_fast_runtime', execution_device='cuda',
                                      vmm_native_transport=True, vmm_component_tensor_ownership=component,
                                      vmm_float8_weight_only=component, vmm_resident_weight_budget_gb=budget,
                                      vmm_source_cache_gb=64 / 1024, vmm_source_registered_gb=64 / 1024,
                                      vmm_staging_buffer_gb=4 / 1024)
        hook = DynamicOffloadHook(config)
        with pytest.raises(BeforeNativeAttach):
            hook._prepare_vmm_fast_runtime(model)
        assert hook.vmm_resident_weight_budget_bytes == expected * MIB
    assert estimates == ['linear']


def test_recommendation_keeps_root_capacity_reserved_for_large_pages():
    observation = {"device_budget_bytes": 8 * 1024 * MIB, "peak_usage_bytes": 2 * 1024 * MIB,
                   "total_weight_resident_bytes": 1024 * MIB, "execution_slots_bytes": 768 * MIB,
                   "page_bytes": 128 * MIB, "component_lifetime_capacity_bytes": 128 * MIB}
    floor = recommend_vmm_resident_budget(observation, 256 * MIB)['floor_bytes']
    assert floor == 1024 * MIB
    assert native_admits(floor - 128 * MIB, 370 * MIB, 128 * MIB)
    without_root = {k: v for k, v in observation.items() if k != 'component_lifetime_capacity_bytes'}
    old_floor = recommend_vmm_resident_budget(without_root, 256 * MIB)['floor_bytes']
    assert not native_admits(old_floor - 128 * MIB, 370 * MIB, 128 * MIB)
    observation['component_lifetime_capacity_bytes'] = None
    assert recommend_vmm_resident_budget(observation, 256 * MIB) is None
