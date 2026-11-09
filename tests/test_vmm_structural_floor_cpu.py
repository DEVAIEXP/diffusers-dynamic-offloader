"""Page-aware structural floor of the native VMM arena (CPU)."""
from diffusers_dynamic_offloader.profiles import (
    VMM_FLOOR_ARENA_HEADROOM_BYTES, recommend_vmm_resident_budget, vmm_structural_floor_bytes,
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
