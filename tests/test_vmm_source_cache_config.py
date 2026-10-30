"""CPU-only configuration contracts for the opt-in native source cache."""

import pytest

from diffusers_dynamic_offloader.dynamic_offload import (
    _native_source_cache_transport_config,
    load_dynamic_offload_settings_from_env,
)

MIB = 1024**2


@pytest.mark.parametrize("registered_gb", [0.5, 24.625])
def test_full_source_registration_is_explicit_and_does_not_change_storage(registered_gb):
    config = _native_source_cache_transport_config(
        1536 * MIB, 0, 0, 0.0, native_transport=True,
        source_cache_gb=24.625, source_registered_gb=registered_gb,
    )
    capacity = int(24.625 * 1024**3)
    assert config["native_source_cache_slot_bytes"] * config["native_source_cache_slots"] == capacity
    assert config["native_source_cache_registered_limit_bytes"] == int(registered_gb * 1024**3)
    assert config["native_source_cache_process_limit_bytes"] == capacity + 1536 * MIB


def test_native_persistent_storage_has_separate_registration_ceiling():
    config = _native_source_cache_transport_config(
        512 * 1024**2, 0, 0, 0.0, native_transport=True,
        source_cache_gb=1.0, source_registered_gb=0.25,
    )
    assert config["native_source_cache_slot_bytes"] * config["native_source_cache_slots"] == 1024**3
    assert config["native_source_cache_registered_limit_bytes"] == 256 * 1024**2
    assert config["native_source_cache_process_limit_bytes"] == 1536 * 1024**2


@pytest.mark.parametrize("registered", [-1, float("nan"), float("inf"), 2, 0.01])
def test_native_persistent_storage_rejects_invalid_registration_ceiling(registered):
    with pytest.raises(ValueError):
        _native_source_cache_transport_config(
            512 * 1024**2, 0, 0, 0.0, native_transport=True,
            source_cache_gb=1.0, source_registered_gb=registered,
        )

def test_native_source_cache_is_off_by_default() -> None:
    assert _native_source_cache_transport_config(640 * MIB, 0, 0, 0.0) == {}
    settings = load_dynamic_offload_settings_from_env(running_on_wsl=False, environ={})
    assert settings.config.vmm_native_source_cache_slot_mb == 0
    assert settings.config.vmm_native_source_cache_slots == 0
    assert settings.config.vmm_native_transport is False


def test_native_transport_cache_off_uses_the_same_ring_capacity() -> None:
    off = _native_source_cache_transport_config(640 * MIB, 0, 0, 0.0, native_transport=True)
    on = _native_source_cache_transport_config(640 * MIB, 64, 2, 0.0, native_transport=True)
    assert off == {
        "native_transport_slot_size_bytes": 64 * MIB,
        "native_transport_slots": 10,
    }
    assert all(on[key] == value for key, value in off.items())
    assert not any(key.startswith("native_source_cache") for key in off)


def test_native_transport_rounds_total_capacity_only_once() -> None:
    config = _native_source_cache_transport_config(65 * MIB, 0, 0, 0.0, native_transport=True)
    assert config["native_transport_slot_size_bytes"] * config["native_transport_slots"] == 128 * MIB


@pytest.mark.parametrize("staging_bytes", [0, -MIB, 2049 * MIB])
def test_native_transport_rejects_unsafe_capacity_without_cache(staging_bytes: int) -> None:
    with pytest.raises(ValueError):
        _native_source_cache_transport_config(staging_bytes, 0, 0, 0.0, native_transport=True)


def test_native_source_cache_accounts_for_ring_and_cache() -> None:
    config = _native_source_cache_transport_config(2 * MIB, 2, 1, 0.0)
    assert config == {
        "native_transport_slot_size_bytes": 2 * MIB,
        "native_transport_slots": 1,
        "native_source_cache_slot_bytes": 2 * MIB,
        "native_source_cache_slots": 1,
        "native_source_cache_process_limit_bytes": 4 * MIB,
    }


def test_model_cache_and_existing_host_ring_are_separate_explicit_capacities() -> None:
    config = _native_source_cache_transport_config(1536 * MIB, 0, 0, 0, source_cache_gb=24)
    assert config["native_transport_slots"] == 24
    assert config["native_source_cache_slot_bytes"] == 64 * MIB
    assert config["native_source_cache_slots"] == 384
    assert config["native_source_cache_process_limit_bytes"] == (24 * 1024 + 1536) * MIB
    assert config["native_source_cache_model_windows"] is True
    # Metadata limits do not approve allocating this much memory in a test.


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), 33])
def test_model_cache_rejects_unbounded_or_invalid_capacity(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        _native_source_cache_transport_config(2 * MIB, 0, 0, 0, source_cache_gb=value)


def test_model_cache_cannot_mix_legacy_slots() -> None:
    with pytest.raises(ValueError, match="not both"):
        _native_source_cache_transport_config(2 * MIB, 2, 1, 0, source_cache_gb=1)


def test_native_source_cache_separates_model_staging_from_local_capacity() -> None:
    config = _native_source_cache_transport_config(640 * MIB, 64, 2, 0.0)
    assert config["native_transport_slot_size_bytes"] == 64 * MIB
    assert config["native_transport_slots"] == 10
    assert config["native_source_cache_process_limit_bytes"] == 768 * MIB


def test_native_source_cache_rejects_process_cap_over_one_gib() -> None:
    with pytest.raises(ValueError, match=r"registered staging \+ cache <=1 GiB"):
        _native_source_cache_transport_config(1024 * MIB, 2, 1, 0.0)


def test_native_source_cache_keeps_local_capacity_bounded() -> None:
    with pytest.raises(ValueError, match="cache <=128 MiB"):
        _native_source_cache_transport_config(2 * MIB, 64, 3, 0.0)


@pytest.mark.parametrize("slot_mb,slots", [(2, 0), (0, 1), (-1, 1), (1, -1)])
def test_native_source_cache_requires_both_positive_parameters(slot_mb: int, slots: int) -> None:
    with pytest.raises(ValueError, match="positive slot MB and slot count"):
        _native_source_cache_transport_config(2 * MIB, slot_mb, slots, 0.0)


def test_native_source_cache_rejects_unaccounted_cpu_weight_registry() -> None:
    with pytest.raises(ValueError, match="unaccounted CPU weight registry"):
        _native_source_cache_transport_config(2 * MIB, 2, 1, 0.125)


def test_native_source_cache_settings_are_explicit_and_reported() -> None:
    settings = load_dynamic_offload_settings_from_env(
        running_on_wsl=False,
        environ={
            "DDO_VMM_NATIVE_TRANSPORT": "1",
            "DDO_VMM_NATIVE_SOURCE_CACHE_SLOT_MB": "2",
            "DDO_VMM_NATIVE_SOURCE_CACHE_SLOTS": "1",
        },
    )
    assert settings.config.vmm_native_source_cache_slot_mb == 2
    assert settings.config.vmm_native_source_cache_slots == 1
    assert settings.config.vmm_native_transport is True
    metrics = settings.as_metrics()
    assert metrics["dynamic_offload_vmm_native_source_cache_slot_mb"] == 2
    assert metrics["dynamic_offload_vmm_native_source_cache_slots"] == 1
    assert metrics["dynamic_offload_vmm_native_transport"] is True
