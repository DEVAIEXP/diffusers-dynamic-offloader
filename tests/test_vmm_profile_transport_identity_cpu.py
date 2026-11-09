"""The weight transport capacities belong to a VMM residency profile identity (CPU)."""
from dataclasses import replace

from torch import nn

from diffusers_dynamic_offloader import DynamicOffloadConfig
from diffusers_dynamic_offloader.profiles import VMM_PROFILE_IDENTITY_VERSION, _NamedProfile


def context(**overrides):
    config = replace(DynamicOffloadConfig(), execution_mode="vmm_fast_runtime", execution_device="cuda",
                     profile_name="p", **overrides)
    return _NamedProfile(nn.Linear(2, 2), config, component="transformer").context


def test_transport_capacities_are_part_of_the_identity():
    base = context(vmm_native_transport=True, vmm_staging_buffer_gb=1.5,
                   vmm_source_cache_gb=24.625, vmm_source_registered_gb=24.625)
    assert base["identity_version"] == VMM_PROFILE_IDENTITY_VERSION == 5
    assert base["transport"] == {"native": True, "staging_gb": 1.5, "source_cache_gb": 24.625,
                                 "source_registered_gb": 24.625, "host_cache_gb": base["transport"]["host_cache_gb"]}
    same = context(vmm_native_transport=True, vmm_staging_buffer_gb=1.5,
                   vmm_source_cache_gb=24.625, vmm_source_registered_gb=24.625)
    assert same == base
    for change in ({"vmm_source_cache_gb": 0.0, "vmm_source_registered_gb": 0.0},
                   {"vmm_staging_buffer_gb": 0.625}, {"vmm_native_transport": False}):
        settings = dict(vmm_native_transport=True, vmm_staging_buffer_gb=1.5,
                        vmm_source_cache_gb=24.625, vmm_source_registered_gb=24.625)
        settings.update(change)
        assert context(**settings)["transport"] != base["transport"], change
