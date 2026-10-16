import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from diffusers_dynamic_offloader import add_dynamic_offload_lifecycle_arguments
from diffusers_dynamic_offloader import add_dynamic_offload_arguments, resolve_dynamic_offload_backend
from diffusers_dynamic_offloader import dynamic_offload as implementation


@pytest.mark.parametrize("phase", ["before-run", "after-text-encoder", "before-transformer", "after-run"])
def test_lifecycle_cli_is_tristate(phase):
    parser = add_dynamic_offload_lifecycle_arguments(argparse.ArgumentParser())
    dest = "purge_" + phase.replace("-", "_")
    assert getattr(parser.parse_args([]), dest) is None
    assert getattr(parser.parse_args(["--purge-" + phase]), dest) is True
    assert getattr(parser.parse_args(["--no-purge-" + phase]), dest) is False


@pytest.mark.parametrize("explicit,inherited", [(None, False), (None, True), (False, True), (True, False)])
def test_explicit_purge_precedes_policy_without_cuda_or_os_purge(explicit, inherited, monkeypatch):
    lookup = Mock(return_value=inherited)
    settings = SimpleNamespace(preset_bool=lookup)
    monkeypatch.setattr(implementation.os, "name", "posix")
    result = implementation.maybe_purge_windows_standby_cache(
        settings, "before_transformer", enabled=explicit, print_message=False,
    )
    expected = inherited if explicit is None else explicit
    assert result["reason"] == ("non_windows" if expected else "disabled")
    assert lookup.call_count == (1 if explicit is None else 0)


@pytest.mark.parametrize("preset,expected", [("auto", 1.5), ("nvme_baseline", .75)])
def test_total_staging_cli_does_not_change_legacy_defaults(preset, expected):
    parser = add_dynamic_offload_arguments(argparse.ArgumentParser())
    route = resolve_dynamic_offload_backend("vmm", preset)
    original = parser.parse_args([])
    assert route.build_config_from_args(original).vmm_staging_buffer_gb == .625
    args = parser.parse_args(["--ddo-vmm-total-staging-gb", "1.5"])
    assert route.build_config_from_args(args).vmm_staging_buffer_gb == expected


@pytest.mark.parametrize("total", ["0", "-1", "nan", "inf"])
def test_total_staging_cli_rejects_invalid_bounds(total):
    parser = add_dynamic_offload_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--ddo-vmm-total-staging-gb=" + total])
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_dynamic_offload_backend("vmm").build_config_from_args(args)
