"""CPU-only validation of the fresh-process allocator diagnostic options."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).parents[1] / "ddo-vmm-allocator/scripts/vmm_torch_allocator_probe.py"
    spec = importlib.util.spec_from_file_location("allocator_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("option", ["verify_stream_lifetime", "verify_nonblocking_reuse"])
def test_reuse_diagnostic_rejects_insufficient_cache_before_cuda(probe, option):
    args = SimpleNamespace(megabytes=16, cache_mib=0,
                           verify_stream_lifetime=False, verify_nonblocking_reuse=False)
    setattr(args, option, True)
    with pytest.raises(ValueError, match="cache capacity"):
        probe.run(args)


def test_parse_nonblocking_probe_options(probe, monkeypatch):
    monkeypatch.setattr("sys.argv", ["probe", "--library", "candidate.dll",
                                    "--megabytes", "16", "--cache-mib", "128",
                                    "--verify-stream-lifetime", "--verify-nonblocking-reuse"])
    args = probe.parse_args()
    assert args.verify_stream_lifetime and args.verify_nonblocking_reuse
    assert args.megabytes == 16 and args.cache_mib == 128
