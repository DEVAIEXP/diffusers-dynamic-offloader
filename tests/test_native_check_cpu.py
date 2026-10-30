"""Check CLI and bootstrap ordering without importing Torch or executing CUDA."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ddo_native_check as check


def test_import_does_not_import_cuda_dependencies():
    code = (
        "import sys, ddo_native_check; "
        "assert not any(n in sys.modules for n in "
        "('torch', 'ddo_vmm_allocator', 'diffusers_dynamic_offloader'))"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


@pytest.mark.parametrize("arguments,code", [([], 2), (["--help"], 0), (["--unknown"], 2)])
def test_no_gpu_without_explicit_flag(monkeypatch, arguments, code):
    def forbidden(**kwargs):
        pytest.fail("bootstrap must not run")

    monkeypatch.setitem(sys.modules, "ddo_vmm_bootstrap", SimpleNamespace(ensure_vmm_fast_process=forbidden))
    monkeypatch.setattr(check, "_run_cuda", forbidden)
    with pytest.raises(SystemExit) as error:
        check.main(arguments)
    assert error.value.code == code


def test_child_runs_check_only_after_bootstrap(monkeypatch):
    calls = []

    def bootstrap(**kwargs):
        assert kwargs == {"allocator_cache_mib": 32, "argv": [check.__file__, "--run-cuda"]}
        calls.append("bootstrap")

    monkeypatch.setitem(sys.modules, "ddo_vmm_bootstrap", SimpleNamespace(ensure_vmm_fast_process=bootstrap))
    monkeypatch.setattr(check, "_run_cuda", lambda: calls.append("cuda"))
    check.main(["--run-cuda"])
    assert calls == ["bootstrap", "cuda"]


def test_parent_exit_does_not_run_cuda(monkeypatch):
    def bootstrap(**kwargs):
        raise SystemExit(7)

    monkeypatch.setitem(sys.modules, "ddo_vmm_bootstrap", SimpleNamespace(ensure_vmm_fast_process=bootstrap))
    monkeypatch.setattr(check, "_run_cuda", lambda: pytest.fail("parent must not execute CUDA"))
    with pytest.raises(SystemExit) as error:
        check.main(["--run-cuda"])
    assert error.value.code == 7


def test_check_is_in_wheel_module_list():
    import tomllib

    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert "ddo_native_check" in metadata["tool"]["setuptools"]["py-modules"]
