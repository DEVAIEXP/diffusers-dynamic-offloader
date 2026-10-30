"""Platform/dependency contracts only; never import Torch or initialize CUDA."""

from pathlib import Path

import pytest
from packaging.requirements import Requirement

tomllib = pytest.importorskip("tomllib", reason="Linux development profile uses Python 3.12")

ROOT = Path(__file__).resolve().parents[1]


def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def selected(extra, platform):
    env = {"sys_platform": platform, "python_version": "3.12", "python_full_version": "3.12.14"}
    result = {}
    for text in project()["project"]["optional-dependencies"][extra]:
        requirement = Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate(env):
            result[requirement.name] = requirement
    return result


def test_linux_and_windows_profiles_preserve_matching_torch_pins():
    linux = selected("vmm-dev-linux", "linux")
    windows = selected("vmm-dev", "win32")
    for name in ("torch", "torchvision"):
        assert str(linux[name].specifier) == str(windows[name].specifier)
    assert {"nvcc", "cccl"}.issubset(linux["cuda-toolkit"].extras)
    assert str(linux["cuda-toolkit"].specifier) == "==13.2.1"


def test_linux_only_compiler_and_wheels_do_not_apply_to_windows():
    windows = selected("vmm-dev-linux", "win32")
    assert not {"torch", "torchvision", "cuda-toolkit"}.intersection(windows)
    assert "cuda-toolkit" not in selected("vmm-dev", "win32")


def test_normal_install_does_not_require_native_development_dependencies():
    names = {Requirement(text).name for text in project()["project"]["dependencies"]}
    assert not {"cuda-toolkit", "cuda-bindings", "cuda-core", "pytest", "ruff"}.intersection(names)


def test_linux_index_selection_is_scoped_to_opt_in_profile():
    for name in ("torch", "torchvision"):
        entries = project()["tool"]["uv"]["sources"][name]
        linux = [entry for entry in entries if "linux" in entry["marker"]]
        assert len(linux) == 1
        assert linux[0]["extra"] == "vmm-dev-linux"
        assert linux[0]["index"] == "pytorch-cu132"
        windows = [entry for entry in entries if "win32" in entry["marker"]]
        assert len(windows) == 1 and "extra" not in windows[0]
