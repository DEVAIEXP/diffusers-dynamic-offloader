"""CPU-only native DLL selection; no CUDA calls or actual DLL loading."""
import sys

import pytest

from diffusers_dynamic_offloader import vmm_model_arena
from ddo_vmm_allocator import native


@pytest.mark.parametrize("explicit, environment, expected", [
    ("explicit.dll", "candidate.dll", "explicit.dll"),
    (None, "candidate.dll", "candidate.dll"),
    (None, None, "default"),
])
def test_library_precedence_matches_bootstrap(monkeypatch, tmp_path, explicit, environment, expected):
    root = tmp_path / "allocator"
    default = root / ("build/Release/ddo_vmm.dll" if sys.platform == "win32" else "build-linux/libddo_vmm.so")
    default.parent.mkdir(parents=True)
    default.touch()
    monkeypatch.setattr(vmm_model_arena, "_allocator_source_root", lambda: root / "src")
    if environment is None:
        monkeypatch.delenv("DDO_VMM_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("DDO_VMM_LIBRARY_PATH", environment)
    selected = []
    sentinel = object()
    monkeypatch.setattr(native, "load_native_vmm", lambda path: selected.append(path) or sentinel)
    result, _ = vmm_model_arena._load_native_allocator(explicit)
    assert result is sentinel
    assert selected == [str(default) if expected == "default" else expected]


def test_invalid_selected_library_must_not_silently_fall_back(monkeypatch, tmp_path):
    root = tmp_path / "allocator"
    default = root / "build" / "Release" / "ddo_vmm.dll"
    default.parent.mkdir(parents=True)
    default.touch()
    monkeypatch.setattr(vmm_model_arena, "_allocator_source_root", lambda: root / "src")
    monkeypatch.setenv("DDO_VMM_LIBRARY_PATH", "missing-candidate.dll")
    def load(path):
        assert path == "missing-candidate.dll"
        raise FileNotFoundError(path)
    monkeypatch.setattr(native, "load_native_vmm", load)
    with pytest.raises(FileNotFoundError, match="missing-candidate"):
        vmm_model_arena._load_native_allocator(None)


def test_no_local_library_delegates_missing_path_to_native_binding(monkeypatch, tmp_path):
    monkeypatch.delenv("DDO_VMM_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(vmm_model_arena, "_allocator_source_root", lambda: tmp_path / "src")
    monkeypatch.setattr(vmm_model_arena, "_installed_allocator_root", lambda: None)
    selected = []
    monkeypatch.setattr(native, "load_native_vmm", lambda path: selected.append(path))
    vmm_model_arena._load_native_allocator(None)
    assert selected == [None]


@pytest.mark.parametrize("platform, relative_path", [
    ("win32", "native/win32/ddo_vmm.dll"),
    ("linux", "native/linux/libddo_vmm.so"),
])
def test_arena_discovers_installed_library(monkeypatch, tmp_path, platform, relative_path):
    package = tmp_path / "ddo_vmm_allocator"
    library = package / relative_path
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.delenv("DDO_VMM_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(vmm_model_arena, "_allocator_source_root", lambda: tmp_path / "checkout/src")
    monkeypatch.setattr(vmm_model_arena, "_installed_allocator_root", lambda: package)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(native, "load_native_vmm", lambda path: path)
    selected, _ = vmm_model_arena._load_native_allocator(None)
    assert selected == str(library)
