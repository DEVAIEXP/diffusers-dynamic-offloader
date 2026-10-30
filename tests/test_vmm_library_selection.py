"""CPU-only native DLL selection; no CUDA calls or actual DLL loading."""
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
    default = root / "build" / "Release" / "ddo_vmm.dll"
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
    selected = []
    monkeypatch.setattr(native, "load_native_vmm", lambda path: selected.append(path))
    vmm_model_arena._load_native_allocator(None)
    assert selected == [None]
