"""CPU DLL compatibility contracts; fake exports must never be called."""
from types import SimpleNamespace
import builtins

import pytest

from ddo_vmm_allocator import _library, native, torch_allocator


class Export:
    def __call__(self, *args):
        raise AssertionError("preflight must not call native exports")


class Library:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.exports = {}

    def __getattr__(self, name):
        if name in self.missing:
            raise AttributeError(f"missing symbol: {name}")
        return self.exports.setdefault(name, Export())


def fake_loader(monkeypatch, value):
    monkeypatch.setattr(_library.platform, "system", lambda: "Windows")
    monkeypatch.setattr(_library.ctypes, "WinDLL", lambda path: value)


def test_missing_required_export_reports_selected_path_without_cuda(monkeypatch, tmp_path):
    path = tmp_path / "selected.dll"
    path.touch()
    fake_loader(monkeypatch, Library(["ddo_vmm_region_create_ex"]))
    with pytest.raises(native.NativeVmmError) as caught:
        native.NativeVmmLibrary(path)
    assert str(path.resolve()) in str(caught.value)
    assert "ddo_vmm_region_create_ex" in str(caught.value)
    assert "not be replaced automatically" in str(caught.value)
    assert isinstance(caught.value.__cause__, AttributeError)


def test_optional_exports_remain_optional(monkeypatch, tmp_path):
    path = tmp_path / "legacy.dll"
    path.touch()
    fake_loader(monkeypatch, Library([
        "ddo_vmm_model_arena_enable_streaming", "ddo_vmm_model_arena_record_compute",
        "ddo_vmm_model_arena_wait_compute", "ddo_vmm_model_arena_drain_compute",
        "ddo_vmm_torch_pool_stats_v1",
    ]))
    library = native.NativeVmmLibrary(path)
    assert library.path == path.resolve()
    assert library.torch_pool_stats() == {}


def test_load_dependency_failure_retains_original_error(monkeypatch, tmp_path):
    path = tmp_path / "selected.dll"
    path.touch()
    original = OSError("dependent DLL unavailable")
    def fail(path):
        raise original
    monkeypatch.setattr(_library.platform, "system", lambda: "Windows")
    monkeypatch.setattr(_library.ctypes, "WinDLL", fail)
    with pytest.raises(native.NativeVmmError) as caught:
        native.NativeVmmLibrary(path)
    assert caught.value.__cause__ is original
    assert str(path.resolve()) in str(caught.value)


def test_bootstrap_missing_callbacks_rejected_before_torch_import(monkeypatch, tmp_path):
    path = tmp_path / "selected.dll"
    path.touch()
    monkeypatch.setattr(torch_allocator, "_installed_allocator", None)
    fake_loader(monkeypatch, SimpleNamespace())
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert name != "torch", "invalid DLL must be rejected before importing Torch"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(torch_allocator.TorchAllocatorInstallError) as caught:
        torch_allocator.install_pytorch_vmm_allocator(path)
    assert "ddo_vmm_torch_malloc, ddo_vmm_torch_free" in str(caught.value)
    assert str(path.resolve()) in str(caught.value)
    assert torch_allocator._installed_allocator is None
