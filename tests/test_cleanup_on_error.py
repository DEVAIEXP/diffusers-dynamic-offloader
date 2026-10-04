"""Public lifecycle error boundary; CPU mocks only, no allocator/CUDA calls."""

import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadApplyResult
from diffusers_dynamic_offloader import dynamic_offload as api


def result():
    return DynamicOffloadApplyResult(torch.nn.Linear(2, 2), hook=object())


def test_success_does_not_unload_or_add_a_completion_boundary(monkeypatch):
    offload = result()
    monkeypatch.setattr(api, "remove_dynamic_offload", lambda module: pytest.fail("unexpected cleanup"))
    with offload.cleanup_on_error() as entered:
        assert entered is offload
    assert offload.hook is not None


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failure_attempts_existing_cleanup_once_and_preserves_original(error_type, monkeypatch):
    offload = result()
    calls = []
    monkeypatch.setattr(api, "remove_dynamic_offload", calls.append)
    original = error_type("inference failed")
    with pytest.raises(error_type) as captured, offload.cleanup_on_error():
        raise original
    assert captured.value is original
    assert calls == [offload.module]


def test_cleanup_failure_preserves_original_and_hook_ownership(monkeypatch):
    offload = result()
    hook = offload.hook
    failure = RuntimeError("DMA completion unproven")

    def fail_cleanup(module):
        assert module is offload.module
        raise failure

    monkeypatch.setattr(api, "remove_dynamic_offload", fail_cleanup)
    original = ValueError("readback failed")
    with pytest.raises(ValueError) as captured, offload.cleanup_on_error():
        raise original
    assert captured.value is original
    assert offload.hook is hook
    if hasattr(original, "__notes__"):
        assert "DMA completion unproven" in original.__notes__[0]
    else:
        assert original.ddo_cleanup_error is failure


def test_unmanaged_result_does_not_remove_other_hooks(monkeypatch):
    offload = DynamicOffloadApplyResult(torch.nn.Linear(2, 2), hook=None)
    monkeypatch.setattr(api, "remove_dynamic_offload", lambda module: pytest.fail("unmanaged cleanup"))
    with pytest.raises(RuntimeError), offload.cleanup_on_error():
        raise RuntimeError("pipeline failed")


@pytest.mark.parametrize("legacy", [False, True])
def test_diagnostic_failure_or_missing_add_note_cannot_mask_original(legacy, monkeypatch):
    class ForwardError(RuntimeError):
        @property
        def add_note(self):
            if legacy:
                raise AttributeError("Python 3.10-style exception")
            return lambda note: (_ for _ in ()).throw(RuntimeError("note failed"))

    def fail_cleanup(module):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(api, "remove_dynamic_offload", fail_cleanup)
    original = ForwardError("original failure")
    with pytest.raises(ForwardError) as captured, result().cleanup_on_error():
        raise original
    assert captured.value is original
    if legacy:
        assert str(original.ddo_cleanup_error) == "cleanup failed"
