"""Prepared small CUDA timing contract; run only after explicit approval."""
import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadExecutionTimer


@pytest.mark.vmm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compute_stream_markers_remain_pending_without_step_waits():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        value = torch.empty(64, device="cuda")
        with DynamicOffloadExecutionTimer(enabled=True) as timer:
            value.fill_(1)
            timer.mark_step()
            torch.cuda._sleep(250_000_000)
            value.add_(2)
            timer.mark_step()
        assert timer.snapshot()["status"] == "pending"
        stream.synchronize()  # Test boundary, never part of the timer's steps.
        report = timer.snapshot()
        assert report["status"] == "complete"
        assert len(report["step_seconds"]) == 2
        assert report["stream"] == int(stream.cuda_stream)
        assert report["phase_seconds"] > 0
        assert report["phase_seconds"] == pytest.approx(
            sum(report["step_seconds"]) + report["tail_seconds"], abs=0.00001)
        assert torch.equal(value.cpu(), torch.full((64,), 3.0))


@pytest.mark.vmm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_diagnostic_step_wait_proves_boundary_completion_and_output():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        value = torch.zeros(64, device="cuda")
        with DynamicOffloadExecutionTimer(enabled=True, diagnostic_step_wait=True) as timer:
            value.add_(1)
            timer.mark_step()
            assert timer._events[-1].query()
            value.add_(2)
            timer.mark_step()
            assert timer._events[-1].query()
        stream.synchronize()  # Explicit final test boundary only.
        report = timer.snapshot()
        assert report["status"] == "complete" and len(report["step_seconds"]) == 2
        diagnostic = report["diagnostic_step_wait"]
        assert diagnostic["calls"] == 2 and diagnostic["benchmark_eligible"] is False
        assert torch.equal(value.cpu(), torch.full((64,), 3.0))
