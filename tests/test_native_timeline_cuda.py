"""Prepared CUDA contracts. Execute only after explicit approval."""
import os
import ctypes

import pytest
import torch

from ddo_vmm_allocator.native import load_native_vmm
from ddo_vmm_allocator.timeline import NativeTimeline
from diffusers_dynamic_offloader import DynamicOffloadConfig, apply_dynamic_offload, remove_dynamic_offload
from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadApplyResult
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorFileSlice, SafetensorSliceRegistry

pytestmark = [pytest.mark.vmm, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_native_timeline_distinct_stream_dependencies_and_pending_owner():
    compute, upload = torch.cuda.Stream(), torch.cuda.Stream()
    cpu = torch.full((16384,), 7., pin_memory=True)
    gpu = torch.empty_like(cpu, device="cuda")
    library = load_native_vmm(os.environ["DDO_VMM_LIBRARY_PATH"])
    with torch.cuda.stream(compute):
        timeline = NativeTimeline(library, capacity=16, stream=compute.cuda_stream)
        try:
            fn = timeline._functions
            first = fn["mark_begin"](1, 10, cpu.numel() * cpu.element_size(), upload.cuda_stream)
            with torch.cuda.stream(upload):
                gpu.copy_(cpu, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(upload)
            fn["mark_end"](first, upload.cuda_stream)
            dependency = fn["mark_begin"](2, 10, 0, compute.cuda_stream)
            compute.wait_event(ready)
            fn["mark_end"](dependency, compute.cuda_stream)
            unit = timeline.begin_unit(1, compute.cuda_stream)
            gpu.add_(1)
            consumed = torch.cuda.Event()
            consumed.record(compute)
            recycle = fn["mark_begin"](3, 20, 0, upload.cuda_stream)
            upload.wait_event(consumed)
            fn["mark_end"](recycle, upload.cuda_stream)
            timeline.seal()
            # An unended scope deterministically retains ownership regardless
            # of GPU speed or host scheduling between these two queries.
            assert timeline.snapshot()["status"] == "pending"
            assert not timeline.close()  # No waiting, native owner preserved.
            timeline.end_unit(unit, compute.cuda_stream)
            compute.synchronize()
            upload.synchronize()  # Explicit test boundary, not recorder behavior.
            report = timeline.snapshot()
            assert report["status"] == "complete" and report["event_cleanup"] == "released"
            assert {s["kind"] for s in report["samples"]} == {
                "upload_batch_window", "compute_upload_dependency", "execution_unit_window", "upload_slot_dependency"}
            assert {s["stream"] for s in report["samples"]} == {compute.cuda_stream, upload.cuda_stream}
            assert all(s["end_seconds"] >= s["start_seconds"] for s in report["samples"])
            assert torch.equal(gpu.cpu(), torch.full_like(cpu, 8.))
        finally:
            compute.synchronize()
            upload.synchronize()
            timeline.close()


def test_native_timeline_capacity_and_stale_token_cannot_overwrite_new_session():
    stream = torch.cuda.current_stream()
    library = load_native_vmm(os.environ["DDO_VMM_LIBRARY_PATH"])
    first = NativeTimeline(library, capacity=1, stream=stream.cuda_stream)
    try:
        old = first.begin_unit(1, stream.cuda_stream)
        first.end_unit(old, stream.cuda_stream)
        assert first.begin_unit(2, stream.cuda_stream) == 0
        first.seal()
        stream.synchronize()
        assert first.snapshot()["status"] == "truncated"
    finally:
        stream.synchronize()
        first.close()
    second = NativeTimeline(library, capacity=1, stream=stream.cuda_stream)
    try:
        token = second.begin_unit(3, stream.cuda_stream)
        assert token != old
        second.end_unit(old, stream.cuda_stream)
        second.end_unit(token, stream.cuda_stream)
        second.seal()
        stream.synchronize()
        report = second.snapshot()
        assert report["status"] == "complete" and report["samples"][0]["unit"] == 3
    finally:
        stream.synchronize()
        second.close()


def test_public_native_scope_observes_source_batch_and_unit_without_output_change(tmp_path):
    class DenseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(512, 512, bias=False) for _ in range(8)])

        def forward(self, value):
            for layer in self.layers:
                value = layer(value)
            return value

    torch.manual_seed(43)
    model = DenseModel().eval()
    value = torch.randn(2, 512)
    with torch.no_grad():
        expected = model(value)
    shard = tmp_path / "weights.bin"
    slices = SafetensorSliceRegistry()
    payload = bytearray()
    for parameter in model.parameters():
        source = parameter.detach()
        size = source.numel() * source.element_size()
        assert slices.bind(source, SafetensorFileSlice(shard, len(payload), size))
        payload.extend(source.view(torch.uint8).numpy().tobytes())
    shard.write_bytes(payload)
    model._ddo_safetensor_slices = slices
    hook = apply_dynamic_offload(model, DynamicOffloadConfig(
        execution_mode="vmm_fast_runtime", vmm_native_transport=True,
        vmm_module_pattern=r"^layers\.\d+$",
        vmm_resident_weight_budget_gb=6/1024, vmm_page_mb=1,
        vmm_staging_buffer_gb=2/1024, vmm_source_cache_gb=64/1024))
    result = DynamicOffloadApplyResult(module=model, hook=hook)
    timer = result.execution_timer(enabled=True, native_units=True, native_capacity=128)
    try:
        with torch.no_grad(), timer:
            actual = model(value.cuda())
            timer.mark_step()
        actual = actual.cpu()
        report = result.metrics_snapshot(collect_completed=True, execution_timing=timer)
        assert report.runtime["vmm"]["fast_source_cache_batch_submissions"] > 0
        assert report.runtime["vmm"]["fast_execution_buffer_uploads"] > 0
        timeline = report.runtime["native_timeline"]
        assert timeline["status"] == "complete" and timeline["event_cleanup"] == "released"
        kinds = {s["kind"] for s in timeline["samples"]}
        assert {"upload_batch_window", "compute_upload_dependency", "execution_unit_window", "step_marker"} <= kinds
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
        assert getattr(hook._vmm_fast_store, "_timeline", None) is None
    finally:
        torch.cuda.synchronize()
        if timer._native is not None:
            timer._native.close()
        remove_dynamic_offload(model)
