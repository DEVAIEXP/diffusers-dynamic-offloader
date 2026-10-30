"""Synthetic offline activity analysis: no CUDA or torch import."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "trace_analysis", Path(__file__).parents[1] / "scripts" / "analyze_cuda_trace.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(cat, start, end, *, device=0, stream=17, name="Memcpy HtoD", size=10):
    return dict(cat=cat, ts=start*1e6, dur=(end-start)*1e6, name=name,
                args=dict(device=device, stream=stream, bytes=size))


def test_union_overlap_and_unattributed_gaps():
    result = module.analyze([event("gpu_memcpy", 0, 2), event("gpu_memcpy", 1, 3),
                             event("gpu_memcpy", 5, 6), event("kernel", 2, 4, name="k"),
                             event("kernel", 0, 6, device=1)], 17)
    assert result["kernel_union_seconds"] == 2
    assert result["upload_union_seconds"] == 4
    assert result["kernel_upload_overlap_seconds"] == 1
    assert result["neither_recorded_activity_seconds"] == 1
    assert result["largest_neither_recorded_activity_gap_seconds"] == 1


def test_tail_clip_retains_total_payload_label_and_counts():
    result = module.analyze([event("gpu_memcpy", 0, 2), event("gpu_memcpy", 5, 6),
                             event("kernel", 2, 5, name="k")], 17, 2)
    assert result["window_seconds"] == 2
    assert result["kernel_union_seconds"] == 1
    assert result["upload_union_seconds"] == 1
    assert result["upload_stream_total_bytes"] == 20
    assert result["upload_stream_total_copies"] == 2


def test_no_matching_stream_and_ambiguous_device_rejected():
    with pytest.raises(ValueError, match="no HtoD"):
        module.analyze([], 17)
    with pytest.raises(ValueError, match="one explicit device"):
        module.analyze([event("gpu_memcpy", 0, 1), event("gpu_memcpy", 0, 1, device=1)], 17)


def test_nonpositive_tail_rejected():
    with pytest.raises(ValueError, match="positive"):
        module.analyze([event("gpu_memcpy", 0, 1)], 17, 0)


def test_cpu_operator_association_requires_explicit_external_id():
    kernel = event("kernel", 0, 1, name="gemm")
    kernel["args"]["External id"] = 42
    result = module.analyze([event("gpu_memcpy", 0, 2), kernel,
                            dict(cat="cpu_op", name="aten::addmm", args={"External id": 42}),
                            dict(cat="cpu_op", name="ignored", args={})], 17)
    assert result["top_cpu_operators_by_associated_kernel_duration"] == [
        dict(name="aten::addmm", kernel_count=1, summed_kernel_seconds=1)]


def test_host_api_intervals_are_clipped_and_nested_rows_not_added():
    result = module.analyze([event("gpu_memcpy", 0, 2),
                            event("cuda_runtime", 1, 4, name="cudaLaunchKernel"),
                            event("cuda_driver", 1.5, 3, name="cuLaunchKernel")], 17)
    assert result["top_host_apis_by_union_duration"] == [
        dict(category="cuda_runtime", name="cudaLaunchKernel", count=1,
             union_seconds=1, max_clipped_seconds=1),
        dict(category="cuda_driver", name="cuLaunchKernel", count=1,
             union_seconds=0.5, max_clipped_seconds=0.5)]
