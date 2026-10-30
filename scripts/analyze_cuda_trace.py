"""Offline Chrome/CUDA trace geometry; no CUDA imports or inference execution.

Intervals with neither a recorded kernel nor memcpy are not proof of CPU
starvation: dependencies, missing activities and other engines may contribute.
"""
import argparse
import collections
import json
from pathlib import Path


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(events, upload_stream, last_seconds=None):
    copies = [e for e in events if e.get("cat") == "gpu_memcpy"
              and "HtoD" in e.get("name", "")
              and e.get("args", {}).get("stream") == upload_stream
              and e.get("dur", 0) > 0]
    if not copies:
        raise ValueError("no HtoD activities for the selected upload stream")
    devices = {e.get("args", {}).get("device") for e in copies}
    if len(devices) != 1 or None in devices:
        raise ValueError("upload stream must identify one explicit device")
    device = next(iter(devices))
    start = min(e["ts"] for e in copies)
    end = max(e["ts"] + e["dur"] for e in copies)
    if last_seconds is not None:
        if last_seconds <= 0:
            raise ValueError("last-seconds must be positive")
        start = max(start, end - last_seconds * 1e6)

    def clipped(event):
        return max(start, event["ts"]), min(end, event["ts"] + event["dur"])

    kernels = [e for e in events if e.get("cat") == "kernel"
               and e.get("args", {}).get("device") == device
               and e.get("dur", 0) > 0 and e["ts"] < end
               and e["ts"] + e["dur"] > start]
    copy_windows = union(clipped(e) for e in copies)
    kernel_windows = union(clipped(e) for e in kernels)
    activities = union(copy_windows + kernel_windows)
    gaps = []
    cursor = start
    for a, b in activities:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = b
    if cursor < end:
        gaps.append((cursor, end))
    names = collections.defaultdict(lambda: [0, 0.0])
    cpu_names = {e["args"]["External id"]: e.get("name", "unknown")
                 for e in events if e.get("cat") == "cpu_op"
                 and e.get("args", {}).get("External id") is not None}
    operators = collections.defaultdict(lambda: [0, 0.0])
    host_apis = collections.defaultdict(list)
    for e in events:
        if (e.get("cat") in ("cuda_runtime", "cuda_driver")
                and e.get("dur", 0) > 0 and e["ts"] < end
                and e["ts"] + e["dur"] > start):
            host_apis[(e["cat"], e["name"])].append(clipped(e))
    for e in kernels:
        a, b = clipped(e)
        names[e["name"]][0] += 1
        names[e["name"]][1] += b - a
        operator = cpu_names.get(e.get("args", {}).get("External id"), "unmatched")
        operators[operator][0] += 1
        operators[operator][1] += b - a
    return {
        "scope": "recorded_gpu_activity_geometry_not_causal_attribution",
        "profiled_run_not_benchmark": True,
        "device": device, "upload_stream": upload_stream,
        "window_seconds": (end - start) / 1e6,
        "upload_stream_total_bytes": sum(e.get("args", {}).get("bytes", 0) for e in copies),
        "upload_stream_total_copies": len(copies),
        "kernel_union_seconds": duration(kernel_windows) / 1e6,
        "upload_union_seconds": duration(copy_windows) / 1e6,
        "kernel_upload_overlap_seconds": intersection(kernel_windows, copy_windows) / 1e6,
        "neither_recorded_activity_seconds": duration(gaps) / 1e6,
        "largest_neither_recorded_activity_gap_seconds": max((b-a for a,b in gaps), default=0) / 1e6,
        "top_kernel_names_by_summed_duration": [
            {"name": name, "count": values[0], "summed_seconds": values[1] / 1e6}
            for name, values in sorted(names.items(), key=lambda item: item[1][1], reverse=True)[:10]],
        "top_cpu_operators_by_associated_kernel_duration": [
            {"name": name, "kernel_count": values[0], "summed_kernel_seconds": values[1] / 1e6}
            for name, values in sorted(operators.items(), key=lambda item: item[1][1], reverse=True)[:10]],
        # CUDA driver/runtime calls may nest. Never add different API rows as
        # exclusive CPU time or interpret a long launch as GPU kernel duration.
        "host_api_scope": "clipped_host_call_intervals_not_exclusive_time_or_cause",
        "top_host_apis_by_union_duration": sorted([
            {"category": category, "name": name, "count": len(spans),
             "union_seconds": duration(union(spans)) / 1e6,
             "max_clipped_seconds": max(b-a for a,b in spans) / 1e6}
            for (category, name), spans in host_apis.items()],
            key=lambda row: row["union_seconds"], reverse=True)[:12],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--upload-stream", type=int, required=True)
    parser.add_argument("--last-seconds", type=float)
    args = parser.parse_args()
    with args.trace.open(encoding="utf-8") as source:
        trace = json.load(source)
    print(json.dumps(analyze(trace["traceEvents"], args.upload_stream, args.last_seconds), indent=2))


if __name__ == "__main__":
    main()
