"""Read-only Nsight SQLite dependency evidence, never CUDA execution.

Recorded predecessor activities are only a lower bound on event completion.
Their end-to-copy gap is NOT measured event wait time or causal attribution.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import median


def identity(row):
    return row["globalPid"], row["deviceId"], row["contextId"], row["streamId"]


def summarize_gaps(values):
    return {"count": len(values),
            "median_ms": median(values) / 1e6 if values else None,
            "min_ms": min(values) / 1e6 if values else None,
            "max_ms": max(values) / 1e6 if values else None}


def analyze(connection, upload_stream, last_seconds=10):
    if last_seconds <= 0:
        raise ValueError("last-seconds must be positive")
    connection.row_factory = sqlite3.Row
    names = dict(connection.execute("SELECT id, value FROM StringIds"))
    api = defaultdict(list)
    for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        api[(row["globalTid"] >> 24, row["correlationId"])].append(row)

    def host(row, prefix=None):
        matches = api.get((row["globalPid"] >> 24, row["correlationId"]), [])
        if prefix:
            matches = [r for r in matches if names[r["nameId"]].startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    events = defaultdict(list)
    for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT"):
        if row["eventSyncId"]:
            key = row["globalPid"], row["deviceId"], row["contextId"], row["eventSyncId"]
            events[key].append(row)
    activities = defaultdict(list)
    copies = defaultdict(list)
    for table in ("KERNEL", "MEMCPY", "MEMSET"):
        for row in connection.execute(f"SELECT * FROM CUPTI_ACTIVITY_KIND_{table}"):
            call = host(row)
            if call is None:
                continue
            activities[identity(row)].append((call["end"], row["end"]))
            if table == "MEMCPY" and row["streamId"] == upload_stream and row["copyKind"] == 1:
                copies[identity(row)].append((call["start"], row["start"], row["end"]))
    if len(copies) != 1:
        raise ValueError("upload stream must identify one process/device/context with H2D copies")
    target, copy_rows = next(iter(copies.items()))
    end = max(row[2] for row in copy_rows)
    start = end - last_seconds * 1e9
    copy_rows.sort()
    copy_starts = [row[0] for row in copy_rows]
    indexed = {}
    for key, rows in activities.items():
        rows.sort()
        ends, maximum = [], 0
        for _, device_end in rows:
            maximum = max(maximum, device_end)
            ends.append(maximum)
        indexed[key] = ([r[0] for r in rows], ends)
    edges = Counter()
    gaps = []
    completion_gaps = []
    device_completions = 0
    missing = Counter()
    upload_waits = []
    for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE syncType=2"):
        call = host(row, "cuStreamWaitEvent")
        if call is None:
            continue
        key = row["globalPid"], row["deviceId"], row["contextId"], row["eventSyncId"]
        records = events.get(key, [])
        if identity(row) == target:
            completion = None
            if len(records) == 1:
                producer = records[0]
                producer_call = host(producer, "cuEventRecord")
                timestamp = producer["timestamp"] if "timestamp" in producer.keys() else 0
                if producer_call is not None and producer_call["end"] <= call["start"] and timestamp and timestamp > 0:
                    completion = timestamp
            upload_waits.append((call["end"], completion))
        if not (start <= call["start"] < end):
            continue
        if len(records) != 1:
            missing["unknown_or_ambiguous_event"] += 1
            continue
        record = records[0]
        record_call = host(record, "cuEventRecord")
        if record_call is None or record_call["end"] > call["start"]:
            missing["unknown_or_unordered_record"] += 1
            continue
        edges[f'{record["streamId"]}->{row["streamId"]}'] += 1
        timestamp = record["timestamp"] if "timestamp" in record.keys() else 0
        if timestamp and timestamp > 0:
            device_completions += 1
        if identity(row) != target or record["streamId"] == upload_stream:
            continue
        times, bounds = indexed.get(identity(record), ([], []))
        prior = bisect_right(times, record_call["start"]) - 1
        following = bisect_left(copy_starts, call["end"])
        if prior < 0 or following >= len(copy_rows):
            missing["missing_predecessor_or_following_copy"] += 1
            continue
        # This is a geometric gap, not device event completion latency.
        gaps.append(copy_rows[following][1] - bounds[prior])
        if timestamp and timestamp > 0:
            completion_gaps.append(copy_rows[following][1] - timestamp)
    upload_waits.sort(key=lambda item: item[0])
    wait_times, wait_bounds, wait_unknown = [], [], []
    maximum, unknown = 0, 0
    for host_end, completion in upload_waits:
        maximum = max(maximum, completion or 0)
        unknown += completion is None
        wait_times.append(host_end)
        wait_bounds.append(maximum)
        wait_unknown.append(unknown)
    activity_times, activity_bounds = indexed.get(target, ([], []))
    residuals, rejected = [], 0
    for host_start, gpu_start, _ in copy_rows:
        if not start <= gpu_start < end:
            continue
        wait = bisect_right(wait_times, host_start) - 1
        if wait >= 0 and wait_unknown[wait]:
            rejected += 1
            continue
        preceding = bisect_right(activity_times, host_start) - 1
        bound = max(wait_bounds[wait] if wait >= 0 else 0,
                    activity_bounds[preceding] if preceding >= 0 else 0,
                    host_start)
        residuals.append(gpu_start - bound)
    return {
        "scope": "recorded_predecessor_lower_bounds_not_event_completion_or_cause",
        "profiled_run_not_benchmark": True,
        "window_seconds": last_seconds,
        "upload_identity": list(target),
        "matched_stream_wait_edges": dict(edges),
        "unmatched": dict(missing),
        "compute_predecessor_to_following_copy_gap": summarize_gaps(gaps),
        "negative_gaps": sum(v < 0 for v in gaps),
        "matched_waits_with_device_completion": device_completions,
        "event_completion_to_following_copy_gap": summarize_gaps(completion_gaps),
        "negative_completion_gaps": sum(v < 0 for v in completion_gaps),
        "all_recorded_prior_upload_dependencies_to_copy_gap": summarize_gaps(residuals),
        "negative_prior_dependency_gaps": sum(v < 0 for v in residuals),
        "copies_rejected_for_missing_prior_completion": rejected,
        "eligibility_caveat": "Recorded dependencies only: API start is not submission completion; hidden dependencies and other engines remain possible.",
        "caveat": "A nonnegative gap does not prove the event had completed; timestamps may be missing.",
        "completion_caveat": "Nonzero event timestamps apply only to this instrumented run; tracing can alter dependencies.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--upload-stream", type=int, required=True)
    parser.add_argument("--last-seconds", type=float, default=10)
    args = parser.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        print(json.dumps(analyze(connection, args.upload_stream, args.last_seconds), indent=2))


if __name__ == "__main__":
    main()
