"""Synthetic SQLite contracts; no GPU libraries or hardware required."""
import importlib.util
from pathlib import Path
import sqlite3

import pytest


spec = importlib.util.spec_from_file_location(
    "cuda_dependencies", Path(__file__).parents[1] / "scripts" / "analyze_cuda_dependencies.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def database(producer_end=140, event_sync_id=9):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
    db.executemany("INSERT INTO StringIds VALUES (?,?)", [
        (1, "cuLaunchKernel"), (2, "cuEventRecord"),
        (3, "cuStreamWaitEvent"), (4, "cuMemcpyHtoDAsync")])
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId,nameId)")
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", [
        (1, 2, 1 << 24, 1, 1), (3, 4, 1 << 24, 2, 2),
        (5, 6, 1 << 24, 3, 3), (7, 8, 1 << 24, 4, 4)])
    fields = "start,end,globalPid,deviceId,contextId,streamId,correlationId"
    for table in ("KERNEL", "MEMCPY", "MEMSET"):
        suffix = ",copyKind" if table == "MEMCPY" else ""
        db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_{table}({fields}{suffix})")
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
               (100, producer_end, 1 << 24, 0, 1, 7, 1))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?,?)",
               (150, 160, 1 << 24, 0, 1, 17, 4, 1))
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT(globalPid,deviceId,contextId,streamId,correlationId,eventSyncId)")
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_CUDA_EVENT VALUES (?,?,?,?,?,?)",
               (1 << 24, 0, 1, 7, 2, event_sync_id))
    db.execute(f"CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION({fields},syncType,eventSyncId)")
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (?,?,?,?,?,?,?,?,?)",
               (5, 6, 1 << 24, 0, 1, 17, 3, 2, event_sync_id))
    return db


def test_known_dependency_is_geometry_not_event_wait():
    with database() as db:
        result = module.analyze(db, 17)
    assert result["matched_stream_wait_edges"] == {"7->17": 1}
    assert result["compute_predecessor_to_following_copy_gap"]["median_ms"] == 10 / 1e6
    assert result["negative_gaps"] == 0
    assert "not_event_completion" in result["scope"]


def test_negative_gap_is_not_hidden():
    with database(producer_end=170) as db:
        result = module.analyze(db, 17)
    assert result["negative_gaps"] == 1


def test_zero_event_identity_is_unknown():
    with database(event_sync_id=0) as db:
        result = module.analyze(db, 17)
    assert result["matched_stream_wait_edges"] == {}
    assert result["unmatched"] == {"unknown_or_ambiguous_event": 1}


def test_ambiguous_record_is_not_silently_selected():
    with database() as db:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_CUDA_EVENT SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT")
        result = module.analyze(db, 17)
    assert result["matched_stream_wait_edges"] == {}
    assert result["unmatched"] == {"unknown_or_ambiguous_event": 1}


def test_event_from_another_process_is_not_matched():
    with database() as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_CUDA_EVENT SET globalPid=?", (2 << 24,))
        result = module.analyze(db, 17)
    assert result["matched_stream_wait_edges"] == {}


def test_event_recorded_after_wait_is_not_matched():
    with database() as db:
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET end=10 WHERE correlationId=2")
        result = module.analyze(db, 17)
    assert result["unmatched"] == {"unknown_or_unordered_record": 1}


@pytest.mark.parametrize("seconds", [0, -1])
def test_positive_window_required(seconds):
    with database() as db, pytest.raises(ValueError, match="positive"):
        module.analyze(db, 17, seconds)


def test_completed_event_gap_separate_from_predecessor_bound():
    with database() as db:
        db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT ADD COLUMN timestamp DEFAULT 0")
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_CUDA_EVENT SET timestamp=145")
        result = module.analyze(db, 17)
    assert result["matched_waits_with_device_completion"] == 1
    assert result["event_completion_to_following_copy_gap"]["median_ms"] == 5 / 1e6
    assert result["compute_predecessor_to_following_copy_gap"]["median_ms"] == 10 / 1e6
    assert result["all_recorded_prior_upload_dependencies_to_copy_gap"]["median_ms"] == 5 / 1e6


def test_missing_completion_is_not_assumed_from_predecessor():
    with database() as db:
        result = module.analyze(db, 17)
    assert result["matched_waits_with_device_completion"] == 0
    assert result["event_completion_to_following_copy_gap"]["count"] == 0
    assert result["copies_rejected_for_missing_prior_completion"] == 1


def test_inconsistent_completion_gap_is_visible():
    with database() as db:
        db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT ADD COLUMN timestamp DEFAULT 0")
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_CUDA_EVENT SET timestamp=155")
        result = module.analyze(db, 17)
    assert result["negative_completion_gaps"] == 1
    assert result["negative_prior_dependency_gaps"] == 1


def test_prior_upload_activity_also_bounds_next_copy():
    with database() as db:
        db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT ADD COLUMN timestamp DEFAULT 0")
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_CUDA_EVENT SET timestamp=140")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
                   (0, 1, 1 << 24, 5, 4))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?,?)",
                   (142, 148, 1 << 24, 0, 1, 17, 5, 1))
        result = module.analyze(db, 17)
    assert result["event_completion_to_following_copy_gap"]["median_ms"] == 10 / 1e6
    assert result["all_recorded_prior_upload_dependencies_to_copy_gap"]["min_ms"] == 2 / 1e6
