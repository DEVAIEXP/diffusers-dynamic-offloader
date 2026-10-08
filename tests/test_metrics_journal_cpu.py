"""Partial report durability without device waits or event collection."""
import gc
import json
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from diffusers_dynamic_offloader import DynamicOffloadApplyResult
from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics
from diffusers_dynamic_offloader.metrics_journal import DynamicOffloadMetricsJournal


class Result:
    def __init__(self):
        self.calls = []
        self.counters = {'pending_events': 2, 'source_read_bytes': 64}

    def metrics_snapshot(self, *, collect_completed):
        assert collect_completed is False
        self.calls.append(collect_completed)
        return DynamicOffloadMetrics('dynamic_offload', runtime=self.counters)


def test_records_remain_partial_detached_and_no_cuda_waits(tmp_path):
    result = Result()
    with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA')), \
         patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('wait')), \
         DynamicOffloadMetricsJournal(result, tmp_path, enabled=True) as journal:
        before = journal.record()
        step = journal.record(0)
        result.counters['pending_events'] = 3
        later = journal.record(1)
    old = json.loads(step.read_text())
    assert old['gpu_completion_proven'] is False
    assert old['completion_collection_requested'] is False
    assert old['metrics']['runtime']['pending_events'] == 2
    assert json.loads(later.read_text())['metrics']['runtime']['pending_events'] == 3
    assert before.exists() and len(result.calls) == 3
    assert not list(tmp_path.glob('*.tmp'))


def test_disabled_journal_does_not_resolve_path_or_observe_result():
    with DynamicOffloadMetricsJournal(object(), object()) as journal:
        assert journal.record() is None


def test_duplicate_record_refuses_without_replacing_evidence(tmp_path):
    result = Result()
    with DynamicOffloadMetricsJournal(result, tmp_path, enabled=True) as journal:
        path = journal.record(0)
        original = path.read_bytes()
        with pytest.raises(FileExistsError):
            journal.record(0)
        assert path.read_bytes() == original
        assert len(result.calls) == 1


def test_failed_snapshot_leaves_previous_records_intact(tmp_path):
    result = Result()
    with DynamicOffloadMetricsJournal(result, tmp_path, enabled=True) as journal:
        previous = journal.record()
        result.counters['invalid'] = float('nan')
        with pytest.raises(ValueError):
            journal.record(0)
    assert json.loads(previous.read_text())['schema_version'] == 1
    assert not (tmp_path / 'step_0000.json').exists()


def test_close_drops_result_reference_and_refuses_further_records(tmp_path):
    result = Result()
    reference = weakref.ref(result)
    with DynamicOffloadMetricsJournal(result, tmp_path, enabled=True) as journal:
        journal.record()
    del result
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match='closed'):
        journal.record(0)


def test_public_result_journal_does_not_collect_completions(tmp_path):
    def refuse():
        raise AssertionError('Unexpected completion collection')
    hook = SimpleNamespace(state=SimpleNamespace(as_dict=lambda: {}),
                           vmm_runtime_stats=lambda: {'pending': 1},
                           collect_completed_runtime_resources=refuse)
    result = DynamicOffloadApplyResult(module=None, hook=hook)
    with result.metrics_journal(tmp_path, enabled=True) as journal:
        path = journal.record(0)
    assert json.loads(path.read_text())['metrics']['runtime']['vmm']['pending'] == 1
