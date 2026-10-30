"""Operator diagnostic CPU contracts; never initialize CUDA."""
import json
from unittest.mock import patch

import pytest
import torch

from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadApplyResult
from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics
from diffusers_dynamic_offloader.operator_diagnostics import DynamicOffloadOperatorDiagnostics


def test_disabled_does_not_inspect_model_or_cuda():
    with DynamicOffloadOperatorDiagnostics(module=object()) as scope:
        pass
    assert scope.snapshot()['status'] == 'disabled'


def test_real_cpu_profile_preserves_outputs_and_removes_hooks(tmp_path):
    model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())).eval()
    x = torch.randn(2, 4)
    expected = model(x)
    forwards = [m.forward for m in model.modules()]
    with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
        result = DynamicOffloadApplyResult(module=model, hook=None)
        with result.operator_diagnostics(enabled=True, module_path='0', output_dir=tmp_path,
                                         use_cuda=False) as scope:
            for _ in range(4):
                assert torch.equal(model(x), expected)
    report = scope.attach(DynamicOffloadMetrics(route='dynamic_offload'))
    data = report.as_dict()['runtime']['operator_diagnostics']
    assert data['status'] == 'complete'
    assert data['benchmark_eligible'] is False
    assert [s['invocation_index'] for s in data['samples']] == [0, 2]
    assert data['observed_invocations'] == 4
    for sample in data['samples']:
        assert sample['device_activity_events'] == 0
        assert sample['child_ranges']
        assert sample['top_host']
        assert json.loads(open(sample['trace_path'], encoding='utf-8').read())['traceEvents']
    assert 'operator_diagnostics=complete samples=2' in report.format_profile()
    assert [m.forward for m in model.modules()] == forwards
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    assert scope.module is None
    assert not torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match='reused'):
        scope.__enter__()


@pytest.mark.parametrize('indices', [(), (0, 0), (-1,), (32,), (True,), tuple(range(5))])
def test_invalid_selection_has_no_hooks(tmp_path, indices):
    model = torch.nn.Sequential(torch.nn.Identity())
    with pytest.raises(ValueError):
        with DynamicOffloadOperatorDiagnostics(enabled=True, module=model, module_path='0',
                invocations=indices, output_dir=tmp_path, use_cuda=False):
            pass
    assert not model[0]._forward_hooks


def test_incomplete_and_exclusive_scope(tmp_path):
    model = torch.nn.Sequential(torch.nn.Identity())
    with DynamicOffloadOperatorDiagnostics(enabled=True, module=model, module_path='0',
            output_dir=tmp_path, use_cuda=False) as scope:
        with pytest.raises(RuntimeError, match='already owned'):
            with DynamicOffloadOperatorDiagnostics(enabled=True, module=model, module_path='0',
                    output_dir=tmp_path, use_cuda=False):
                pass
    assert scope.snapshot()['status'] == 'incomplete'
    assert not model[0]._forward_hooks


def test_original_model_error_and_existing_hooks_preserved(tmp_path):
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise ValueError('original model failure')
    model = torch.nn.Sequential(Broken())
    handle = model[0].register_forward_pre_hook(lambda module, args: None)
    with pytest.raises(ValueError, match='original model failure'):
        with DynamicOffloadOperatorDiagnostics(enabled=True, module=model, module_path='0',
                output_dir=tmp_path, use_cuda=False) as scope:
            model(torch.ones(1))
    assert scope.snapshot()['status'] == 'failed'
    assert len(model[0]._forward_pre_hooks) == 1
    handle.remove()
    assert not torch.cuda.is_initialized()


def test_refuses_trace_overwrite_before_hooks(tmp_path):
    trace = tmp_path / 'operators_invocation_00.json'
    trace.touch()
    model = torch.nn.Sequential(torch.nn.Identity())
    with pytest.raises(FileExistsError):
        with DynamicOffloadOperatorDiagnostics(enabled=True, module=model, module_path='0',
                output_dir=tmp_path, use_cuda=False):
            pass
    assert not model[0]._forward_pre_hooks


def test_cuda_tracing_unavailable_refused_without_initialization(tmp_path):
    model = torch.nn.Sequential(torch.nn.Identity())
    with patch.object(torch.profiler, 'supported_activities', return_value={torch.profiler.ProfilerActivity.CPU}), \
            patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
        with pytest.raises(RuntimeError, match='tracing unavailable'):
            with DynamicOffloadOperatorDiagnostics(enabled=True, module=model, module_path='0',
                    output_dir=tmp_path, use_cuda=True):
                pass
    assert not model[0]._forward_pre_hooks
    assert not torch.cuda.is_initialized()
