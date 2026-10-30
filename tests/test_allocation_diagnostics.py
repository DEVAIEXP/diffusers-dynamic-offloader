import pytest
from diffusers_dynamic_offloader.allocation_diagnostics import DynamicOffloadAllocationDiagnostics
from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics


class Native:
    def __init__(self):
        self.enabled = False
        self.calls = 0
        self.transitions = []

    def set_allocation_diagnostics(self, enabled):
        self.transitions.append(enabled)
        self.enabled = enabled

    def allocation_diagnostics_snapshot(self):
        self.calls += 1
        return dict(schema_version=1, enabled=self.enabled, create_calls=self.calls,
                    create_host_seconds=self.calls / 100, create_failed_calls=0,
                    pool_reuse_calls=self.calls * 2, pool_new_backings=self.calls,
                    pool_observed_peak_mapped_bytes=2048)


def test_disabled_never_inspects_native():
    with DynamicOffloadAllocationDiagnostics(native=object()) as scope:
        scope.mark_step()
    assert scope.snapshot()["status"] == "disabled"


def test_steps_deltas_and_public_report():
    native = Native()
    with DynamicOffloadAllocationDiagnostics(enabled=True, native=native) as scope:
        scope.mark_step()
        scope.mark_step()
    assert native.transitions == [True, False]
    result = scope.attach(DynamicOffloadMetrics(route="dynamic_offload")).as_dict()
    diag = result["runtime"]["allocation_diagnostics"]
    assert diag["status"] == "complete" and diag["benchmark_eligible"] is False
    assert len(diag["steps"]) == 2
    assert diag["steps"][0]["driver_delta"]["create_calls"] == 1
    assert diag["total_delta"]["create_calls"] == 3
    assert diag["pool_observed_peak_mapped_bytes"] == 2048
    assert "allocation_step=1" in scope.attach(DynamicOffloadMetrics(route="dynamic_offload")).format_profile()
    with pytest.raises(RuntimeError, match="reused"):
        scope.__enter__()
    with pytest.raises(RuntimeError, match="active"):
        scope.mark_step()


def test_failure_releases_scope():
    native = Native()
    with pytest.raises(ValueError, match="model failed"):
        with DynamicOffloadAllocationDiagnostics(enabled=True, native=native) as scope:
            raise ValueError("model failed")
    assert native.transitions == [True, False]
    assert scope.snapshot()["failed"] is True


def test_extended_fields_reach_public_report_without_runner_changes():
    class Extended(Native):
        def allocation_diagnostics_snapshot(self):
            snapshot = super().allocation_diagnostics_snapshot()
            return {**snapshot, 'schema_version': 2,
                    'access_retry_sync_calls': self.calls,
                    'access_retry_sync_host_seconds': self.calls * .5,
                    'access_return_not_ready_calls': self.calls,
                    'fault_range_host_seconds': self.calls * .75}
    with DynamicOffloadAllocationDiagnostics(enabled=True, native=Extended()) as scope:
        scope.mark_step()
    report = scope.attach(DynamicOffloadMetrics(route='dynamic_offload'))
    diag = report.as_dict()['runtime']['allocation_diagnostics']
    assert diag['native_schema_version'] == 2
    assert diag['steps'][0]['driver_delta']['access_retry_sync_host_seconds'] == .5
    assert diag['total_delta']['access_return_not_ready_calls'] == 2
    assert 'access_retry_sync_host_s=0.500000' in report.format_profile()
    assert 'access_not_ready_calls=1' in report.format_profile()


def test_unavailable_baseline_releases_scope():
    native = Native()
    native.allocation_diagnostics_snapshot = lambda: {}
    with pytest.raises(RuntimeError, match="unavailable"):
        with DynamicOffloadAllocationDiagnostics(enabled=True, native=native):
            pass
    assert native.transitions == [True, False]


def test_diagnostic_read_failure_does_not_mask_model_failure():
    native = Native()
    with pytest.raises(ValueError, match="model failed"):
        with DynamicOffloadAllocationDiagnostics(enabled=True, native=native) as scope:
            native.allocation_diagnostics_snapshot = lambda: {}
            raise ValueError("model failed")
    assert native.transitions == [True, False]
    assert scope.snapshot()["status"] == "failed"


def test_public_api_disabled_and_protected_route():
    from diffusers_dynamic_offloader.dynamic_offload import DynamicOffloadApplyResult
    result = DynamicOffloadApplyResult(module=None, hook=None)
    with result.allocation_diagnostics() as scope:
        scope.mark_step()
    assert scope.snapshot()["status"] == "disabled"
    with pytest.raises(ValueError, match="vmm_fast"):
        result.allocation_diagnostics(enabled=True)
