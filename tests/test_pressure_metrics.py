"""CPU-only public pressure report serialization/rendering contracts."""
import json

from diffusers_dynamic_offloader.metrics import DynamicOffloadMetrics


def test_pressure_report_is_json_ready_and_readable():
    prefix = "fast_source_cache_pressure_observation_"
    values = {prefix + "schema_version": 1, prefix + "entries": 3,
              prefix + "recoveries": 2, prefix + "last_reason": 2,
              prefix + "failed_observations": 1}
    report = DynamicOffloadMetrics(route="vmm_fast_runtime", runtime={"vmm": values})
    assert json.loads(json.dumps(report.as_dict()))["runtime"]["vmm"] == values
    # Render through the public API, without host access to VMM internals.
    assert "source_cache_pressure=entries=3 recoveries=2 last_reason=commit failed_observations=1" in report.format_profile()


def test_older_binary_does_not_invent_pressure_metrics():
    report = DynamicOffloadMetrics(route="vmm_fast_runtime", runtime={"vmm": {"fast_faults": 1}})
    assert "source_cache_pressure=" not in report.format_profile()
