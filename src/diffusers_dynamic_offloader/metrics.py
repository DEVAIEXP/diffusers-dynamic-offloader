"""Stable, runner-neutral reports for Dynamic Offload execution.

The library owns the meaning and schema of its telemetry.  Hosts may render or
persist this object, but must not inspect hook state or VMM implementation
attributes directly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

DYNAMIC_OFFLOAD_METRICS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DynamicOffloadMetrics:
    """A serializable snapshot of one DDO component route.

    ``configuration`` describes the resolved request, ``setup`` captures the
    apply operation, and ``runtime`` is populated from a hook when one exists.
    Values are deliberately ordinary Python data so JSON writers and host
    trackers can consume them without importing DDO internals.
    """

    route: str
    configuration: Mapping[str, Any] = field(default_factory=dict)
    setup: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)

    @property
    def schema_version(self) -> int:
        return DYNAMIC_OFFLOAD_METRICS_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-ready payload with a versioned schema."""
        return {
            "schema_version": self.schema_version,
            "route": self.route,
            "configuration": deepcopy(dict(self.configuration)),
            "setup": deepcopy(dict(self.setup)),
            "runtime": deepcopy(dict(self.runtime)),
        }

    def format_profile(self) -> str:
        """Render the compact, stable summary used by ``--show-profile``."""
        lines = [
            "DDO profile",
            f"  schema={self.schema_version} route={self.route}",
        ]
        requested = self.configuration.get("dynamic_offload_requested_preset")
        detachment = self.setup.get("auxiliary_source_detachment", {})
        if detachment:
            lines.append(f"  auxiliary_source_detachment={detachment['status']} "
                         f"copied_bytes={detachment['copied_bytes']} "
                         "(CPU ownership only; commit recovery not proven)")
        host_memory = self.setup.get("host_memory", {})
        if host_memory:
            host = host_memory.get("host", {})
            storages = host_memory.get("cpu_storages", {})
            lines.append(f"  host_memory={host_memory['status']} phase={host_memory['phase']} "
                         f"private_bytes={host.get('process_private_bytes', 'unavailable')} "
                         f"cpu_storage_bytes={storages.get('unique_storage_bytes', 'unavailable')} "
                         "(phase observation, not a peak or allocation census)")
        host_cache = self.setup.get("host_cache_release", {})
        if host_cache:
            delta = host_cache.get("allocator_bytes_delta")
            lines.append(f"  host_cache_release={host_cache['status']} phase={host_cache['phase']} "
                         f"allocator_bytes_delta={delta if delta is not None else 'unavailable'}")
        source = self.setup.get("weight_source", {})
        if source:
            lines.append(f"  weight_source={source.get('resolver', 'legacy_exact_names')} bound={source.get('bound', 0)} "
                         f"available={source.get('available', 0)} rejected={source.get('rejected_storage_groups', 0)} "
                         "(source metadata, not native execution support)")
        capacity = self.setup.get("capacity_inspection", {})
        if capacity:
            lines.append(f"  component={self.configuration.get('component', 'model')} metadata_only (not execution support)")
            lines.append(f"  selected_modules={capacity['selected_module_count']} "
                         f"aligned_payload_gib={capacity['selected_aligned_payload_bytes'] / 1024**3:.4f} "
                         f"nonselected_gib={capacity['nonselected_unique_parameter_bytes'] / 1024**3:.4f} "
                         f"shared_owner_gib={capacity['shared_selected_nonselected_bytes'] / 1024**3:.4f}")
        effective = self.configuration.get("dynamic_offload_effective_preset")
        mode = self.configuration.get("dynamic_offload_execution_mode")
        if requested or effective or mode:
            lines.append(f"  preset={requested or 'none'} -> {effective or 'none'} mode={mode or 'none'}")

        state = self.runtime.get("state", {})
        component = self.runtime.get("component_ownership", {})
        if component:
            lines.append(f"  experimental_component_ownership parameters={component['parameter_groups']} "
                         f"buffers={component['buffer_groups']} buffer_bytes={component['buffer_bytes']} "
                         f"units={component['execution_units']} lookahead={component['lookahead']}")
        if state:
            lines.append(
                "  modules="
                f"{state.get('module_count', 0)} patched={state.get('patched_module_count', 0)} "
                f"total_gb={state.get('total_gb', 0):.4f}"
            )
        vmm = self.runtime.get("vmm", {})
        if vmm:
            resident_bytes = vmm.get("fast_resident_bytes", vmm.get("arena_resident_bytes", 0))
            materializations = vmm.get("fast_faults", vmm.get("arena_materializations", 0))
            temporary = vmm.get("fast_temporary_tensors", vmm.get("arena_temporary_fallbacks", 0))
            lines.append(
                "  vmm="
                f"resident_gb={resident_bytes / 1024**3:.4f} "
                f"materializations={materializations} "
                f"temporary_fallbacks={temporary}"
            )
        timing = self.runtime.get("execution_timing", {})
        allocation = self.runtime.get("allocation_diagnostics", {})
        if allocation:
            lines.append(f"  allocation_diagnostics={allocation.get('status', 'unknown')} "
                         f"steps={len(allocation.get('steps', []))} "
                         "(native host calls, no boundary wait; diagnostic only)")
            for step in allocation.get("steps", []):
                counters = step.get("driver_delta", {})
                values = " ".join(f"{name}_host_s={counters.get(name + '_host_seconds', 0):.6f}"
                                  for name in ("create", "map", "access", "unmap", "release"))
                for name in ("access_retry_sync", "fault_range"):
                    if name + "_host_seconds" in counters:
                        values += f" {name}_host_s={counters[name + '_host_seconds']:.6f}"
                if "access_return_not_ready_calls" in counters:
                    values += f" access_not_ready_calls={counters['access_return_not_ready_calls']}"
                lines.append(f"  allocation_step={step['step_index'] + 1} "
                             f"host_submit_s={step['host_submission_seconds']:.6f} {values}")
        operators = self.runtime.get('operator_diagnostics')
        if operators:
            lines.append(f"  operator_diagnostics={operators['status']} "
                         f"samples={len(operators['samples'])} (sampled profiler; may synchronize; diagnostic only)")
            for sample in operators['samples']:
                lines.append(f"  operator_invocation={sample['invocation_index'] + 1} "
                             f"host_forward_s={sample['host_forward_seconds']:.6f} "
                             f"profiler_stop_s={sample['profiler_stop_seconds']:.6f} "
                             f"device_events={sample['device_activity_events']} trace={sample['trace_path']}")
                for category in ('top_host', 'top_device'):
                    for row in sample[category][:3]:
                        lines.append(f"    {category}={row['name']} calls={row['calls']} "
                                     f"self_host_s={row['self_host_seconds']:.6f} "
                                     f"self_device_s={row['self_device_seconds']:.6f}")
        pressure_prefix = "fast_source_cache_pressure_observation_"
        if vmm.get(pressure_prefix + "schema_version") == 1:
            reasons = {0: "none", 1: "ram", 2: "commit", 3: "ram_and_commit", 4: "observation_unavailable"}
            reason = reasons.get(vmm.get(pressure_prefix + "last_reason"), "unknown")
            lines.append(
                "  source_cache_pressure="
                f"entries={vmm.get(pressure_prefix + 'entries', 0)} "
                f"recoveries={vmm.get(pressure_prefix + 'recoveries', 0)} "
                f"last_reason={reason} "
                f"failed_observations={vmm.get(pressure_prefix + 'failed_observations', 0)}"
            )
        if timing:
            lines.append(f"  cuda_timing={timing.get('status', 'unknown')} scope={timing.get('scope', 'unknown')}")
            diagnostic = timing.get("diagnostic_step_wait", {})
            if diagnostic.get("enabled"):
                lines.append(f"  diagnostic_step_wait_calls={diagnostic['calls']} "
                             f"host_seconds={diagnostic['host_seconds']:.6f} (not a benchmark)")
            if timing.get("status") == "complete":
                lines.append(f"  cuda_phase_seconds={timing['phase_seconds']:.6f} tail_seconds={timing['tail_seconds']:.6f}")
                steady = timing.get("steady_state_seconds_per_step")
                if steady is not None:
                    lines.append(f"  cuda_steady_state_seconds_per_step={steady:.6f}")
        collection = self.runtime.get("completion_collection", {})
        timeline = self.runtime.get("native_timeline", {})
        if timeline:
            lines.append(f"  native_timeline={timeline.get('status', 'unknown')} "
                         f"samples={len(timeline.get('samples', []))} "
                         f"dropped={timeline.get('dropped', 0)} (stream windows, not engine activity)")
        release_seconds = vmm.get("fast_torch_allocator_cache_eviction_sync_seconds")
        memory = []
        for name, label in (("live_mapped_bytes", "live_gib"),
                            ("cached_mapped_bytes", "cached_gib")):
            value = vmm.get(f"fast_torch_allocator_{name}")
            if value is not None:
                memory.append(f"{label}={value / 1024**3:.4f}")
        if memory:
            lines.append("  torch_allocator=" + " ".join(memory)
                         + " (process ledger snapshot, not GPU total/peak)")
        footprint = []
        for name, label in (("requested", "requested_gib"),
                            ("granularity_padding", "padding_gib"),
                            ("reuse_excess", "reuse_excess_gib")):
            value = vmm.get(f"fast_torch_allocator_footprint_{name}_bytes")
            if value is not None:
                footprint.append(f"{label}={value / 1024**3:.4f}")
        if footprint:
            lines.append("  torch_allocation_footprint=" + " ".join(footprint)
                         + " (atomic live ledger, not GPU total/peak)")
        pool_mapped = vmm.get("fast_torch_allocator_pool_mapped_bytes")
        if pool_mapped is not None:
            pool = []
            for name in ("mapped", "requested", "leased", "available", "pending", "quarantined"):
                value = vmm.get(f"fast_torch_allocator_pool_{name}_bytes")
                if value is not None:
                    pool.append(f"{name}_gib={value / 1024**3:.4f}")
            lines.append("  torch_span_pool=" + " ".join(pool)
                         + " (unique backing snapshot, not GPU total/peak)")
        if release_seconds is not None:
            lines.append(
                f"  torch_cache_eviction_sync_seconds={release_seconds:.6f} "
                f"attempts={vmm.get('fast_torch_allocator_cache_eviction_attempts', 0)} "
                "(process lifetime, host time)"
            )
        if collection.get("status") == "collected":
            lines.append(f"  source_tickets={collection['tickets_before']} -> {collection['tickets_after']} (nonblocking collection)")
        return "\n".join(lines)
