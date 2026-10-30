"""Bounded, opt-in detachment of residual loader-owned CPU inference tensors."""
from __future__ import annotations

import time

import torch

from .metrics import DynamicOffloadMetrics
from .host_cache import _host_snapshot
from .vmm_weight_store import SafetensorSliceRegistry


def _detach_cpu_sources(module, registry, max_copy_bytes, protected_intervals=()):
    """Transactional CPU-only worker; native readiness is checked by the caller."""
    if not isinstance(registry, SafetensorSliceRegistry):
        raise ValueError("Residual source detachment requires DDO source provenance")
    if any(child.training for child in module.modules()):
        raise ValueError("Residual source detachment requires an eval component")
    groups = {}
    for kind, iterator in (("parameter", module.named_parameters(remove_duplicate=False)),
                           ("buffer", module.named_buffers(remove_duplicate=False))):
        for name, tensor in iterator:
            if tensor.device.type != "cpu" or tensor.numel() == 0:
                continue
            start = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            group = groups.setdefault((start, size), {"owners": {}, "names": [], "descriptor": None})
            group["owners"][id(tensor)] = tensor
            group["names"].append(f"{kind}:{name}")
            descriptor = registry.lookup(tensor)
            if descriptor is not None:
                if group["descriptor"] is not None and group["descriptor"] != descriptor:
                    raise ValueError("Conflicting residual source aliases")
                group["descriptor"] = descriptor
    selected = {key: group for key, group in groups.items() if group["descriptor"] is not None}
    for start, size in selected:
        if any(start < protected_start + protected_size and protected_start < start + size
               for protected_start, protected_size in protected_intervals):
            raise ValueError("Residual CPU tensor overlaps a retained native CPU source")
    # Partial overlaps/strides need a more general storage-preserving transform.
    # Refuse instead of silently severing alias semantics or copying shard gaps.
    intervals = sorted(groups)
    for index, key in enumerate(intervals):
        start, size = key
        for other in intervals[index + 1:]:
            if other[0] >= start + size:
                break
            if key in selected or other in selected:
                raise ValueError("Overlapping residual CPU views are unsupported")
    total = sum(size for _start, size in selected)
    if total > max_copy_bytes:
        raise ValueError(f"Residual source copy requires {total} bytes, above the explicit {max_copy_bytes} byte limit")
    for group in selected.values():
        descriptor = group["descriptor"]
        for tensor in group["owners"].values():
            if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or not tensor.is_contiguous()
                    or tensor.layout != torch.strided or tensor.is_pinned()):
                raise ValueError("Residual source ownership/layout is unsupported")
            probe = SafetensorSliceRegistry()
            if not probe.bind(tensor, descriptor):
                raise ValueError("Residual alias shape/dtype does not match source provenance")
    prepared = []
    with torch.no_grad():
        for group in selected.values():
            owners = list(group["owners"].values())
            old = owners[0].detach()
            new = old.clone(memory_format=torch.preserve_format)
            prepared.append((group, owners, old, new))
        changed = []
        try:
            for group, owners, _old, new in prepared:
                for tensor in owners:
                    old_data = tensor.detach()
                    tensor.data = new
                    changed.append((tensor, old_data))
            registry.rebind_many([(old, new) for _group, _owners, old, new in prepared])
        except BaseException:
            for tensor, old in reversed(changed):
                tensor.data = old
            raise
    paths = sorted({str(group["descriptor"].path) for group in selected.values()})
    return {"copied_bytes": total, "copied_storage_groups": len(selected),
            "copied_tensor_owners": sum(len(group["owners"]) for group in selected.values()),
            "copied_named_references": sum(len(group["names"]) for group in selected.values()),
            "unbound_cpu_storage_groups": len(groups) - len(selected),
            "source_shard_count": len(paths), "source_shard_samples": paths[:16],
            "scope": "source_backed_registered_cpu_tensors_after_fast_attach",
            "gpu_placement_changed": False, "commit_recovery_proven": False}


def _protected_native_cpu_intervals(hook, stats):
    """Fail closed unless every retained native CPU binding is observed."""
    count = stats.get("cpu_source_bindings")
    if type(count) is not int or count < 0:
        raise ValueError("Residual source detachment requires known CPU native bindings and no started execution")
    if count == 0:
        return ()
    adapter = getattr(hook, "_vmm_fast_adapter", None)
    observe = getattr(adapter, "cpu_source_intervals", None)
    if not callable(observe):
        raise ValueError("Residual source detachment requires observable CPU native bindings and no started execution")
    observation = observe()
    if (not isinstance(observation, dict) or type(observation.get("schema_version")) is not int
            or observation["schema_version"] != 1
            or type(observation.get("binding_count")) is not int
            or observation["binding_count"] != count):
        raise ValueError("Retained native CPU source observation does not match binding count")
    intervals = observation.get("intervals")
    if not isinstance(intervals, (tuple, list)) or len(intervals) != count:
        raise ValueError("Retained native CPU source intervals are incomplete")
    for interval in intervals:
        if (not isinstance(interval, (tuple, list)) or len(interval) != 2
                or any(type(value) is not int or value <= 0 for value in interval)):
            raise ValueError("Retained native CPU source interval is invalid")
    return intervals


def detach_auxiliary_cpu_sources(result, *, enabled=False, max_copy_bytes=16 * 1024**2):
    """Detach bounded CPU loader views once before the first native forward.

    Diagnostic opt-in only. Requires exclusive component ownership; cannot find
    arbitrary escaped tensor references. Exact registered aliases retain object
    identity, values, device and dtype. Unsupported/partial aliases fail closed.
    Never changes CUDA residency, native cache budgets or a protected route.
    """
    if type(max_copy_bytes) is not int or max_copy_bytes <= 0:
        raise ValueError("max_copy_bytes must be a positive integer")
    started = time.perf_counter()
    observation = {"status": "disabled", "copied_bytes": 0}
    if enabled:
        hook = result.hook
        if hook is None or hook.execution_mode != "vmm_fast_runtime" or not hook._vmm_runtime_active:
            raise ValueError("Residual source detachment requires an attached vmm_fast component")
        previous = getattr(hook, "_auxiliary_source_detachment_report", None)
        if previous is not None:
            return previous
        store = hook._vmm_fast_store
        if store is None:
            raise ValueError("Native source ownership is unavailable")
        stats = store.runtime_stats()
        required = ("faults", "execution_plan_passes")
        if any(stats.get(key) != 0 for key in required):
            raise ValueError("Residual source detachment requires no started execution")
        if hook.config.vmm_component_tensor_ownership:
            raise ValueError("Whole-component ownership is outside residual CPU detachment scope")
        protected = _protected_native_cpu_intervals(hook, stats)
        host_before = _host_snapshot()
        observation = _detach_cpu_sources(result.module, hook._vmm_file_slices, max_copy_bytes, protected)
        observation["protected_native_cpu_bindings"] = len(protected)
        observation["host_before"] = host_before
        observation["host_after"] = _host_snapshot()
        observation["status"] = "detached" if observation["copied_bytes"] else "no_source_backed_cpu_tensors"
    observation["elapsed_seconds"] = time.perf_counter() - started
    report = DynamicOffloadMetrics(route="vmm_auxiliary_source_detachment",
                                  configuration={"enabled": bool(enabled), "max_copy_bytes": max_copy_bytes},
                                  setup={"auxiliary_source_detachment": observation})
    if enabled:
        hook._auxiliary_source_detachment_report = report
    return report
