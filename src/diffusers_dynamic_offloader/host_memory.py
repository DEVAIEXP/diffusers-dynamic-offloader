"""Read-only host observations at caller-selected, nonconcurrent phase boundaries."""
from __future__ import annotations

import time

from .host_cache import _host_snapshot
from .metrics import DynamicOffloadMetrics


def inspect_host_memory(module=None, *, phase: str = "", max_storage_samples: int = 16) -> DynamicOffloadMetrics:
    """Observe host memory and registered CPU tensor storages without touching data.

    Never initializes/synchronizes CUDA, collects garbage, copies weights or
    changes placement. The census includes only the passed module's registered
    parameters/buffers, not Python closures, native storage or other components.
    Storage bytes are mapping extents, not RSS or proven private commitment.
    Call only while the component is not concurrently loading/forwarding/unloading.
    """
    if type(max_storage_samples) is not int or max_storage_samples < 0:
        raise ValueError("max_storage_samples must be a non-negative integer")
    started = time.perf_counter()
    observation = {"phase": phase, "host": _host_snapshot(), "status": "observed"}
    if module is not None:
        groups = {}
        errors = []
        counts = {"cpu_tensor_references": 0, "cpu_tensor_reference_bytes": 0,
                  "non_cpu_tensor_references": 0, "inspection_errors": 0}
        registry = getattr(module, "_ddo_safetensor_slices", None)
        for kind, iterator in (("parameter", module.named_parameters(remove_duplicate=False)),
                               ("buffer", module.named_buffers(remove_duplicate=False))):
            for name, tensor in iterator:
                if tensor.device.type != "cpu":
                    counts["non_cpu_tensor_references"] += 1
                    continue
                counts["cpu_tensor_references"] += 1
                counts["cpu_tensor_reference_bytes"] += tensor.numel() * tensor.element_size()
                try:
                    storage = tensor.untyped_storage()
                    size = storage.nbytes()
                    # Storage extent and address identify aliases without
                    # retaining the tensor/storage in the returned report.
                    key = (storage.data_ptr(), size)
                    group = groups.setdefault(key, {"storage_bytes": size, "references": 0,
                                                   "examples": [], "shards": set()})
                    group["references"] += 1
                    if len(group["examples"]) < 3:
                        group["examples"].append({"kind": kind, "name": name})
                    if registry is not None:
                        descriptor = registry.lookup(tensor)
                        if descriptor is not None:
                            group["shards"].add(str(descriptor.path))
                except Exception as error:
                    counts["inspection_errors"] += 1
                    if len(errors) < max_storage_samples:
                        errors.append({"kind": kind, "name": name, "error": str(error)})
        ordered = sorted(groups.values(), key=lambda item: item["storage_bytes"], reverse=True)
        samples = [{**group, "shards": sorted(group["shards"])}
                   for group in ordered[:max_storage_samples]]
        observation["cpu_storages"] = {
            **counts, "unique_storage_count": len(groups),
            "unique_storage_bytes": sum(group["storage_bytes"] for group in ordered),
            "storage_samples": samples, "omitted_storage_count": max(0, len(groups) - len(samples)),
            "errors": errors, "scope": "registered_module_parameters_and_buffers_only",
        }
        if errors or counts["inspection_errors"]:
            observation["status"] = "partial"
    observation["elapsed_seconds"] = time.perf_counter() - started
    return DynamicOffloadMetrics(route="host_memory_observation",
                                configuration={"max_storage_samples": max_storage_samples},
                                setup={"host_memory": observation})
