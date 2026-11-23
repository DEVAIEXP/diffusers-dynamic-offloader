"""Persistent, model-agnostic capacity profiles for Dynamic Offloader.

Profiles deliberately store observations, not executable model code.  A host
runner supplies a stable context and records the outcome of a real workload;
DDO then makes that observation available to later runs on the same machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

DYNAMIC_OFFLOAD_PROFILE_SCHEMA_VERSION = 1
_PROFILE_PREFIX = "ddo-capacity-v1"
# VMM residency profiles use their own identity and file prefix so a linear
# profile with the same name can never be selected for the native route.
_VMM_PROFILE_PREFIX = "ddo-vmm-residency-v1"
VMM_PROFILE_IDENTITY_VERSION = 5


def _vmm_transport_identity(config) -> dict[str, Any]:
    """Weight transport capacities of a VMM residency profile identity.

    The measured gain of extra residency depends on where non-resident weights
    come from (retained host source versus file reads) and on the staging
    capacity, so a profile built with one transport is never applied to another.
    """
    def gib(name):
        value = getattr(config, name, 0.0)
        return round(float(value), 3) if isinstance(value, (int, float)) and math.isfinite(value) else None

    return {"transport": {"native": bool(getattr(config, "vmm_native_transport", False)),
                          "staging_gb": gib("vmm_staging_buffer_gb"),
                          "source_cache_gb": gib("vmm_source_cache_gb"),
                          "source_registered_gb": gib("vmm_source_registered_gb"),
                          "host_cache_gb": gib("vmm_host_cache_gb")}}
# Arena space kept above the two execution slots at the structural floor.
VMM_FLOOR_ARENA_HEADROOM_BYTES = 64 * 1024**2
_VMM_DEFAULT_PAGE_BYTES = 2 * 1024**2


def vmm_structural_floor_bytes(slot_bytes: int, page_bytes: int | None, *, slots: int = 2,
                               persistent_bytes: int = 0) -> int:
    """Smallest residency budget the native arena accepts with ``slots`` execution slots.

    Native admission page-aligns each slot and requires at least one free arena
    page beyond them, so the floor is ``slots`` page-aligned slots plus the
    larger of the arena headroom and one page, rounded up to whole pages.
    """
    page = page_bytes if isinstance(page_bytes, int) and page_bytes > 0 else _VMM_DEFAULT_PAGE_BYTES

    def align(value: int) -> int:
        return -(-value // page) * page

    return align(slots * align(slot_bytes) + align(persistent_bytes)
                 + max(VMM_FLOOR_ARENA_HEADROOM_BYTES, page))


def vmm_component_structural_floor(module, file_slices, *, float8_weight_only=False,
                                   buffer_limit_bytes=1024**2, page_bytes=None):
    """Size the attachment's component geometry without native allocation or tensor copies."""
    if float8_weight_only:
        from ddo_vmm_allocator.quantized_arena_adapter import discover_float8_component_layout

        layout, composites = discover_float8_component_layout(
            module, file_slices, buffer_limit_bytes=buffer_limit_bytes)
        del composites
    else:
        from ddo_vmm_allocator.component_arena_adapter import discover_component_layout

        layout = discover_component_layout(module, file_slices, buffer_limit_bytes=buffer_limit_bytes)
    slot = layout.capacity_snapshot()["execution_slot_payload_bytes"]
    persistent = 0
    for group, _descriptor in layout.component_lifetime_parameters:
        persistent = ((persistent + 511) // 512) * 512 + group.nbytes
    if type(slot) is not int or slot <= 0:
        raise ValueError("component ownership has no execution unit for its structural floor")
    floor = vmm_structural_floor_bytes(slot, page_bytes, persistent_bytes=persistent)
    # Return detached sizes only; do not retain loaded CPU tensor owners.
    return floor, {"layout": "float8_component" if float8_weight_only else "component",
                   "execution_slot_payload_bytes": slot, "execution_slots": 2,
                   "component_lifetime_payload_bytes": persistent, "floor_bytes": floor}
_SDNQ_DEFAULT_RESIDENT_CEILING_GB = 6.0
_SDNQ_PROBE_MIN_SAFE_BUDGET_GB = 2.0


def inspect_vmm_component(module, *, component: str = "model", module_pattern: str = ".*"):
    """Observe native selection on CPU/meta; no attachment or support claim.

    The optional allocator package is imported only when this is requested.
    No budget, profile file, module, source or CUDA state is modified.
    """
    from .metrics import DynamicOffloadMetrics

    if not isinstance(component, str) or not component.strip():
        raise ValueError("component must be a nonempty string")
    from ddo_vmm_allocator import inspect_model_linear_layout

    return DynamicOffloadMetrics(
        route="vmm_capacity_inspection",
        configuration={"component": component, "module_pattern": module_pattern},
        setup={"capacity_inspection": inspect_model_linear_layout(module, module_pattern)},
    )


class _CudaPeakSampler:
    """Sample driver-visible CUDA usage while the profiled region executes."""

    def __init__(self, device: torch.device, interval_seconds: float = 0.05):
        self.device = device
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            free, total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return
        self.peak_bytes = max(self.peak_bytes, int(total - free))

    def start(self) -> None:
        self._sample()

        def run() -> None:
            while not self._stop.wait(self.interval_seconds):
                self._sample()

        self._thread = threading.Thread(target=run, name="ddo-cuda-profile", daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2 + 0.1)
        self._sample()
        return self.peak_bytes


VMM_RESIDENCY_OBSERVATION_SCHEMA_VERSION = 1


def _total_system_ram_bytes() -> int | None:
    """Total physical RAM, or None when it cannot be observed."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]

            status = _MemoryStatus(length=ctypes.sizeof(_MemoryStatus))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
            return None
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None


_NATIVE_IDENTITY_CACHE: dict[tuple[str, int, int], str] = {}


def _native_library_identity() -> str | None:
    """SHA256 of the installed native allocator library, or None."""
    try:
        from ddo_vmm_allocator.torch_allocator import installed_library_path

        path = installed_library_path()
    except Exception:
        return None
    if path is None:
        return None
    try:
        status = Path(path).stat()
        key = (str(path), status.st_size, status.st_mtime_ns)
        if key not in _NATIVE_IDENTITY_CACHE:
            _NATIVE_IDENTITY_CACHE[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return _NATIVE_IDENTITY_CACHE[key]
    except OSError:
        return None


def _vmm_machine_identity(config) -> dict[str, Any]:
    """Machine part of a VMM residency profile identity (model + RAM/VRAM).

    Device memory is rounded to 256 MiB and RAM to 1 GiB so that small reported
    differences do not split profiles; the WDDM budget is observed per build and
    is not part of the key (it varies between sessions).
    """
    gpu_name = device_memory_mib = None
    try:
        properties = torch.cuda.get_device_properties(torch.device(config.execution_device))
        gpu_name = str(properties.name)
        device_memory_mib = int(round(properties.total_memory / (256 * 1024**2))) * 256
    except Exception:
        pass
    ram = _total_system_ram_bytes()
    return {
        "platform": sys.platform,
        "gpu_name": gpu_name,
        "device_memory_mib": device_memory_mib,
        "host_ram_gib": None if ram is None else int(round(ram / 1024**3)),
        "native_library_sha256": _native_library_identity(),
    }


def recommend_vmm_resident_budget(observation: Mapping[str, Any], margin_bytes: int) -> dict[str, Any] | None:
    """Residency ceiling from one measured build region, or None when unobservable.

    ``ceiling = device_budget - margin - (peak - weights_resident)``, clamped
    below by the structural floor (two execution slots plus arena headroom) and
    above by the floor plus the arena's virtual size. The non-weight part
    (activations, allocator pool, context) is what the measured region used
    besides the resident weights, so the formula is valid at any build budget.
    """
    budget = observation.get("device_budget_bytes")
    peak = observation.get("peak_usage_bytes")
    weights = observation.get("total_weight_resident_bytes")
    slots = observation.get("execution_slots_bytes")
    if not all(isinstance(value, int) and value >= 0 for value in (budget, peak, weights, slots)):
        return None
    # ``slots`` is the measured total of the (already page-aligned) slots.
    page = observation.get("page_bytes")
    persistent = observation.get("component_lifetime_capacity_bytes", 0)
    if type(persistent) is not int or persistent < 0:
        return None
    floor = (vmm_structural_floor_bytes(slots, page, slots=1, persistent_bytes=persistent)
             if slots else VMM_FLOOR_ARENA_HEADROOM_BYTES)
    non_weight = max(0, peak - weights)
    available = budget - margin_bytes - non_weight
    virtual = observation.get("arena_virtual_bytes")
    ceiling = max(floor, available)
    if isinstance(virtual, int) and virtual > 0:
        ceiling = min(ceiling, floor + virtual)
    if non_weight + floor > budget:
        fit = "does_not_fit"
    elif available < floor + margin_bytes:
        fit = "fits_at_limit"
    else:
        fit = "fits_with_headroom"
    return {"vmm_resident_weight_budget_gb": round(ceiling / 1024**3, 4), "vmm_resident_weight_budget_bytes": ceiling,
            "fit_class": fit, "floor_bytes": floor, "non_weight_peak_bytes": non_weight,
            "margin_bytes": margin_bytes, "device_budget_bytes": budget}


def recheck_vmm_profile_budget(saved_bytes: int, floor_bytes: int | None, non_weight_peak_bytes: int | None,
                               margin_bytes: int, device_budget_bytes: int | None) -> dict[str, Any]:
    """Fit a saved residency budget to the device budget observed at use time.

    The device budget (Windows DXGI local budget) can shrink between the build
    and a later run when other applications take memory. With the build's
    non-weight peak, ``device_budget - margin - non_weight`` is the largest
    budget that keeps this run under ``device_budget - margin``. The saved
    budget is reduced to it, never below the structural floor and never
    raised. Missing observations leave the saved budget unchanged.
    """
    result = {"saved_bytes": saved_bytes, "applied_bytes": saved_bytes,
              "device_budget_bytes": device_budget_bytes, "margin_bytes": margin_bytes}
    if not all(isinstance(value, int) and value >= 0 for value in (device_budget_bytes, non_weight_peak_bytes)):
        return {**result, "status": "unavailable"}
    limit = device_budget_bytes - margin_bytes - non_weight_peak_bytes
    if limit >= saved_bytes:
        return {**result, "status": "fits"}
    floor = floor_bytes if isinstance(floor_bytes, int) and floor_bytes > 0 else None
    if floor is not None and limit < floor:
        applied = min(saved_bytes, floor)
        return {**result, "applied_bytes": applied, "status": "device_budget_below_floor"}
    applied = max(limit, floor or 0)
    if applied <= 0:
        return {**result, "status": "device_budget_below_floor"}
    return {**result, "applied_bytes": applied, "status": "reduced"}


def _tensor_shapes(value: Any) -> Any:
    """Shapes of the tensors in a forward input, recursively; other values omitted."""
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        shapes = [_tensor_shapes(item) for item in value]
        return [shape for shape in shapes if shape is not None] or None
    if isinstance(value, Mapping):
        shapes = {str(key): _tensor_shapes(item) for key, item in value.items()}
        return {key: shape for key, shape in sorted(shapes.items()) if shape is not None} or None
    return None


class _VmmResidencySampler:
    """Observe the device during a VMM profile build region (no residency change).

    Samples driver-visible device usage and, on Windows, the WDDM local budget and
    this process's local usage; captures the first forward's tensor input shapes
    of the component. Missing observations stay ``None`` (unavailable, not zero).
    """

    def __init__(self, hook, module, device: torch.device, interval_seconds: float = 0.05, *, timing: bool = True):
        self.hook = hook
        # Without timing (use runs) no forward hook is installed, so the
        # device is never synchronized; only memory is sampled.
        self.module = module if timing else None
        self.device = device
        self.interval_seconds = interval_seconds
        self.peak_driver_used_bytes: int | None = None
        self.device_total_bytes: int | None = None
        self.peak_local_usage_bytes: int | None = None
        self.local_budget_bytes: int | None = None
        self.min_local_budget_bytes: int | None = None
        self.input_shapes: Any = None
        # Device-complete duration of each component forward (build runs only:
        # the post-forward hook synchronizes the device to time GPU work).
        self.forward_seconds: list[float] = []
        self._forward_seen = False
        self._forward_started: float | None = None
        self._handle = None
        self._post_handle = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            free, total = torch.cuda.mem_get_info(self.device)
        except Exception:
            free = total = None
        if total is not None:
            used = int(total - free)
            self.device_total_bytes = int(total)
            self.peak_driver_used_bytes = used if self.peak_driver_used_bytes is None else max(self.peak_driver_used_bytes, used)
        query = getattr(self.hook, "vmm_device_memory", None)
        memory = query() if callable(query) else None
        if memory:
            usage = int(memory["local_usage_bytes"])
            self.local_budget_bytes = int(memory["local_budget_bytes"])
            # WDDM lowers the process budget under the process's own load, so
            # the lowest budget of the region is the one its peak must fit.
            self.min_local_budget_bytes = (self.local_budget_bytes if self.min_local_budget_bytes is None
                                           else min(self.min_local_budget_bytes, self.local_budget_bytes))
            self.peak_local_usage_bytes = usage if self.peak_local_usage_bytes is None else max(self.peak_local_usage_bytes, usage)

    def _first_forward(self, module, args, kwargs) -> None:
        if not self._forward_seen:
            self._forward_seen = True
            self.input_shapes = {"args": _tensor_shapes(tuple(args)), "kwargs": _tensor_shapes(dict(kwargs))}
        torch.cuda.synchronize(self.device)
        self._forward_started = time.perf_counter()

    def _forward_done(self, module, args, kwargs, output) -> None:
        if self._forward_started is not None:
            torch.cuda.synchronize(self.device)
            self.forward_seconds.append(time.perf_counter() - self._forward_started)
            self._forward_started = None

    def start(self) -> None:
        if isinstance(self.module, torch.nn.Module):
            self._handle = self.module.register_forward_pre_hook(self._first_forward, with_kwargs=True)
            self._post_handle = self.module.register_forward_hook(self._forward_done, with_kwargs=True)
        self._sample()

        def run() -> None:
            while not self._stop.wait(self.interval_seconds):
                self._sample()

        self._thread = threading.Thread(target=run, name="ddo-vmm-residency-profile", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop sampling and remove the forward hook; idempotent."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2 + 0.1)
            self._thread = None
        for name in ("_handle", "_post_handle"):
            handle = getattr(self, name)
            if handle is not None:
                handle.remove()
                setattr(self, name, None)

    def stop(self, stats: Mapping[str, Any]) -> dict[str, Any]:
        self.close()
        self._sample()
        if self.local_budget_bytes is not None:
            budget, source, peak = self.min_local_budget_bytes, "dxgi_local_budget", self.peak_local_usage_bytes
        elif self.device_total_bytes is not None:
            budget, source, peak = self.device_total_bytes, "cuda_device_total", self.peak_driver_used_bytes
        else:
            budget = source = peak = None

        def stat(name):
            value = stats.get(name)
            return int(value) if isinstance(value, (int, float)) else None

        return {
            "schema_version": VMM_RESIDENCY_OBSERVATION_SCHEMA_VERSION,
            "execution_slots_bytes": stat("fast_execution_buffer_capacity_bytes"),
            **({"component_lifetime_capacity_bytes": stat("fast_component_lifetime_capacity_bytes")}
               if "fast_component_lifetime_capacity_bytes" in stats else {}),
            "page_bytes": stat("fast_page_bytes"),
            "arena_resident_bytes": stat("fast_resident_bytes"),
            "arena_virtual_bytes": stat("fast_virtual_bytes"),
            "total_weight_resident_bytes": stat("fast_total_weight_resident_bytes"),
            "device_total_bytes": self.device_total_bytes,
            "device_budget_bytes": budget,
            "budget_source": source,
            "peak_usage_bytes": peak,
            "peak_driver_used_bytes": self.peak_driver_used_bytes,
            "peak_local_usage_bytes": self.peak_local_usage_bytes,
            "first_forward_seen": self._forward_seen,
            "input_shapes": self.input_shapes,
            "forward_seconds": [round(value, 6) for value in self.forward_seconds],
            # Steady forward time: forwards after the first (which includes the
            # initial weight load). None with fewer than two forwards.
            "steady_forward_seconds": (round(sum(self.forward_seconds[1:]) / len(self.forward_seconds[1:]), 6)
                                       if len(self.forward_seconds) >= 2 else None),
            # Spread (max - min) of the steady forwards: the run's own timing
            # noise. None with fewer than two steady forwards (three forwards).
            "steady_forward_spread_seconds": (round(max(self.forward_seconds[1:]) - min(self.forward_seconds[1:]), 6)
                                              if len(self.forward_seconds) >= 3 else None),
        }


def decide_vmm_resident_budget(floor_seconds: float | None, ceiling_seconds: float | None,
                               floor_bytes: int, ceiling_bytes: int,
                               min_gain_per_gib: float, *,
                               floor_spread_seconds: float | None = None,
                               ceiling_spread_seconds: float | None = None) -> dict[str, Any]:
    """Choose between the structural floor and the hardware ceiling from two builds.

    The ceiling is kept only when both hold: the steady-forward speedup per
    extra GiB is at least ``min_gain_per_gib`` (fraction of the floor forward
    time per GiB), and the speedup exceeds the timing noise, taken as the
    larger steady-forward spread of the two builds. Otherwise the floor is kept
    and the device memory stays free. Missing timings or spreads keep the floor
    (conservative).
    """
    extra_gib = (ceiling_bytes - floor_bytes) / 1024**3
    noise = None if floor_spread_seconds is None or ceiling_spread_seconds is None else max(
        floor_spread_seconds, ceiling_spread_seconds)
    result = {"floor_forward_seconds": floor_seconds, "ceiling_forward_seconds": ceiling_seconds,
              "floor_forward_spread_seconds": floor_spread_seconds,
              "ceiling_forward_spread_seconds": ceiling_spread_seconds, "noise_seconds": noise,
              "extra_gib": round(extra_gib, 4), "min_gain_per_gib": min_gain_per_gib}
    if extra_gib <= 0:
        return {**result, "decision": "floor", "reason": "no_headroom", "gain_per_gib": None,
                "budget_bytes": floor_bytes}
    if not floor_seconds or ceiling_seconds is None or noise is None:
        return {**result, "decision": "floor", "reason": "timing_unavailable_use_three_or_more_steps",
                "gain_per_gib": None, "budget_bytes": floor_bytes}
    gain_seconds = floor_seconds - ceiling_seconds
    gain_per_gib = gain_seconds / floor_seconds / extra_gib
    if gain_seconds <= noise:
        reason = "gain_within_timing_noise"
    elif gain_per_gib < min_gain_per_gib:
        reason = "gain_below_threshold"
    else:
        reason = "gain_at_or_above_threshold"
    keep = reason == "gain_at_or_above_threshold"
    return {**result, "decision": "ceiling" if keep else "floor", "reason": reason,
            "gain_seconds": round(gain_seconds, 6), "gain_per_gib": round(gain_per_gib, 6),
            "budget_bytes": ceiling_bytes if keep else floor_bytes}


class _NamedProfile:
    """Library-owned identity, application and measurement for a named workload."""

    def __init__(self, module, config, *, component="model"):
        self.config = config
        self.context = {
            "name": config.profile_name,
            "model_class": type(module).__qualname__,
            "identity_version": 2,
            "execution_mode": config.execution_mode,
        }
        self.report = {"name": config.profile_name, "status": "disabled"}
        # Explicit VMM component profiles must not share observations merely
        # because two components have the same Python class/profile name.
        # Preserve generic VMM and all existing linear/SDNQ profile identities.
        if config.execution_mode == "vmm_fast_runtime":
            if not isinstance(component, str) or not component.strip():
                raise ValueError("VMM profile component must be a non-empty string.")
            # Residency profile identity: model, component, unit level and the
            # machine's RAM/VRAM and native artifact; own file prefix.
            self.context.update(identity_version=VMM_PROFILE_IDENTITY_VERSION, runtime="vmm",
                                component=component, unit_level=config.vmm_execution_unit_level)
            if config.profile_name:
                self.context.update(_vmm_machine_identity(config))
                self.context.update(_vmm_transport_identity(config))
            self.report["component"] = component
        self._active = False
        self._sdnq_probe_budget_gb: float | None = None
        self._vmm_hook = None
        # The component module, for the first-forward input-shape observation.
        self._module = module
        # Two-build VMM residency calibration: "floor" or "ceiling" while building.
        self._vmm_build_stage: str | None = None
        self._vmm_pending: dict[str, Any] | None = None

    def attach_hook(self, hook) -> None:
        """Associate the applied runtime so a VMM profile can take a snapshot."""
        self._vmm_hook = hook
        if (self.config.execution_mode == "vmm_fast_runtime" and self.config.profile_name
                and not self.config.build_profile and isinstance(self._module, torch.nn.Module)):
            saved = (self.report.get("vmm_residency") or {}).get("input_shapes")
            if saved is None:
                self.report["shape_check"] = "unavailable"
                return
            self.report["shape_check"] = "pending"

            def check_first_forward(module, args, kwargs):
                observed = {"args": _tensor_shapes(tuple(args)), "kwargs": _tensor_shapes(dict(kwargs))}
                self.report["shape_check"] = "match" if _json_value(observed) == _json_value(saved) else "mismatch"
                if self.report["shape_check"] == "mismatch":
                    print(f"[dynamic-offload] VMM profile {self.config.profile_name!r}: first-forward input "
                          "shapes differ from the build; rebuild the profile for this workload.")
                handle.remove()

            handle = self._module.register_forward_pre_hook(check_first_forward, with_kwargs=True)

    def residency_report(self) -> dict[str, Any] | None:
        """Public VMM residency profile summary for the metrics report, or None."""
        if self.config.execution_mode != "vmm_fast_runtime" or not self.config.profile_name:
            return None
        residency = self.report.get("vmm_residency") or {}
        keys = ("status", "fit_class", "resident_budget_gb", "shape_check", "path", "build_stage", "decision")
        result = {"name": self.config.profile_name, **{key: self.report.get(key) for key in keys}}
        result.update({key: residency.get(key) for key in (
            "device_budget_bytes", "budget_source", "execution_slots_bytes", "steady_forward_seconds")})
        # Peak of the last build run (the saved observation), and of this use run.
        result["build_peak_usage_bytes"] = residency.get("peak_usage_bytes")
        for key in ("use_peak_usage_bytes", "use_peak_driver_used_bytes", "use_device_budget_bytes"):
            result[key] = self.report.get(key)
        recommendation = self.report.get("recommendation") or {}
        result.update({key: recommendation.get(key) for key in (
            "reason", "gain_per_gib", "gain_seconds", "noise_seconds", "floor_forward_seconds",
            "ceiling_forward_seconds", "floor_budget_bytes", "ceiling_budget_bytes", "ceiling_trim_bytes",
            "build_min_device_budget_bytes")})
        result["margin_gb"] = self.config.vmm_profile_margin_gb
        result["min_gain_per_gib"] = self.config.vmm_profile_min_gain_per_gib
        return result

    def prepare(self):
        config = self.config
        if config.build_profile and not config.profile_name.strip():
            raise ValueError("build_profile requires a non-empty profile_name.")
        if not config.profile_name:
            return config
        if config.build_profile:
            if torch.device(config.execution_device).type != "cuda":
                raise ValueError("Profile calibration requires a CUDA execution device.")
            if config.execution_mode not in {"linear_runtime", "sdnq_runtime", "vmm_fast_runtime"}:
                raise ValueError(
                    "Profile calibration requires execution_mode='linear_runtime', 'sdnq_runtime', or 'vmm_fast_runtime'."
                )
            if config.execution_mode == "vmm_fast_runtime":
                if config.vmm_resident_weight_budget_gb > 0:
                    raise ValueError("A VMM residency profile build measures its own budgets; "
                                     "omit vmm_resident_weight_budget_gb.")
                self.report["profile_measurement_kind"] = "vmm_capacity"
                pending = get_dynamic_offload_profile_recommendation(load_dynamic_offload_profile(self.context) or {}) or {}
                ceiling = pending.get("ceiling_budget_bytes")
                if pending.get("vmm_build_stage") == "floor_measured" and isinstance(ceiling, int) and ceiling > 0:
                    # Second build: measure at the hardware ceiling found by the first.
                    self._vmm_build_stage, self._vmm_pending = "ceiling", dict(pending)
                    self.report.update(status="measuring_ceiling", build_stage="ceiling")
                    print(f"[dynamic-offload] VMM profile {config.profile_name!r}: build 2/2, measuring at the "
                          f"hardware ceiling {ceiling / 1024**3:.3f} GiB.")
                    return replace(config, vmm_resident_weight_budget_gb=ceiling / 1024**3)
                # First build: measure at the structural floor (DDO sizes it from budget 0).
                self._vmm_build_stage, self._vmm_pending = "floor", None
                self.report.update(status="measuring_floor", build_stage="floor")
                print(f"[dynamic-offload] VMM profile {config.profile_name!r}: build 1/2, measuring at the structural floor.")
                return config
            if not math.isfinite(config.profile_vram_headroom_gb) or config.profile_vram_headroom_gb < 0:
                raise ValueError("profile_vram_headroom_gb must be finite and non-negative.")
            self.report["status"] = "calibrating"
            # All profile builds begin with zero residents.  This preserves the
            # established BF16 flow and gives SDNQ a real activation baseline;
            # an SDNQ resident budget cannot safely be inferred before that
            # workload peak is known.
            if config.execution_mode == "sdnq_runtime":
                print(f"[dynamic-offload] profile {config.profile_name!r}: calibrating SDNQ zero-resident baseline.")
            print(f"[dynamic-offload] profile {config.profile_name!r}: calibrating with zero resident budget.")
            return replace(config, resident_module_budget_gb=0.0, profile_resident_module_budget_gb=0.0)
        if config.execution_mode == "vmm_fast_runtime":
            record = load_dynamic_offload_profile(self.context)
            if record is not None:
                last = (record.get("observations") or [{}])[-1]
                self.report["vmm_capacity"] = dict(last.get("vmm", {}))
                if last.get("vmm_residency") is not None:
                    self.report["vmm_residency"] = dict(last["vmm_residency"])
            recommendation = get_dynamic_offload_profile_recommendation(record or {}) or {}
            if config.vmm_resident_weight_budget_gb > 0:
                # An explicit budget stays authoritative over a saved profile.
                self.report["status"] = "explicit"
                self.report["resident_budget_gb"] = config.vmm_resident_weight_budget_gb
                return config
            budget = recommendation.get("vmm_resident_weight_budget_gb")
            if not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
                self.report["status"] = "missing"
                print(f"[dynamic-offload] VMM profile {config.profile_name!r}: no saved residency budget; "
                      "running at the structural floor (build it with build_profile=True).")
                return config
            self.report.update(status="applied", resident_budget_gb=budget, recommendation=dict(recommendation),
                               fit_class=recommendation.get("fit_class"),
                               build_stage=recommendation.get("vmm_build_stage"),
                               decision=recommendation.get("decision"))
            pending_note = (" (floor until the second build measures the ceiling)"
                            if recommendation.get("vmm_build_stage") == "floor_measured" else "")
            print(f"[dynamic-offload] VMM profile {config.profile_name!r}: residency budget={budget:.3f} GiB "
                  f"({recommendation.get('fit_class')}, decision={recommendation.get('decision')}){pending_note}.")
            # The arena re-checks this budget against the device budget at attach time.
            floor_bytes, non_weight = (recommendation.get("floor_budget_bytes"),
                                       recommendation.get("non_weight_peak_bytes"))
            return replace(config, vmm_resident_weight_budget_gb=float(budget), _profile_budget_applied=True,
                           _profile_floor_bytes=floor_bytes if isinstance(floor_bytes, int) else None,
                           _profile_non_weight_peak_bytes=non_weight if isinstance(non_weight, int) else None)
        # An explicit caller budget has always taken precedence over a saved
        # recommendation.  It is not itself a profile calibration.
        if config.resident_module_budget_gb > 0:
            return config
        record = load_dynamic_offload_profile(self.context)
        recommendation = get_dynamic_offload_profile_recommendation(record or {}) or {}
        budget = recommendation.get("profile_resident_module_budget_gb")
        if not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
            self.report["status"] = "missing"
            print(f"[dynamic-offload] profile {config.profile_name!r}: no recommendation; calibrate with build_profile=True.")
            return config
        if config.max_resident_module_budget_gb > 0:
            budget = min(budget, config.max_resident_module_budget_gb)
        probe_budget = recommendation.get("sdnq_probe_resident_budget_gb")
        if isinstance(probe_budget, (int, float)) and math.isfinite(probe_budget) and probe_budget > 0:
            budget = min(float(probe_budget), budget)
            self._sdnq_probe_budget_gb = budget
            self.report["status"] = "probing"
            print(
                f"[dynamic-offload] profile {config.profile_name!r}: "
                f"validating SDNQ resident probe={budget:.2f} GiB."
            )
        else:
            self.report["status"] = "applied"
        self.report["resident_budget_gb"] = budget
        print(f"[dynamic-offload] profile {config.profile_name!r}: requested resident budget={budget:.2f} GiB.")
        if config.auto_budget_policy == "off":
            return replace(config, resident_module_budget_gb=budget, _profile_budget_applied=True)
        return replace(config, profile_resident_module_budget_gb=budget, _profile_budget_applied=True)

    def _vmm_build_recommendation(self, observation: Mapping[str, Any]) -> dict[str, Any] | None:
        """Recommendation of one VMM build run (two-build calibration), or None.

        Build 1 runs at the structural floor and finds the hardware ceiling; it
        saves a final floor decision when there is no headroom or no steady
        timing, a whole-model decision when the model fits under the ceiling,
        and otherwise a pending record (floor budget usable meanwhile). Build 2
        runs at that ceiling and keeps it only if the steady-forward speedup per
        extra GiB reaches ``vmm_profile_min_gain_per_gib``.
        """
        margin = int(self.config.vmm_profile_margin_gb * 1024**3)
        threshold = float(self.config.vmm_profile_min_gain_per_gib)
        steady = observation.get("steady_forward_seconds")
        if self._vmm_build_stage == "ceiling":
            pending = self._vmm_pending or {}
            if _json_value(observation.get("input_shapes")) != _json_value(pending.get("input_shapes")):
                self.report.update(build_result="shape_mismatch between the two builds; rebuild from the floor",
                                   shape_check="mismatch")
                return {**pending, "vmm_build_stage": "restart_required"} if pending else None
            floor, ceiling = int(pending["floor_budget_bytes"]), int(pending["ceiling_budget_bytes"])
            decision = decide_vmm_resident_budget(
                pending.get("floor_forward_seconds"), steady, floor, ceiling, threshold,
                floor_spread_seconds=pending.get("floor_forward_spread_seconds"),
                ceiling_spread_seconds=observation.get("steady_forward_spread_seconds"))
            peak = observation.get("peak_usage_bytes")
            budgets = [value for value in (pending.get("device_budget_bytes"), observation.get("device_budget_bytes"))
                       if isinstance(value, int)]
            budget = min(budgets) if budgets else None
            if isinstance(peak, int) and budget is not None and peak > budget - margin:
                # Safety: the ceiling run's peak must fit the lowest device budget
                # of both builds minus the margin. Trim the ceiling by the
                # overshoot; fall back to the floor only if nothing remains.
                overshoot = peak - (budget - margin)
                trimmed = ceiling - overshoot
                decision["ceiling_trim_bytes"] = overshoot
                if trimmed <= floor:
                    decision.update(decision="floor", reason="ceiling_run_exceeded_device_budget", budget_bytes=floor)
                elif decision.get("decision") == "ceiling":
                    decision.update(reason="ceiling_trimmed_to_device_budget", budget_bytes=trimmed)
            chosen = decision.pop("budget_bytes")
            return {**{key: pending[key] for key in ("fit_class", "floor_budget_bytes", "ceiling_budget_bytes",
                                                     "device_budget_bytes",
                                                     "non_weight_peak_bytes", "margin_bytes", "input_shapes")
                       if key in pending},
                    **decision, "ceiling_peak_usage_bytes": peak, "vmm_build_stage": "final",
                    "build_min_device_budget_bytes": budget,
                    "vmm_resident_weight_budget_bytes": chosen,
                    "vmm_resident_weight_budget_gb": round(chosen / 1024**3, 4)}
        ceiling_info = recommend_vmm_resident_budget(observation, margin)
        if ceiling_info is None:
            return None
        floor, ceiling = ceiling_info["floor_bytes"], ceiling_info["vmm_resident_weight_budget_bytes"]
        spread = observation.get("steady_forward_spread_seconds")
        base = {"fit_class": ceiling_info["fit_class"], "floor_budget_bytes": floor, "ceiling_budget_bytes": ceiling,
                "floor_forward_seconds": steady, "floor_forward_spread_seconds": spread,
                "device_budget_bytes": ceiling_info["device_budget_bytes"],
                "non_weight_peak_bytes": ceiling_info["non_weight_peak_bytes"], "margin_bytes": margin,
                "min_gain_per_gib": threshold, "input_shapes": observation.get("input_shapes")}
        virtual = observation.get("arena_virtual_bytes")
        if isinstance(virtual, int) and virtual > 0 and ceiling >= floor + virtual:
            final = {"decision": "whole_model", "reason": "model_fits_under_the_ceiling", "budget": ceiling}
        elif ceiling <= floor:
            final = {"decision": "floor", "reason": "no_headroom", "budget": floor}
        elif steady is None or spread is None:
            # The decision needs at least two steady forwards (three steps).
            final = {"decision": "floor", "reason": "timing_unavailable_use_three_or_more_steps", "budget": floor}
        else:
            return {**base, "vmm_build_stage": "floor_measured", "decision": "pending_ceiling_build",
                    "vmm_resident_weight_budget_bytes": floor,
                    "vmm_resident_weight_budget_gb": round(floor / 1024**3, 4)}
        chosen = final.pop("budget")
        return {**base, **final, "gain_per_gib": None, "vmm_build_stage": "final",
                "vmm_resident_weight_budget_bytes": chosen, "vmm_resident_weight_budget_gb": round(chosen / 1024**3, 4)}

    @contextmanager
    def measure(self):
        """Measure only the caller's inference region, then persist on success."""
        if self.config.execution_mode == "vmm_fast_runtime":
            if not self.config.build_profile:
                if not self.config.profile_name or self._vmm_hook is None:
                    yield self.report
                    return
                # Use run: observe this region's device peak (memory only).
                sampler = _VmmResidencySampler(self._vmm_hook, None, torch.device(self.config.execution_device),
                                               timing=False)
                try:
                    sampler.start()
                    yield self.report
                    sampler.close()
                    sampler._sample()
                    self.report.update(use_peak_usage_bytes=(sampler.peak_local_usage_bytes
                                                             if sampler.local_budget_bytes is not None
                                                             else sampler.peak_driver_used_bytes),
                                       use_peak_driver_used_bytes=sampler.peak_driver_used_bytes,
                                       use_device_budget_bytes=sampler.min_local_budget_bytes)
                finally:
                    sampler.close()
                return
            if self._vmm_hook is None:
                raise RuntimeError("VMM capacity profiling requires an applied dynamic-offload hook.")
            if self._active:
                raise RuntimeError("A profile measurement is already active.")
            self._active = True
            device = torch.device(self.config.execution_device)
            residency = _VmmResidencySampler(self._vmm_hook, self._module, device)
            try:
                residency.start()
                yield self.report
                torch.cuda.synchronize(device)
                stats = self._vmm_hook.vmm_runtime_stats() or {}
                residency_observation = residency.stop(stats)
                capacity = {
                    "backend": "vmm_fast_runtime",
                    "platform": sys.platform,
                    "virtual_bytes": int(stats.get("fast_virtual_bytes", 0)),
                    "page_bytes": int(stats.get("fast_page_bytes", 0)),
                    "resident_bytes": int(stats.get("fast_resident_bytes", 0)),
                    "file_backed_bindings": int(stats.get("fast_file_backed_bindings", 0)),
                    "file_backed_bytes": int(stats.get("fast_file_backed_bytes", 0)),
                    "cpu_source_bindings": int(stats.get("fast_cpu_source_bindings", 0)),
                    "faults": int(stats.get("fast_faults", 0)),
                    "signature_hits": int(stats.get("fast_signature_hits", 0)),
                    "uploads": int(stats.get("fast_uploads", 0)),
                    "temporary_tensors": int(stats.get("fast_temporary_tensors", 0)),
                    "temporary_bytes": int(stats.get("fast_temporary_bytes", 0)),
                }
                observation = {
                    "status": "success",
                    "profile_measurement_kind": "vmm_capacity",
                    "capacity": {"metric": "vmm_virtual_bytes", "value": capacity["virtual_bytes"]},
                    "vmm": capacity,
                    # Device observation of this build region.
                    "vmm_residency": residency_observation,
                }
                recommendation = self._vmm_build_recommendation(residency_observation)
                path = record_dynamic_offload_profile(self.context, observation, recommendation=recommendation)
                self.report.update(path=str(path), **observation)
                self.report["status"] = "saved"
                name = self.config.profile_name
                if recommendation is None:
                    self.report.setdefault("fit_class", "unavailable")
                    print(f"[dynamic-offload] VMM profile {name!r}: no residency budget saved "
                          f"({self.report.get('build_result', 'device observations unavailable')}); saved to {path}")
                else:
                    self.report.update(fit_class=recommendation["fit_class"], recommendation=dict(recommendation),
                                       resident_budget_gb=recommendation["vmm_resident_weight_budget_gb"],
                                       build_stage=recommendation["vmm_build_stage"],
                                       decision=recommendation.get("decision"))
                    if recommendation["vmm_build_stage"] == "restart_required":
                        print(f"[dynamic-offload] VMM profile {name!r}: the two builds saw different input "
                              f"shapes; the next build restarts from the floor; saved to {path}")
                    elif recommendation["vmm_build_stage"] == "floor_measured":
                        print(f"[dynamic-offload] VMM profile {name!r}: build 1/2 done; floor forward="
                              f"{recommendation['floor_forward_seconds']:.3f} s, hardware ceiling="
                              f"{recommendation['ceiling_budget_bytes'] / 1024**3:.3f} GiB "
                              f"({recommendation['fit_class']}); run the same build again to measure the ceiling; "
                              f"saved to {path}")
                    else:
                        gain = recommendation.get("gain_per_gib")
                        print(f"[dynamic-offload] VMM profile {name!r}: residency budget="
                              f"{recommendation['vmm_resident_weight_budget_gb']:.3f} GiB "
                              f"(decision={recommendation['decision']}, reason={recommendation['reason']}, "
                              f"gain_s={recommendation.get('gain_seconds')}, noise_s={recommendation.get('noise_seconds')}, "
                              f"gain_per_gib={'unavailable' if gain is None else f'{gain * 100:.3f}%'}, "
                              f"threshold={recommendation['min_gain_per_gib'] * 100:.3f}%); saved to {path}")
            except BaseException:
                self.report["status"] = "failed"
                raise
            finally:
                residency.close()
                self._active = False
            return
        if not self.config.build_profile and self._sdnq_probe_budget_gb is None:
            yield self.report
            return
        if self._active:
            raise RuntimeError("A profile measurement is already active.")
        self._active = True
        device = torch.device(self.config.execution_device)
        sampler = _CudaPeakSampler(device) if self.config.execution_mode == "sdnq_runtime" else None
        try:
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            external_bytes = max(0, total - free - torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
            if sampler is not None:
                sampler.start()
            yield self.report
            torch.cuda.synchronize(device)
            free_end, _ = torch.cuda.mem_get_info(device)
            peak = max(
                total - free,
                total - free_end,
                external_bytes + torch.cuda.max_memory_reserved(device),
                sampler.stop() if sampler is not None else 0,
            ) / 1024**3
            total_gb = total / 1024**3
            target_gb = max(0.0, total_gb - self.config.profile_vram_headroom_gb)
            if self._sdnq_probe_budget_gb is not None:
                budget = max(0.0, self._sdnq_probe_budget_gb - max(0.0, peak - target_gb))
                recommendation = {"profile_resident_module_budget_gb": round(budget, 4)}
                observation_kind = "sdnq_probe"
            elif self.config.execution_mode == "sdnq_runtime":
                safe_budget = DynamicOffloadProfileSession.recommend_resident_budget(
                    total_gb, peak, self.config.profile_vram_headroom_gb,
                )
                configured_ceiling = float(self.config.max_resident_module_budget_gb)
                ceiling = configured_ceiling if configured_ceiling > 0 else _SDNQ_DEFAULT_RESIDENT_CEILING_GB
                probe_budget = min(ceiling, target_gb)
                if safe_budget >= _SDNQ_PROBE_MIN_SAFE_BUDGET_GB and probe_budget > safe_budget:
                    budget = probe_budget
                    recommendation = {
                        "profile_resident_module_budget_gb": round(budget, 4),
                        "sdnq_probe_resident_budget_gb": round(budget, 4),
                    }
                    observation_kind = "sdnq_baseline_probe_pending"
                else:
                    budget = safe_budget
                    recommendation = {"profile_resident_module_budget_gb": round(budget, 4)}
                    observation_kind = "sdnq_baseline_safe"
            else:
                budget = DynamicOffloadProfileSession.recommend_resident_budget(
                    total_gb, peak, self.config.profile_vram_headroom_gb,
                )
                recommendation = {"profile_resident_module_budget_gb": budget}
                observation_kind = "baseline"
            observation = {"status": "success", "peak_vram_gb": peak,
                           "total_vram_gb": total_gb,
                           "headroom_gb": self.config.profile_vram_headroom_gb,
                           "profile_measurement_kind": observation_kind,
                           "measurement": "cuda_driver_peak_sampled_plus_allocator"}
            path = record_dynamic_offload_profile(
                self.context, observation,
                recommendation=recommendation,
            )
            self.report.update(path=str(path), resident_budget_gb=budget, **observation)
            self.report["status"] = "saved"
            if observation_kind == "sdnq_baseline_probe_pending":
                print(
                    f"[dynamic-offload] profile {self.config.profile_name!r}: "
                    f"baseline peak={peak:.2f} GiB; next SDNQ run will validate "
                    f"resident probe={budget:.2f} GiB; saved to {path}"
                )
                return
            print(f"[dynamic-offload] profile {self.config.profile_name!r}: "
                  f"peak={peak:.2f} GiB, margin={self.config.profile_vram_headroom_gb:.2f} GiB, "
                  f"resident budget={budget:.2f} GiB; saved to {path}")
        except BaseException:
            self.report["status"] = "failed"
            raise
        finally:
            if sampler is not None:
                sampler.stop()
            self._active = False


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation of ``value``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"DDO profile context contains a non-JSON value: {type(value).__name__}")


def normalize_dynamic_offload_profile_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the host-provided profile identity context."""
    if not isinstance(context, Mapping) or not context:
        raise ValueError("DDO profile context must be a non-empty mapping.")
    normalized = _json_value(context)
    if not isinstance(normalized, dict):  # Defensive for type checkers and future changes.
        raise TypeError("DDO profile context must normalize to an object.")
    return normalized


def dynamic_offload_profile_key(context: Mapping[str, Any]) -> str:
    """Return the stable, privacy-preserving filename key for a context."""
    normalized = normalize_dynamic_offload_profile_context(context)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    prefix = _VMM_PROFILE_PREFIX if normalized.get("runtime") == "vmm" else _PROFILE_PREFIX
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:24]}"


def get_dynamic_offload_profile_dir(profile_dir: str | Path | None = None) -> Path:
    """Return the profile directory, honoring ``DDO_PROFILE_DIR`` when set."""
    raw_directory = profile_dir if profile_dir is not None else os.getenv("DDO_PROFILE_DIR")
    return Path(raw_directory).expanduser() if raw_directory else Path.home() / ".ddo" / "profiles"


def dynamic_offload_profile_path(context: Mapping[str, Any], profile_dir: str | Path | None = None) -> Path:
    """Return the JSON path assigned to ``context`` without creating anything."""
    return get_dynamic_offload_profile_dir(profile_dir) / f"{dynamic_offload_profile_key(context)}.json"


def load_dynamic_offload_profile(
    context: Mapping[str, Any], profile_dir: str | Path | None = None
) -> dict[str, Any] | None:
    """Load a matching profile, returning ``None`` for missing or invalid data."""
    expected_context = normalize_dynamic_offload_profile_context(context)
    path = dynamic_offload_profile_path(expected_context, profile_dir)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("schema_version") != DYNAMIC_OFFLOAD_PROFILE_SCHEMA_VERSION:
        return None
    if record.get("profile_key") != dynamic_offload_profile_key(expected_context):
        return None
    if record.get("context") != expected_context:
        return None
    return record


def _merge_capacity_limits(existing: Mapping[str, Any] | None, observation: Mapping[str, Any]) -> dict[str, Any]:
    limits = dict(existing or {})
    capacity = observation.get("capacity")
    if not isinstance(capacity, Mapping):
        return limits
    metric = capacity.get("metric")
    value = capacity.get("value")
    status = observation.get("status")
    if not isinstance(metric, str) or not metric or not isinstance(value, (int, float)):
        return limits
    previous = dict(limits.get(metric) or {})
    if status == "success":
        previous["max_success"] = max(float(value), float(previous.get("max_success", value)))
    elif status == "failure":
        previous["min_failure"] = min(float(value), float(previous.get("min_failure", value)))
    else:
        return limits
    limits[metric] = previous
    return limits


def record_dynamic_offload_profile(
    context: Mapping[str, Any],
    observation: Mapping[str, Any],
    profile_dir: str | Path | None = None,
    recommendation: Mapping[str, Any] | None = None,
) -> Path:
    """Persist an observation and merge generic capacity limits for its context.

    ``observation`` may include ``{"status": "success", "capacity":
    {"metric": "tokens", "value": 123}}``.  DDO keeps the largest success
    and the smallest failure per metric, without knowing model-specific terms.
    """
    normalized_context = normalize_dynamic_offload_profile_context(context)
    normalized_observation = _json_value(observation)
    normalized_recommendation = None if recommendation is None else _json_value(recommendation)
    if not isinstance(normalized_observation, dict):
        raise TypeError("DDO profile observation must normalize to an object.")
    if normalized_recommendation is not None and not isinstance(normalized_recommendation, dict):
        raise TypeError("DDO profile recommendation must normalize to an object.")
    path = dynamic_offload_profile_path(normalized_context, profile_dir)
    prior = load_dynamic_offload_profile(normalized_context, profile_dir) or {}
    observations = list(prior.get("observations") or [])
    observations.append(normalized_observation)
    record = {
        "schema_version": DYNAMIC_OFFLOAD_PROFILE_SCHEMA_VERSION,
        "profile_key": dynamic_offload_profile_key(normalized_context),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "context": normalized_context,
        "capacity_limits": _merge_capacity_limits(prior.get("capacity_limits"), normalized_observation),
        "observations": observations[-20:],
    }
    if normalized_recommendation is not None:
        record["recommendation"] = normalized_recommendation
    elif isinstance(prior.get("recommendation"), Mapping):
        record["recommendation"] = dict(prior["recommendation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)
    return path


def get_dynamic_offload_profile_capacity(profile: Mapping[str, Any], metric: str) -> dict[str, float] | None:
    """Return validated limits for a generic capacity metric, if present."""
    limits = profile.get("capacity_limits")
    if not isinstance(limits, Mapping) or not isinstance(limits.get(metric), Mapping):
        return None
    result = {key: float(value) for key, value in limits[metric].items() if key in {"max_success", "min_failure"}}
    return result or None


def get_dynamic_offload_profile_recommendation(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a JSON-safe host recommendation stored with a matching profile."""
    recommendation = profile.get("recommendation")
    return dict(recommendation) if isinstance(recommendation, Mapping) else None


@dataclass
class DynamicOffloadProfileSession:
    """One generic lifecycle for exact presets plus family capacity history."""

    execution_context: Mapping[str, Any]
    capacity_context: Mapping[str, Any]
    profile_dir: str | Path | None = None
    execution_profile: dict[str, Any] | None = field(init=False)
    capacity_profile: dict[str, Any] | None = field(init=False)

    def __post_init__(self):
        self.execution_profile = load_dynamic_offload_profile(self.execution_context, self.profile_dir)
        self.capacity_profile = load_dynamic_offload_profile(self.capacity_context, self.profile_dir)

    def apply_settings(self, settings: Any, *, allow_recommendation: bool = True) -> tuple[Any, dict[str, Any] | None]:
        """Apply DDO's known resident-budget recommendation, if one exists."""
        recommendation = get_dynamic_offload_profile_recommendation(self.execution_profile or {})
        if not allow_recommendation or not recommendation:
            return settings, recommendation
        if settings.config.resident_module_budget_gb > 0:
            return settings, None
        budget = recommendation.get("profile_resident_module_budget_gb")
        if budget is None:
            # Read profiles written by DDO versions that exposed the old
            # environment variable, without retaining that public setting.
            budget = recommendation.get("auto_full_pin_resident_budget_gb")
        if budget is None:
            return settings, recommendation
        return replace(
            settings,
            config=replace(settings.config, profile_resident_module_budget_gb=max(0.0, float(budget))),
        ), recommendation

    @staticmethod
    def recommend_resident_budget(total_vram_gb: float, peak_vram_gb: float, headroom_gb: float) -> float:
        """Convert a zero-resident calibration peak into a conservative DDO budget."""
        return round(max(0.0, float(total_vram_gb) - float(peak_vram_gb) - float(headroom_gb)), 4)

    def record_success(
        self, observation: Mapping[str, Any], recommendation: Mapping[str, Any] | None = None
    ) -> tuple[Path, Path]:
        """Record an exact observation/recommendation and family capacity observation."""
        execution_path = record_dynamic_offload_profile(
            self.execution_context, observation, self.profile_dir, recommendation=recommendation
        )
        capacity_path = record_dynamic_offload_profile(self.capacity_context, observation, self.profile_dir)
        self.execution_profile = load_dynamic_offload_profile(self.execution_context, self.profile_dir)
        self.capacity_profile = load_dynamic_offload_profile(self.capacity_context, self.profile_dir)
        return execution_path, capacity_path
