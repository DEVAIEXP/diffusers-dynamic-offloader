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
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

DYNAMIC_OFFLOAD_PROFILE_SCHEMA_VERSION = 1
_PROFILE_PREFIX = "ddo-capacity-v1"


class _NamedProfile:
    """Library-owned identity, application and measurement for a named workload."""

    def __init__(self, module, config):
        self.config = config
        self.context = {
            "name": config.profile_name,
            "model_class": type(module).__qualname__,
            "identity_version": 2,
        }
        self.report = {"name": config.profile_name, "status": "disabled"}
        self._active = False

    def prepare(self):
        config = self.config
        if config.build_profile and not config.profile_name.strip():
            raise ValueError("build_profile requires a non-empty profile_name.")
        if not config.profile_name:
            return config
        if config.build_profile:
            if torch.device(config.execution_device).type != "cuda":
                raise ValueError("Profile calibration requires a CUDA execution device.")
            if config.execution_mode != "linear_runtime":
                raise ValueError("Profile calibration requires execution_mode='linear_runtime'.")
            if not math.isfinite(config.profile_vram_headroom_gb) or config.profile_vram_headroom_gb < 0:
                raise ValueError("profile_vram_headroom_gb must be finite and non-negative.")
            self.report["status"] = "calibrating"
            print(f"[dynamic-offload] profile {config.profile_name!r}: calibrating with zero resident budget.")
            return replace(config, resident_module_budget_gb=0.0, profile_resident_module_budget_gb=0.0)
        record = load_dynamic_offload_profile(self.context)
        recommendation = get_dynamic_offload_profile_recommendation(record or {}) or {}
        budget = recommendation.get("profile_resident_module_budget_gb")
        if config.resident_module_budget_gb > 0:
            self.report["status"] = "manual_override"
            return config
        if not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
            self.report["status"] = "missing"
            print(f"[dynamic-offload] profile {config.profile_name!r}: no recommendation; calibrate with build_profile=True.")
            return config
        if config.max_resident_module_budget_gb > 0:
            budget = min(budget, config.max_resident_module_budget_gb)
        self.report.update(status="applied", resident_budget_gb=budget)
        print(f"[dynamic-offload] profile {config.profile_name!r}: requested resident budget={budget:.2f} GiB.")
        if config.auto_budget_policy == "off":
            return replace(config, resident_module_budget_gb=budget, _profile_budget_applied=True)
        return replace(config, profile_resident_module_budget_gb=budget, _profile_budget_applied=True)

    @contextmanager
    def measure(self):
        """Measure only the caller's inference region, then persist on success."""
        if not self.config.build_profile:
            yield self.report
            return
        if self._active:
            raise RuntimeError("A profile measurement is already active.")
        self._active = True
        device = torch.device(self.config.execution_device)
        try:
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            external_bytes = max(0, total - free - torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
            yield self.report
            torch.cuda.synchronize(device)
            free_end, _ = torch.cuda.mem_get_info(device)
            peak = max(total - free, total - free_end,
                       external_bytes + torch.cuda.max_memory_reserved(device)) / 1024**3
            budget = DynamicOffloadProfileSession.recommend_resident_budget(
                total / 1024**3, peak, self.config.profile_vram_headroom_gb,
            )
            observation = {"status": "success", "peak_vram_gb": peak,
                           "total_vram_gb": total / 1024**3,
                           "headroom_gb": self.config.profile_vram_headroom_gb,
                           "measurement": "cuda_peak_reserved_plus_initial_external"}
            path = record_dynamic_offload_profile(
                self.context, observation,
                recommendation={"profile_resident_module_budget_gb": budget},
            )
            self.report.update(path=str(path), resident_budget_gb=budget, **observation)
            self.report["status"] = "saved"
            print(f"[dynamic-offload] profile {self.config.profile_name!r}: "
                  f"peak={peak:.2f} GiB, margin={self.config.profile_vram_headroom_gb:.2f} GiB, "
                  f"resident budget={budget:.2f} GiB; saved to {path}")
        except BaseException:
            self.report["status"] = "failed"
            raise
        finally:
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
    return f"{_PROFILE_PREFIX}-{hashlib.sha256(encoded).hexdigest()[:24]}"


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
