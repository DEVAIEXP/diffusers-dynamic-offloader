"""Persistent, model-agnostic capacity profiles for Dynamic Offloader.

Profiles deliberately store observations, not executable model code.  A host
runner supplies a stable context and records the outcome of a real workload;
DDO then makes that observation available to later runs on the same machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DYNAMIC_OFFLOAD_PROFILE_SCHEMA_VERSION = 1
_PROFILE_PREFIX = "ddo-capacity-v1"


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
    context: Mapping[str, Any], observation: Mapping[str, Any], profile_dir: str | Path | None = None
) -> Path:
    """Persist an observation and merge generic capacity limits for its context.

    ``observation`` may include ``{"status": "success", "capacity":
    {"metric": "tokens", "value": 123}}``.  DDO keeps the largest success
    and the smallest failure per metric, without knowing model-specific terms.
    """
    normalized_context = normalize_dynamic_offload_profile_context(context)
    normalized_observation = _json_value(observation)
    if not isinstance(normalized_observation, dict):
        raise TypeError("DDO profile observation must normalize to an object.")
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
