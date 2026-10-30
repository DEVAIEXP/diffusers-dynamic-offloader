"""Public backend selection contract for thin host runners."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

from .dynamic_offload import (
    DynamicOffloadConfig,
    dynamic_offload_preset_env_value,
    resolve_dynamic_offload_preset,
)

_BACKENDS = ("linear", "vmm")
_VMM_PRESETS = ("auto", "nvme_baseline")


@dataclass(frozen=True)
class DynamicOffloadBackend:
    """The resolved route a runner passes to DDO.

    ``backend`` is intentionally distinct from ``effective_preset``: the
    stable linear default keeps its existing preset semantics, while VMM has
    only an automatic native route and its protected NVMe fallback.
    """

    backend: str
    requested_preset: str
    effective_preset: str
    execution_mode: str

    @property
    def requires_native_bootstrap(self) -> bool:
        return self.effective_preset == "vmm_fast"

    def build_config(self, **overrides: Any) -> DynamicOffloadConfig:
        """Create the DDO configuration without exposing runtime internals to a host.

        A runner may provide normal DDO options such as the VMM capacity and
        staging budgets, but it never needs to select an execution mode.
        """
        if "execution_mode" in overrides:
            raise ValueError("execution_mode is selected by the resolved DDO backend")
        if overrides.get("vmm_component_tensor_ownership") and self.execution_mode != "vmm_fast_runtime":
            raise ValueError("component ownership is exclusive to experimental vmm_fast")
        if overrides.get("vmm_execution_auxiliary_ownership") and self.execution_mode != "vmm_fast_runtime":
            raise ValueError("execution auxiliary ownership is exclusive to experimental vmm_fast")
        return DynamicOffloadConfig(execution_mode=self.execution_mode, **overrides)

    def build_config_from_args(self, args: argparse.Namespace) -> DynamicOffloadConfig:
        """Build the route configuration from the standard runner arguments."""
        staging = args.ddo_vmm_staging_buffer_gb
        total = getattr(args, "ddo_vmm_total_staging_gb", None)
        if total is not None:
            if not math.isfinite(total) or total <= 0:
                raise ValueError("--ddo-vmm-total-staging-gb must be finite and positive")
            # Protected NVMe config is per slot; native config is total capacity.
            staging = total / 2 if self.execution_mode == "vmm_runtime" else total
        return self.build_config(
            # Hosts using explicit configs bypass the settings environment loader.
            # Resolve this opt-in here, within DDO, only for the native fast route.
            vmm_execution_backing_reclaim=(
                self.execution_mode == "vmm_fast_runtime"
                and dynamic_offload_preset_env_value(
                    "DDO_VMM_EXECUTION_BACKING_RECLAIM", "0",
                    effective_preset=self.effective_preset,
                ).strip().lower() in {"1", "true", "yes", "on"}
            ),
            vmm_resident_weight_budget_gb=args.ddo_vmm_resident_weight_budget_gb,
            vmm_staging_buffer_gb=staging,
            vmm_native_transport=getattr(args, "ddo_vmm_native_transport", False),
            vmm_native_source_cache_slot_mb=getattr(args, "ddo_vmm_native_source_cache_slot_mb", 0),
            vmm_native_source_cache_slots=getattr(args, "ddo_vmm_native_source_cache_slots", 0),
            vmm_source_cache_gb=getattr(args, "ddo_vmm_source_cache_gb", 0.0),
            vmm_source_registered_gb=getattr(args, "ddo_vmm_source_registered_gb", 0.0),
            vmm_component_tensor_ownership=getattr(args, "ddo_vmm_component_tensor_ownership", False),
            vmm_execution_auxiliary_ownership=getattr(args, "ddo_vmm_execution_auxiliary_ownership", False),
            vmm_component_buffer_limit_mb=getattr(args, "ddo_vmm_component_buffer_limit_mb", 1),
        )


def resolve_dynamic_offload_backend(
    backend: str = "linear", preset: str = "auto", *, running_on_wsl: bool | None = None
) -> DynamicOffloadBackend:
    """Resolve CLI backend and preset without requiring environment variables.

    ``linear/auto`` remains the established ``one_shot_fast`` default. For
    ``vmm``, ``auto`` selects the native fast route and an explicit
    ``nvme_baseline`` selects the known-safe VMM fallback.
    """
    normalized_backend = backend.strip().lower()
    normalized_preset = preset.strip().lower()
    if normalized_backend not in _BACKENDS:
        raise ValueError(f"Invalid DDO backend {backend!r}. Valid values: {', '.join(_BACKENDS)}")
    if normalized_backend == "linear":
        return DynamicOffloadBackend(
            backend="linear",
            requested_preset=normalized_preset,
            effective_preset=resolve_dynamic_offload_preset(normalized_preset, running_on_wsl=running_on_wsl),
            execution_mode="linear_runtime",
        )
    if normalized_preset not in _VMM_PRESETS:
        raise ValueError(f"Invalid VMM preset {preset!r}. Valid values: {', '.join(_VMM_PRESETS)}")
    return DynamicOffloadBackend(
        backend="vmm",
        requested_preset=normalized_preset,
        effective_preset="vmm_fast" if normalized_preset == "auto" else "nvme_baseline",
        execution_mode="vmm_fast_runtime" if normalized_preset == "auto" else "vmm_runtime",
    )


def add_dynamic_offload_lifecycle_arguments(
    parser: argparse.ArgumentParser,
    phases: tuple[str, ...] = ("before_run", "after_text_encoder", "before_transformer", "after_run"),
) -> argparse.ArgumentParser:
    """Expose tri-state Windows lifecycle controls without mutating the environment."""
    for phase in phases:
        if phase not in ("before_run", "after_text_encoder", "before_transformer", "after_run"):
            raise ValueError(f"Unsupported standby lifecycle phase: {phase}")
        parser.add_argument(
            "--purge-" + phase.replace("_", "-"),
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Windows standby purge: explicit CLI overrides environment/preset; omitted inherits them.",
        )
    return parser


def add_dynamic_offload_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the standard DDO route arguments to a host runner parser."""
    parser.add_argument("--ddo-backend", choices=_BACKENDS, default="linear", help="DDO backend (default: linear).")
    parser.add_argument("--ddo-preset", default="auto", help="DDO preset; VMM accepts auto or nvme_baseline.")
    parser.add_argument(
        "--ddo-vmm-resident-weight-budget-gb",
        type=float,
        default=4.0,
        help="Native VMM active-residency capacity in GiB (default: 4).",
    )
    parser.add_argument(
        "--ddo-vmm-total-staging-gb", type=float, default=None,
        help="Total reusable host-staging capacity in GiB for either VMM route; omitted preserves legacy defaults.",
    )
    parser.add_argument(
        "--ddo-vmm-staging-buffer-gb",
        type=float,
        default=0.625,
        help="Legacy route-specific staging value; prefer --ddo-vmm-total-staging-gb for total capacity.",
    )
    parser.add_argument(
        "--ddo-vmm-native-transport", action="store_true",
        help="Use the experimental vmm_fast native host ring, also with source cache disabled.",
    )
    parser.add_argument(
        "--ddo-vmm-native-source-cache-slot-mb", type=int, default=0,
        help="Experimental vmm_fast retained-source slot size in MiB (0 disables; maximum 64).",
    )
    parser.add_argument(
        "--ddo-vmm-native-source-cache-slots", type=int, default=0,
        help="Retained-source slots (set with slot size; current total cache maximum 128 MiB).",
    )
    parser.add_argument("--show-profile", action="store_true", help="Print the DDO profile report.")
    parser.add_argument(
        "--ddo-vmm-source-cache-gb", type=float, default=0.0,
        help="Opt-in native shared-window source capacity in GiB (maximum 32; lazy registration; default off).",
    )
    parser.add_argument(
        "--ddo-vmm-source-registered-gb", type=float, default=0.0,
        help="Separate CUDA registration ceiling within retained source storage, in GiB (0 uses the storage ceiling).",
    )
    parser.add_argument("--ddo-vmm-component-tensor-ownership", action="store_true",
                        help="Experimental whole-component ownership; requires native file transport; selected text-only workloads validated, not general encoder support.")
    parser.add_argument("--ddo-vmm-execution-auxiliary-ownership", action="store_true",
                        help="Experimental auxiliary parameters in native execution-unit batches; unchanged GPU ceiling; default off.")
    parser.add_argument("--ddo-vmm-component-buffer-limit-mb", type=int, default=1,
                        help="Explicit device-buffer ceiling in MiB for the opt-in component route.")
    parser.add_argument("--show-metrics", action="store_true", help="Print the DDO metrics JSON.")
    return parser
