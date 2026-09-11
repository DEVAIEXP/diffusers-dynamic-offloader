from __future__ import annotations

import contextlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.hooks.hooks import HookRegistry, ModelHook

_DEFAULT_TARGET_MODULE_CLASSES = (nn.Linear, nn.Embedding)
_DYNAMIC_OFFLOAD_HOOK = "dynamic_offload"
_DDO_ENV_PREFIX = "DDO_"
_PIN_MEMORY_ERRORS = (RuntimeError, getattr(torch, "AcceleratorError", RuntimeError))
_DEFAULT_ALWAYS_RESIDENT_MODULE_PATTERNS = (
    r"(^|\.)(proj_in|time_embed|prompt_adaln|norm_out|proj_out)(\.|$)",
)
_AUTO_BUDGET_DISABLED = "off"
_AUTO_BUDGET_BALANCED = "balanced"
_DYNAMIC_OFFLOAD_PLAN_CACHE: dict[tuple[Any, ...], "DynamicOffloadState"] = {}
_DYNAMIC_OFFLOAD_PLAN_CACHE_LOCK = threading.Lock()
_DYNAMIC_OFFLOAD_PINNED_TENSOR_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_DYNAMIC_OFFLOAD_PINNED_TENSOR_CACHE_LOCK = threading.Lock()


def dynamic_offload_env_value(name: str, default: str = "", environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(name, default)


def is_wsl_environment() -> bool:
    if os.name == "nt":
        return False
    try:
        release = os.uname().release.lower()
    except AttributeError:
        release = ""
    if "microsoft" in release or "wsl" in release:
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def get_available_system_ram_gb() -> float:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return psutil.virtual_memory().available / 1024**3
        except Exception:
            pass

    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / 1024**3
        except Exception:
            return 0.0
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            return page_size * available_pages / 1024**3
        except (AttributeError, OSError, ValueError):
            return 0.0

    return 0.0


def get_cuda_total_vram_gb(device: str | torch.device = "cuda:0") -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.get_device_properties(device).total_memory / 1024**3
    except Exception:
        return 0.0


def purge_windows_standby_cache() -> dict[str, Any]:
    """Purge the Windows standby page list using NtSetSystemInformation.

    This mirrors tools such as EmptyStandbyList while keeping the behavior optional
    and explicit. The caller must run with a token that can enable
    SeProfileSingleProcessPrivilege.
    """
    if os.name != "nt":
        raise RuntimeError("Windows standby cache purge is only available on Windows.")

    import ctypes
    from ctypes import wintypes

    system_memory_list_information = 80
    memory_purge_standby_list = 4
    token_adjust_privileges = 0x0020
    token_query = 0x0008
    se_privilege_enabled = 0x00000002
    error_not_all_assigned = 1300

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [
            ("PrivilegeCount", wintypes.DWORD),
            ("Luid", LUID),
            ("Attributes", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(LUID)]
    advapi32.LookupPrivilegeValueW.restype = wintypes.BOOL
    advapi32.AdjustTokenPrivileges.argtypes = [
        wintypes.HANDLE,
        wintypes.BOOL,
        ctypes.POINTER(TOKEN_PRIVILEGES),
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi32.AdjustTokenPrivileges.restype = wintypes.BOOL

    ntdll.NtSetSystemInformation.argtypes = [wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG]
    ntdll.NtSetSystemInformation.restype = wintypes.LONG

    token = wintypes.HANDLE()
    process = kernel32.GetCurrentProcess()
    if not advapi32.OpenProcessToken(process, token_adjust_privileges | token_query, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, "SeProfileSingleProcessPrivilege", ctypes.byref(luid)):
            raise ctypes.WinError(ctypes.get_last_error())

        privileges = TOKEN_PRIVILEGES(1, luid, se_privilege_enabled)
        ctypes.set_last_error(0)
        if not advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(privileges), 0, None, None):
            raise ctypes.WinError(ctypes.get_last_error())
        last_error = ctypes.get_last_error()
        if last_error == error_not_all_assigned:
            raise PermissionError("SeProfileSingleProcessPrivilege is not assigned to this process token.")
    finally:
        kernel32.CloseHandle(token)

    command = ctypes.c_int(memory_purge_standby_list)
    status = ntdll.NtSetSystemInformation(
        system_memory_list_information,
        ctypes.byref(command),
        ctypes.sizeof(command),
    )
    if status != 0:
        raise OSError(f"NtSetSystemInformation failed with NTSTATUS 0x{status & 0xFFFFFFFF:08X}")

    return {
        "system_information_class": system_memory_list_information,
        "command": memory_purge_standby_list,
        "privilege": "SeProfileSingleProcessPrivilege",
    }


def purge_windows_standby_cache_event(record_event, event_name: str, *, print_message: bool = True) -> dict[str, Any]:
    event_t0 = time.time()
    result = purge_windows_standby_cache()
    if record_event is not None:
        record_event(event_name, time.time() - event_t0, **result)
    if print_message:
        print(f"  [windows-memory] {event_name}: standby cache purge requested", flush=True)
    return result


def maybe_purge_windows_standby_cache(
    settings_or_phase: "DynamicOffloadSettings | str | None" = None,
    phase: str | None = None,
    *,
    settings: "DynamicOffloadSettings | None" = None,
    preset: str = "auto",
    execution_device: str | torch.device = "cuda:0",
    offload_device: str | torch.device = "cpu",
    running_on_wsl: bool | None = None,
    record_event: Any | None = None,
    event_name: str | None = None,
    default: str = "0",
    print_message: bool = True,
    ignore_errors: bool = True,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Purge Windows standby cache when preset/env enables it for a lifecycle phase.

    Call either `maybe_purge_windows_standby_cache(settings, "before_run")` or
    `maybe_purge_windows_standby_cache("before_run", preset="auto")`. It is a
    no-op outside Windows, so runners can call the same lifecycle hook on every
    platform while DDO decides whether a purge is useful.
    """
    resolved_settings = settings
    resolved_phase = phase
    if isinstance(settings_or_phase, str):
        resolved_phase = settings_or_phase if resolved_phase is None else resolved_phase
    elif settings_or_phase is not None and resolved_settings is None:
        resolved_settings = settings_or_phase

    if resolved_phase is None:
        raise TypeError("phase is required")

    if resolved_settings is None:
        resolved_settings = load_dynamic_offload_settings_from_env(
            execution_device=execution_device,
            offload_device=offload_device,
            running_on_wsl=running_on_wsl,
            environ=environ,
            default_preset=preset,
        )

    key = "DDO_PURGE_WINDOWS_STANDBY_" + resolved_phase.strip().upper()
    enabled = resolved_settings.preset_bool(key, default, environ)
    if not enabled:
        return {"purged": False, "reason": "disabled", "phase": resolved_phase}
    if os.name != "nt":
        return {"purged": False, "reason": "non_windows", "phase": resolved_phase}

    resolved_event_name = event_name or key.lower().removeprefix("ddo_")
    try:
        result = purge_windows_standby_cache_event(record_event, resolved_event_name, print_message=print_message)
    except Exception as exc:
        if not ignore_errors:
            raise
        result = {"purged": False, "reason": exc.__class__.__name__, "phase": resolved_phase, "error": str(exc)}
        if record_event is not None:
            record_event(resolved_event_name, 0.0, **result)
        if print_message:
            print(f"  [windows-memory] {resolved_event_name}: standby cache purge skipped ({exc})", flush=True)
        return result
    result.update({"purged": True, "phase": resolved_phase})
    return result

_DYNAMIC_OFFLOAD_PRESET_VALUES: dict[str, dict[str, str]] = {
    "off": {
        "DDO_EXECUTION_MODE": "plan",
        "DDO_PLAN": "0",
    },
    "diffusers_leaf_offload_compat": {
        "DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "1",
        "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD": "0",
        "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DDO_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "1",
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_TYPE": "leaf_level",
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_STREAM": "1",
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_RECORD_STREAM": "1",
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_LOW_CPU_MEM_USAGE": "1",
        "DDO_RUNNER_ATTENTION_BACKEND": "native",
        "DDO_EXECUTION_MODE": "plan",
        "DDO_PLAN": "0",
        "DDO_RUNNER_PRE_VAE_CLEANUP_REPEATS": "1",
    },
    "diffusers_offload_compat": {
        "DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "1",
        "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD": "0",
        "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DDO_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "1",
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_TYPE": "block_level",
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_STREAM": "1",
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_RECORD_STREAM": "1",
        "DDO_RUNNER_TRANSFORMER_OFFLOAD_LOW_CPU_MEM_USAGE": "1",
        "DDO_RUNNER_TRANSFORMER_NUM_BLOCKS_PER_GROUP": "1",
        "DDO_RUNNER_ATTENTION_BACKEND": "native",
        "DDO_EXECUTION_MODE": "plan",
        "DDO_PLAN": "0",
        "DDO_RUNNER_PRE_VAE_CLEANUP_REPEATS": "1",
    },
    "one_shot_fast": {
        "DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "1",
        "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD": "0",
        "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DDO_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DDO_RUNNER_ATTENTION_BACKEND": "native",
        "DDO_EXECUTION_MODE": "linear_runtime",
        "DDO_PIN_CPU_MEMORY": "1",
        "DDO_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DDO_PIN_CPU_WORKERS": "4",
        "DDO_AUTO_BUDGET_POLICY": "balanced",
        "DDO_MAX_RESIDENT_MODULE_BUDGET_GB": "0",
        "DDO_SYSTEM_RAM_HEADROOM_GB": "8",
        "DDO_RESIDENT_MODULE_BUDGET_GB": "0",
        "DDO_RESIDENT_MODULE_PATTERNS": "auto",
        "DDO_RESIDENT_MODULE_SELECTION": "spread",
        "DDO_SMALL_TENSOR_THRESHOLD_KB": "1024",
        "DDO_RUNNER_PRE_VAE_CLEANUP_REPEATS": "1",
        "DDO_PURGE_WINDOWS_STANDBY_BEFORE_RUN": "1",
        "DDO_PURGE_WINDOWS_STANDBY_AFTER_TEXT_ENCODER": "1",
        "DDO_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER": "1",
    },
    "wsl_compat": {
        "DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "0",
        "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DDO_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DDO_RUNNER_ATTENTION_BACKEND": "native",
        "DDO_EXECUTION_MODE": "linear_runtime",
        "DDO_PIN_CPU_MEMORY": "0",
        "DDO_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DDO_DISABLE_PIN_ON_WSL": "1",
        "DDO_PIN_CPU_WORKERS": "1",
        "DDO_AUTO_BUDGET_POLICY": "off",
        "DDO_RESIDENT_MODULE_BUDGET_GB": "6",
        "DDO_RESIDENT_MODULE_PATTERNS": "auto",
        "DDO_RESIDENT_MODULE_SELECTION": "spread",
        "DDO_SMALL_TENSOR_THRESHOLD_KB": "1024",
        "DDO_RUNNER_PRE_VAE_CLEANUP_REPEATS": "3",
    },
    "warm_process": {
        "DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "1",
        "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD": "0",
        "DDO_RUNNER_GENERATION_REPEATS": "2",
        "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DDO_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DDO_RUNNER_ATTENTION_BACKEND": "native",
        "DDO_EXECUTION_MODE": "linear_runtime",
        "DDO_PIN_CPU_MEMORY": "1",
        "DDO_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DDO_PIN_CPU_WORKERS": "4",
        "DDO_CACHE_PINNED_WEIGHTS": "1",
        "DDO_AUTO_BUDGET_POLICY": "off",
        "DDO_RESIDENT_MODULE_BUDGET_GB": "6",
        "DDO_RESIDENT_MODULE_PATTERNS": "auto",
        "DDO_RESIDENT_MODULE_SELECTION": "spread",
        "DDO_SMALL_TENSOR_THRESHOLD_KB": "1024",
    },
    "low_ram_safe": {
        "DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD": "1",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_TYPE": "leaf_level",
        "DDO_RUNNER_TEXT_ENCODER_OFFLOAD_STREAM": "0",
        "DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD": "0",
        "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER": "off",
        "DDO_RUNNER_TRANSFORMER_GROUP_OFFLOAD": "0",
        "DDO_RUNNER_ATTENTION_BACKEND": "native",
        "DDO_EXECUTION_MODE": "linear_runtime",
        "DDO_PIN_CPU_MEMORY": "0",
        "DDO_ALLOW_PIN_MEMORY_FALLBACK": "1",
        "DDO_PIN_CPU_WORKERS": "1",
        "DDO_AUTO_BUDGET_POLICY": "off",
        "DDO_RESIDENT_MODULE_BUDGET_GB": "3",
        "DDO_RESIDENT_MODULE_PATTERNS": "auto",
        "DDO_RESIDENT_MODULE_SELECTION": "spread",
        "DDO_SMALL_TENSOR_THRESHOLD_KB": "1024",
    },
}


DDO_PRESETS = _DYNAMIC_OFFLOAD_PRESET_VALUES


def get_dynamic_offload_presets() -> dict[str, dict[str, str]]:
    return {preset_name: dict(values) for preset_name, values in DDO_PRESETS.items()}


def format_dynamic_offload_presets(*, default_preset: str = "auto", running_on_wsl: bool | None = None) -> str:
    resolved_default = resolve_dynamic_offload_preset(default_preset, running_on_wsl=running_on_wsl)
    lines = [
        "DDO presets",
        f"  {default_preset} -> {resolved_default}",
    ]
    for preset_name in sorted(DDO_PRESETS):
        lines.append(f"\n{preset_name}")
        preset_values = DDO_PRESETS[preset_name]
        if not preset_values:
            lines.append("  <no preset values>")
            continue
        for key in sorted(preset_values):
            lines.append(f"  {key}={preset_values[key]}")
    return "\n".join(lines)


def resolve_dynamic_offload_preset(requested_preset: str, *, running_on_wsl: bool | None = None) -> str:
    requested_preset = requested_preset.strip().lower()
    if requested_preset != "auto":
        if requested_preset and requested_preset not in DDO_PRESETS:
            valid_presets = ", ".join(["auto", *sorted(DDO_PRESETS)])
            raise ValueError(
                f"Invalid DDO_PRESET={requested_preset!r}. "
                f"Valid values: {valid_presets}"
            )
        return requested_preset

    return "one_shot_fast"


def dynamic_offload_preset_env_value(
    name: str,
    default: str = "",
    *,
    requested_preset: str = "",
    effective_preset: str | None = None,
    running_on_wsl: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    if name in env:
        return env[name]

    preset_name = effective_preset
    if preset_name is None:
        preset_name = resolve_dynamic_offload_preset(requested_preset, running_on_wsl=running_on_wsl)
    if not preset_name:
        return default

    preset_values = DDO_PRESETS[preset_name]
    return preset_values.get(name, default)


def _parse_bool_value(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_pattern_list_value(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[;,]", value) if item.strip())


@dataclass(frozen=True)
class DynamicOffloadConfig:
    """Configuration for the experimental dynamic offload planner/runtime."""

    execution_device: str | torch.device = "cuda:0"
    offload_device: str | torch.device = "cpu"
    target_module_classes: tuple[type[nn.Module], ...] = _DEFAULT_TARGET_MODULE_CLASSES
    skip_modules_pattern: tuple[str, ...] = ()
    always_resident_modules_pattern: tuple[str, ...] = ()
    small_tensor_threshold_bytes: int = 16 * 1024
    execution_mode: str = "plan"
    linear_runtime_strategy: str = "functional"
    pin_cpu_memory: bool = False
    allow_pin_memory_fallback: bool = True
    pin_cpu_workers: int = 1
    cache_plan: bool = True
    cache_pinned_weights: bool = False
    pinned_weight_cache_namespace: str = ""
    auto_budget_policy: str = _AUTO_BUDGET_DISABLED
    max_resident_module_budget_gb: float = 6.0
    auto_vram_headroom_gb: float = 0.0
    auto_full_pin_min_model_to_vram_ratio: float = 4.0
    auto_full_pin_resident_budget_gb: float = 0.0
    max_pin_weight_budget_gb: float = 0.0
    available_system_ram_gb: float = 0.0
    system_ram_headroom_gb: float = 6.0
    pin_weight_budget_gb: float = 0.0
    pin_weight_budget_ratio: float = 0.0
    pin_weight_selection: str = "spread"
    resident_module_budget_gb: float = 0.0
    resident_module_patterns: tuple[str, ...] = ()
    resident_module_selection: str = "spread"
    load_safetensors_backend: str = ""
    verbose: bool = False
    show_profile: bool = True


@dataclass(frozen=True)
class DynamicOffloadSettings:
    """Environment-derived dynamic offload settings plus the resolved runtime config."""

    requested_preset: str
    effective_preset: str
    enabled: bool
    plan: bool
    config: DynamicOffloadConfig
    requested_pin_cpu_memory: bool
    effective_pin_cpu_memory: bool
    disable_pin_on_wsl: bool
    running_on_wsl: bool
    component_policies: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def execution_mode(self) -> str:
        return self.config.execution_mode

    def preset_value(self, name: str, default: str = "", environ: Mapping[str, str] | None = None) -> str:
        return dynamic_offload_preset_env_value(
            name,
            default,
            requested_preset=self.requested_preset,
            effective_preset=self.effective_preset,
            running_on_wsl=self.running_on_wsl,
            environ=environ,
        )

    def preset_bool(self, name: str, default: str = "0", environ: Mapping[str, str] | None = None) -> bool:
        return _parse_bool_value(self.preset_value(name, default, environ))

    def as_metrics(self) -> dict[str, Any]:
        config = self.config
        return {
            "dynamic_offload_requested_preset": self.requested_preset or None,
            "dynamic_offload_effective_preset": self.effective_preset or None,
            "dynamic_offload_execution_mode": config.execution_mode,
            "dynamic_offload_linear_runtime_strategy": config.linear_runtime_strategy,
            "dynamic_offload_pin_cpu_memory": self.requested_pin_cpu_memory,
            "dynamic_offload_effective_pin_cpu_memory": self.effective_pin_cpu_memory,
            "dynamic_offload_allow_pin_memory_fallback": config.allow_pin_memory_fallback,
            "dynamic_offload_disable_pin_on_wsl": self.disable_pin_on_wsl,
            "dynamic_offload_pin_cpu_workers": config.pin_cpu_workers,
            "dynamic_offload_cache_plan": config.cache_plan,
            "dynamic_offload_cache_pinned_weights": config.cache_pinned_weights,
            "dynamic_offload_pinned_weight_cache_namespace": config.pinned_weight_cache_namespace or None,
            "dynamic_offload_auto_budget_policy": config.auto_budget_policy,
            "dynamic_offload_max_resident_module_budget_gb": config.max_resident_module_budget_gb,
            "dynamic_offload_auto_vram_headroom_gb": config.auto_vram_headroom_gb,
            "dynamic_offload_auto_full_pin_min_model_to_vram_ratio": (
                config.auto_full_pin_min_model_to_vram_ratio
            ),
            "dynamic_offload_auto_full_pin_resident_budget_gb": config.auto_full_pin_resident_budget_gb,
            "dynamic_offload_max_pin_weight_budget_gb": config.max_pin_weight_budget_gb,
            "dynamic_offload_available_system_ram_gb": config.available_system_ram_gb,
            "dynamic_offload_system_ram_headroom_gb": config.system_ram_headroom_gb,
            "dynamic_offload_pin_weight_budget_gb": config.pin_weight_budget_gb,
            "dynamic_offload_pin_weight_budget_ratio": config.pin_weight_budget_ratio,
            "dynamic_offload_pin_weight_selection": config.pin_weight_selection,
            "dynamic_offload_small_tensor_threshold_kb": config.small_tensor_threshold_bytes // 1024,
            "dynamic_offload_resident_module_budget_gb": config.resident_module_budget_gb,
            "dynamic_offload_resident_module_patterns": config.resident_module_patterns,
            "dynamic_offload_resident_module_selection": config.resident_module_selection,
            "dynamic_offload_load_safetensors_backend": config.load_safetensors_backend or None,
            "dynamic_offload_show_profile": config.show_profile,
            "dynamic_offload_component_policies": dict(self.component_policies),
        }

    @classmethod
    def from_env(cls, **kwargs: Any) -> "DynamicOffloadSettings":
        return load_dynamic_offload_settings_from_env(**kwargs)


def build_dynamic_offload_event_payload(
    settings: DynamicOffloadSettings,
    state: "DynamicOffloadState",
    config: DynamicOffloadConfig | None = None,
) -> dict[str, Any]:
    summary = state.as_dict()
    config = config or settings.config
    return {
        "module_count": summary["module_count"],
        "total_gb": summary["total_gb"],
        "bytes_by_placement": summary["bytes_by_placement"],
        "execution_mode": config.execution_mode,
        "pin_cpu_memory": settings.effective_pin_cpu_memory,
        "allow_pin_memory_fallback": config.allow_pin_memory_fallback,
        "cache_plan": config.cache_plan,
        "cache_pinned_weights": config.cache_pinned_weights,
        "pinned_weight_cache_namespace": config.pinned_weight_cache_namespace or None,
        "load_safetensors_backend": config.load_safetensors_backend or None,
        "auto_budget_policy": config.auto_budget_policy,
        "max_resident_module_budget_gb": config.max_resident_module_budget_gb,
        "auto_full_pin_min_model_to_vram_ratio": config.auto_full_pin_min_model_to_vram_ratio,
        "auto_full_pin_resident_budget_gb": config.auto_full_pin_resident_budget_gb,
        "max_pin_weight_budget_gb": config.max_pin_weight_budget_gb,
        "available_system_ram_gb": config.available_system_ram_gb,
        "system_ram_headroom_gb": config.system_ram_headroom_gb,
        "pin_weight_budget_gb": config.pin_weight_budget_gb,
        "pin_weight_budget_ratio": config.pin_weight_budget_ratio,
        "pin_weight_selection": config.pin_weight_selection,
        "patched_module_count": summary["patched_module_count"],
        "resolved_resident_module_patterns": summary["resolved_resident_module_patterns"],
        "planner_decisions": summary["planner_decisions"],
        "setup_runtime": summary["setup_runtime"],
    }


def detect_quantized_backend_modules(module: nn.Module) -> dict[str, Any]:
    sdnq_module_count = sum(1 for child in module.modules() if hasattr(child, "sdnq_dequantizer"))
    if not sdnq_module_count:
        return {}
    return {"sdnq_module_count": sdnq_module_count, "sdnq_status": "detected_forward_preserved"}


@dataclass(frozen=True)
class DynamicOffloadPlanEntry:
    module_name: str
    module_type: str
    tensor_name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    bytes: int
    placement: str


@dataclass
class DynamicOffloadLoadResult:
    module: nn.Module
    hook: "DynamicOffloadHook | None" = None
    should_move_to_execution_device: bool = True


@dataclass
class DynamicOffloadApplyResult:
    module: nn.Module
    hook: "DynamicOffloadHook | None" = None
    settings: DynamicOffloadSettings | None = None
    should_move_to_execution_device: bool = True
    event_payload: dict[str, Any] = field(default_factory=dict)
    route: str = "dynamic_offload"


@dataclass
class DiffusersGroupOffloadResult:
    module: nn.Module
    event_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicOffloadState:
    entries: list[DynamicOffloadPlanEntry] = field(default_factory=list)
    module_count: int = 0
    total_bytes: int = 0
    bytes_by_placement: dict[str, int] = field(default_factory=dict)
    setup_seconds_by_action: dict[str, float] = field(default_factory=dict)
    setup_bytes_by_action: dict[str, int] = field(default_factory=dict)
    copy_seconds_by_name: dict[str, float] = field(default_factory=dict)
    copy_bytes_by_name: dict[str, int] = field(default_factory=dict)
    copy_calls_by_name: dict[str, int] = field(default_factory=dict)
    patched_module_count: int = 0
    selected_resident_modules: list[str] = field(default_factory=list)
    selected_resident_linear_weights: list[str] = field(default_factory=list)
    selected_pinned_linear_weights: list[str] = field(default_factory=list)
    resolved_resident_module_patterns: list[str] = field(default_factory=list)
    planner_decisions: dict[str, Any] = field(default_factory=dict)

    def clone_for_runtime(self) -> "DynamicOffloadState":
        return DynamicOffloadState(
            entries=list(self.entries),
            module_count=self.module_count,
            total_bytes=self.total_bytes,
            bytes_by_placement=dict(self.bytes_by_placement),
            resolved_resident_module_patterns=list(self.resolved_resident_module_patterns),
            planner_decisions=dict(self.planner_decisions),
        )

    def add_setup(self, action: str, seconds: float, byte_count: int = 0) -> None:
        self.setup_seconds_by_action[action] = self.setup_seconds_by_action.get(action, 0.0) + seconds
        self.setup_bytes_by_action[action] = self.setup_bytes_by_action.get(action, 0) + byte_count

    def add_copy(self, name: str, seconds: float, byte_count: int) -> None:
        self.copy_seconds_by_name[name] = self.copy_seconds_by_name.get(name, 0.0) + seconds
        self.copy_bytes_by_name[name] = self.copy_bytes_by_name.get(name, 0) + byte_count
        self.copy_calls_by_name[name] = self.copy_calls_by_name.get(name, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_count": self.module_count,
            "patched_module_count": self.patched_module_count,
            "total_gb": round(self.total_bytes / 1024**3, 4),
            "bytes_by_placement": {
                key: round(value / 1024**3, 4) for key, value in sorted(self.bytes_by_placement.items())
            },
            "setup_runtime": {
                key: {
                    "seconds": round(self.setup_seconds_by_action[key], 4),
                    "gb": round(self.setup_bytes_by_action.get(key, 0) / 1024**3, 4),
                }
                for key in sorted(self.setup_seconds_by_action)
            },
            "copy_runtime": {
                key: {
                    "calls": self.copy_calls_by_name.get(key, 0),
                    "seconds": round(self.copy_seconds_by_name[key], 4),
                    "gb": round(self.copy_bytes_by_name.get(key, 0) / 1024**3, 4),
                }
                for key in sorted(self.copy_seconds_by_name)
            },
            "selected_resident_modules": self.selected_resident_modules,
            "selected_resident_linear_weights": self.selected_resident_linear_weights,
            "selected_pinned_linear_weights": self.selected_pinned_linear_weights,
            "resolved_resident_module_patterns": self.resolved_resident_module_patterns,
            "planner_decisions": self.planner_decisions,
            "entries": [
                {
                    "module_name": entry.module_name,
                    "module_type": entry.module_type,
                    "tensor_name": entry.tensor_name,
                    "shape": list(entry.shape),
                    "dtype": entry.dtype,
                    "device": entry.device,
                    "gb": round(entry.bytes / 1024**3, 6),
                    "placement": entry.placement,
                }
                for entry in self.entries
            ],
        }


class DynamicOffloadHook(ModelHook):
    def __init__(self, config: DynamicOffloadConfig) -> None:
        super().__init__()
        self.config = config
        self.state = DynamicOffloadState()
        self._patched_modules: list[tuple[nn.Module, object]] = []
        self._pin_memory_disabled = False
        self._skip_pin_weights = False
        self._force_full_pin_weights = False
        self._resident_module_names: set[str] = set()
        self._pinned_weight_cache_namespace = ""
        self.execution_device = torch.device(config.execution_device)
        self.offload_device = torch.device(config.offload_device)
        self.execution_mode = config.execution_mode.lower()
        self.linear_runtime_strategy = config.linear_runtime_strategy.lower()
        if self.linear_runtime_strategy not in {"functional", "swap"}:
            raise ValueError("DynamicOffloadConfig.linear_runtime_strategy must be 'functional' or 'swap'")
        self.pin_cpu_workers = max(1, int(config.pin_cpu_workers))
        self.auto_budget_policy = config.auto_budget_policy.lower()
        if self.auto_budget_policy not in {_AUTO_BUDGET_DISABLED, _AUTO_BUDGET_BALANCED}:
            raise ValueError("DynamicOffloadConfig.auto_budget_policy must be 'off' or 'balanced'")
        self.max_resident_module_budget_bytes = int(max(0.0, float(config.max_resident_module_budget_gb)) * 1024**3)
        self.max_pin_weight_budget_bytes = int(max(0.0, float(config.max_pin_weight_budget_gb)) * 1024**3)
        self.available_system_ram_bytes = int(max(0.0, float(config.available_system_ram_gb)) * 1024**3)
        self.system_ram_headroom_bytes = int(max(0.0, float(config.system_ram_headroom_gb)) * 1024**3)
        self.pin_weight_budget_bytes = int(max(0.0, float(config.pin_weight_budget_gb)) * 1024**3)
        self.pin_weight_selection = config.pin_weight_selection.lower()
        if self.pin_weight_selection not in {"first", "spread", "largest"}:
            raise ValueError("DynamicOffloadConfig.pin_weight_selection must be 'first', 'spread', or 'largest'")
        self.pin_weight_budget_ratio = max(0.0, min(1.0, float(config.pin_weight_budget_ratio)))
        self.resident_module_budget_bytes = int(max(0.0, float(config.resident_module_budget_gb)) * 1024**3)
        self.resident_module_selection = config.resident_module_selection.lower()
        if self.resident_module_selection not in {"first", "spread", "largest"}:
            raise ValueError("DynamicOffloadConfig.resident_module_selection must be 'first', 'spread', or 'largest'")

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        self._pinned_weight_cache_namespace = (
            self.config.pinned_weight_cache_namespace.strip()
            or f"{module.__class__.__module__}.{module.__class__.__qualname__}"
        )
        self.state = build_dynamic_offload_plan(module, self.config)
        if self.execution_mode not in {"plan", "linear_runtime"}:
            raise ValueError("DynamicOffloadConfig.execution_mode must be 'plan' or 'linear_runtime'")
        if self.execution_mode == "linear_runtime":
            self._prepare_linear_runtime(module)
        if self.config.verbose:
            summary = self.state.as_dict()
            print(
                "  [dynamic-offload] "
                f"mode={self.execution_mode} modules={summary['module_count']} "
                f"patched={summary['patched_module_count']} total_gb={summary['total_gb']:.4f} "
                f"placements={summary['bytes_by_placement']}",
                flush=True,
            )
            if self.config.pin_cpu_memory and self.pin_weight_budget_bytes > 0:
                print(
                    "  [dynamic-offload] warning: partial pinned-memory budgets can be slower than full pinning; "
                    "clear DDO_PIN_WEIGHT_BUDGET_GB "
                    "when benchmarking the fast preset.",
                    flush=True,
                )
            if summary["setup_runtime"]:
                print(f"  [dynamic-offload] setup={summary['setup_runtime']}", flush=True)
        auto_pin_decision = self.state.planner_decisions.get("auto_pin_weight_decision")
        if auto_pin_decision in {"full_pin_zero_resident", "full_pin_residual_vram"}:
            decisions = self.state.planner_decisions
            selected_resident_gb = decisions.get("selected_resident_module_gb", 0.0)
            resident_note = (
                "no extra resident modules"
                if selected_resident_gb <= 0
                else f"{selected_resident_gb:.2f} GiB of resident modules"
            )
            print(
                f"  [dynamic-offload] auto plan: full pinned CPU weights with {resident_note} "
                f"(weights={decisions['candidate_weight_gb']:.2f} GiB, "
                f"usable_ram={decisions['auto_usable_system_ram_gb']:.2f} GiB, "
                f"model_to_vram={decisions['auto_model_to_vram_ratio']:.2f}x).",
                flush=True,
            )
        elif self.auto_budget_policy == _AUTO_BUDGET_BALANCED:
            decisions = self.state.planner_decisions
            print(
                "  [dynamic-offload] auto plan: "
                f"pin={auto_pin_decision or 'not_applicable'}, "
                f"resident_budget={decisions.get('auto_resident_module_budget_gb', 0.0):.2f} GiB, "
                f"pin_budget={decisions.get('auto_pin_weight_budget_gb', 0.0):.2f} GiB.",
                flush=True,
            )
        return module

    def detach_hook(self, module: nn.Module) -> nn.Module:
        self._restore_patches()
        return module

    def deinitalize_hook(self, module: nn.Module) -> nn.Module:
        self._restore_patches()
        return module

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.execution_mode == "plan":
            return args, kwargs
        return (
            tuple(self._move_value_to_execution_device(value, "forward_input") for value in args),
            {
                key: self._move_value_to_execution_device(value, f"forward_input:{key}")
                for key, value in kwargs.items()
            },
        )

    def _restore_patches(self) -> None:
        while self._patched_modules:
            patched_module, original_forward = self._patched_modules.pop()
            patched_module.forward = original_forward
        self._pin_memory_disabled = False
        self._resident_module_names.clear()

    def _move_value_to_execution_device(self, value: Any, name: str) -> Any:
        if isinstance(value, torch.Tensor):
            if value.device == self.execution_device:
                return value
            if value.device.type == "meta":
                return value
            start = time.perf_counter()
            moved = value.to(device=self.execution_device, non_blocking=True)
            self.state.add_copy(name, time.perf_counter() - start, _tensor_size_bytes(moved))
            return moved
        if isinstance(value, tuple):
            return tuple(self._move_value_to_execution_device(item, name) for item in value)
        if isinstance(value, list):
            return [self._move_value_to_execution_device(item, name) for item in value]
        if isinstance(value, dict):
            return {key: self._move_value_to_execution_device(item, f"{name}:{key}") for key, item in value.items()}
        return value

    def _prepare_linear_runtime(self, module: nn.Module) -> None:
        skip_patterns = tuple(re.compile(pattern) for pattern in self.config.skip_modules_pattern)
        resident_patterns = tuple(re.compile(pattern) for pattern in self.config.always_resident_modules_pattern)
        if self.state.resolved_resident_module_patterns:
            resident_module_patterns = tuple(
                re.compile(pattern) for pattern in self.state.resolved_resident_module_patterns
            )
        else:
            resident_module_patterns = _resolve_resident_module_patterns(module, self.config)
            self.state.resolved_resident_module_patterns = [pattern.pattern for pattern in resident_module_patterns]
        modules_to_pin: list[tuple[str, nn.Module, int]] = []

        self._apply_auto_budget_policy(module, skip_patterns, resident_patterns, resident_module_patterns)
        if self.state.planner_decisions.get("auto_pin_weight_decision") == "full_pin_zero_resident":
            # A full-pinned, zero-resident plan must not quietly retain the
            # default "always resident" leaves on CUDA.  Besides contradicting
            # the plan label, those leaves consume the activation headroom that
            # this mode is explicitly trying to reserve.  Treat them as normal
            # pinned leaves instead; the runtime still onloads them just in time.
            if resident_patterns:
                self.state.planner_decisions["auto_always_resident_decision"] = "offload_for_zero_resident"
            resident_patterns = ()
            # The automatic budget was calculated before the default resident
            # leaves joined the pin candidates. Select every one of them rather
            # than accidentally leaving that residual unpinned.
            self._force_full_pin_weights = True

        self._move_root_local_tensors_to_device(module)
        if self.resident_module_budget_bytes > 0 and resident_module_patterns:
            start = time.perf_counter()
            selected_bytes = self._select_resident_modules(module, skip_patterns, resident_module_patterns)
            self.state.add_setup("select_resident_modules", time.perf_counter() - start, selected_bytes)
            self.state.planner_decisions["selected_resident_module_gb"] = round(selected_bytes / 1024**3, 4)

        for module_name, submodule in module.named_modules():
            if module_name == "":
                continue
            if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
                continue
            if self._is_descendant_of_resident_module(module_name):
                continue
            if module_name in self._resident_module_names:
                start = time.perf_counter()
                before = self._move_resident_module_to_execution_device(module_name, submodule)
                self.state.add_setup("resident_budget_modules_to_device", time.perf_counter() - start, before)
                continue

            is_resident_module = bool(
                resident_patterns and any(pattern.search(module_name) for pattern in resident_patterns)
            )
            if is_resident_module:
                start = time.perf_counter()
                before = self._move_resident_module_to_execution_device(module_name, submodule)
                self.state.add_setup("resident_modules_to_device", time.perf_counter() - start, before)
                continue

            if isinstance(submodule, nn.Linear):
                if not self._can_patch_linear(submodule):
                    start = time.perf_counter()
                    moved_bytes = self._move_module_tensors_to_execution_device(submodule)
                    self.state.add_setup(
                        "unsupported_linear_modules_to_device",
                        time.perf_counter() - start,
                        moved_bytes,
                    )
                    unsupported_count = self.state.planner_decisions.get("unsupported_linear_module_count", 0)
                    self.state.planner_decisions["unsupported_linear_module_count"] = unsupported_count + 1
                    continue
                self._move_linear_to_runtime_devices(
                    module_name,
                    submodule,
                    modules_to_pin,
                )
                self._patch_linear(submodule)
            elif isinstance(submodule, nn.Embedding):
                self._move_embedding_to_runtime_devices(
                    module_name,
                    submodule,
                    modules_to_pin,
                )
                self._patch_embedding(submodule)
            else:
                self._move_small_local_tensors_to_device(submodule)

        if self.config.pin_cpu_memory and modules_to_pin and not self._skip_pin_weights:
            start = time.perf_counter()
            selected_linears_to_pin = self._select_linear_weights_to_pin(modules_to_pin)
            pinned_bytes = self._pin_linear_weights(selected_linears_to_pin)
            self.state.add_setup("pin_linear_weights", time.perf_counter() - start, pinned_bytes)
        elif self.config.pin_cpu_memory and modules_to_pin:
            self.state.add_setup("pin_linear_weights_skipped", 0.0, 0)

    def _collect_linear_modules_to_pin(
        self,
        module: nn.Module,
        skip_patterns: tuple[re.Pattern[str], ...],
        resident_patterns: tuple[re.Pattern[str], ...],
    ) -> list[tuple[str, nn.Module, int]]:
        candidates: list[tuple[str, nn.Module, int]] = []
        for module_name, submodule in module.named_modules():
            if module_name == "" or not isinstance(submodule, (nn.Linear, nn.Embedding)):
                continue
            if isinstance(submodule, nn.Linear) and not self._can_patch_linear(submodule):
                continue
            if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
                continue
            if self._is_descendant_of_resident_module(module_name):
                continue
            if module_name in self._resident_module_names:
                continue
            if resident_patterns and any(pattern.search(module_name) for pattern in resident_patterns):
                continue
            if submodule.weight.device.type == "cpu" and not submodule.weight.data.is_pinned():
                candidates.append((module_name, submodule, _tensor_size_bytes(submodule.weight.data)))
        return candidates

    def _move_linear_to_runtime_devices(
        self,
        module_name: str,
        linear: nn.Linear,
        modules_to_pin: list[tuple[str, nn.Module, int]] | None,
    ) -> None:
        if linear.weight.device != self.offload_device:
            start = time.perf_counter()
            weight_bytes = _tensor_size_bytes(linear.weight.data)
            linear.weight.data = linear.weight.data.to(self.offload_device)
            self.state.add_setup("linear_weights_to_offload", time.perf_counter() - start, weight_bytes)
        if linear.bias is not None and linear.bias.device != self.execution_device:
            start = time.perf_counter()
            bias_bytes = _tensor_size_bytes(linear.bias.data)
            linear.bias.data = linear.bias.data.to(self.execution_device)
            self.state.add_setup("linear_bias_to_device", time.perf_counter() - start, bias_bytes)
        if modules_to_pin is not None and linear.weight.device.type == "cpu" and not linear.weight.data.is_pinned():
            modules_to_pin.append((module_name, linear, _tensor_size_bytes(linear.weight.data)))

    def _move_embedding_to_runtime_devices(
        self,
        module_name: str,
        embedding: nn.Embedding,
        modules_to_pin: list[tuple[str, nn.Module, int]] | None,
    ) -> None:
        if embedding.weight.device != self.offload_device:
            start = time.perf_counter()
            weight_bytes = _tensor_size_bytes(embedding.weight.data)
            embedding.weight.data = embedding.weight.data.to(self.offload_device)
            self.state.add_setup("embedding_weights_to_offload", time.perf_counter() - start, weight_bytes)
        if modules_to_pin is not None and embedding.weight.device.type == "cpu" and not embedding.weight.data.is_pinned():
            modules_to_pin.append((module_name, embedding, _tensor_size_bytes(embedding.weight.data)))

    def _select_linear_weights_to_pin(self, candidates: list[tuple[str, nn.Module, int]]) -> list[tuple[str, nn.Module]]:
        candidate_bytes = sum(weight_bytes for _, _, weight_bytes in candidates)
        if self._force_full_pin_weights:
            self.state.planner_decisions["resolved_pin_weight_budget_mode"] = "snap_to_full_pin"
            self.state.planner_decisions["resolved_pin_weight_budget_gb"] = round(candidate_bytes / 1024**3, 4)
            self.state.selected_pinned_linear_weights = [module_name for module_name, _, _ in candidates]
            return [(module_name, linear) for module_name, linear, _ in candidates]
        pin_weight_budget_bytes = self.pin_weight_budget_bytes
        requested_unlimited_pin = pin_weight_budget_bytes <= 0 and self.pin_weight_budget_ratio <= 0
        if pin_weight_budget_bytes <= 0 and self.pin_weight_budget_ratio > 0:
            pin_weight_budget_bytes = int(candidate_bytes * self.pin_weight_budget_ratio)
        if pin_weight_budget_bytes > 0 and self.max_pin_weight_budget_bytes > 0:
            pin_weight_budget_bytes = min(pin_weight_budget_bytes, self.max_pin_weight_budget_bytes)
        if candidate_bytes > 0 and pin_weight_budget_bytes >= int(candidate_bytes * 0.95):
            pin_weight_budget_bytes = 0
            self.state.planner_decisions["pin_weight_budget_snap_to_full"] = True
        if pin_weight_budget_bytes <= 0:
            self.state.planner_decisions["resolved_pin_weight_budget_mode"] = (
                "unlimited_full_pin" if requested_unlimited_pin else "snap_to_full_pin"
            )
            self.state.planner_decisions["resolved_pin_weight_budget_gb"] = round(candidate_bytes / 1024**3, 4)
            self.state.selected_pinned_linear_weights = [module_name for module_name, _, _ in candidates]
            return [(module_name, linear) for module_name, linear, _ in candidates]

        self.state.planner_decisions["resolved_pin_weight_budget_mode"] = "limited"
        self.state.planner_decisions["resolved_pin_weight_budget_gb"] = round(pin_weight_budget_bytes / 1024**3, 4)
        ordered_candidates = candidates
        if self.pin_weight_selection == "spread":
            ordered_candidates = _spread_order(candidates, pin_weight_budget_bytes)
        elif self.pin_weight_selection == "largest":
            ordered_candidates = _largest_first_order(candidates)

        selected_linears: list[tuple[str, nn.Module]] = []
        selected_bytes = 0
        selected_names: list[str] = []
        for module_name, linear, weight_bytes in ordered_candidates:
            if selected_bytes + weight_bytes > pin_weight_budget_bytes:
                continue
            selected_linears.append((module_name, linear))
            selected_names.append(module_name)
            selected_bytes += weight_bytes

        self.state.selected_pinned_linear_weights = selected_names
        return selected_linears

    def _apply_auto_budget_policy(
        self,
        module: nn.Module,
        skip_patterns: tuple[re.Pattern[str], ...],
        resident_patterns: tuple[re.Pattern[str], ...],
        resident_module_patterns: tuple[re.Pattern[str], ...],
    ) -> None:
        if self.auto_budget_policy == _AUTO_BUDGET_DISABLED:
            return

        candidate_weight_bytes = sum(
            weight_bytes
            for _, _, weight_bytes in self._collect_linear_modules_to_pin(module, skip_patterns, resident_patterns)
        )
        candidate_module_bytes = 0
        if resident_module_patterns:
            candidate_module_bytes = sum(
                module_bytes
                for module_name, submodule in module.named_modules()
                if module_name
                and not (skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns))
                and any(pattern.search(module_name) for pattern in resident_module_patterns)
                for module_bytes in (_module_size_bytes(submodule),)
            )

        self.state.planner_decisions.update(
            {
                "auto_budget_policy": self.auto_budget_policy,
                "candidate_weight_gb": round(candidate_weight_bytes / 1024**3, 4),
                "candidate_resident_module_gb": round(candidate_module_bytes / 1024**3, 4),
                "available_system_ram_gb": round(self.available_system_ram_bytes / 1024**3, 4),
                "system_ram_headroom_gb": round(self.system_ram_headroom_bytes / 1024**3, 4),
            }
        )

        total_vram_gb = get_cuda_total_vram_gb(self.execution_device)
        total_vram_bytes = int(total_vram_gb * 1024**3)
        model_to_vram_ratio = candidate_module_bytes / total_vram_bytes if total_vram_bytes > 0 else 0.0
        if total_vram_gb > 0:
            self.state.planner_decisions["cuda_total_vram_gb"] = round(total_vram_gb, 4)
            self.state.planner_decisions["auto_model_to_vram_ratio"] = round(model_to_vram_ratio, 4)

        usable_ram_bytes = max(0, self.available_system_ram_bytes - self.system_ram_headroom_bytes)
        requested_full_pin_resident_bytes = int(
            max(0.0, float(self.config.auto_full_pin_resident_budget_gb)) * 1024**3
        )
        can_auto_full_pin = (
            self.config.pin_cpu_memory
            and self.pin_weight_budget_bytes <= 0
            and self.pin_weight_budget_ratio <= 0
            and self.resident_module_budget_bytes <= 0
            and self.max_pin_weight_budget_bytes <= 0
            and self.available_system_ram_bytes > 0
            and candidate_weight_bytes > 0
            and usable_ram_bytes >= candidate_weight_bytes
            and self.config.auto_full_pin_min_model_to_vram_ratio > 0
            and model_to_vram_ratio >= self.config.auto_full_pin_min_model_to_vram_ratio
        )
        full_pin_resident_budget_bytes = min(requested_full_pin_resident_bytes, candidate_module_bytes)
        can_auto_full_pin_resident = (
            can_auto_full_pin
            and full_pin_resident_budget_bytes > 0
            and usable_ram_bytes >= candidate_weight_bytes + full_pin_resident_budget_bytes
        )
        self.state.planner_decisions["auto_usable_system_ram_gb"] = round(usable_ram_bytes / 1024**3, 4)
        self.state.planner_decisions["auto_full_pin_min_model_to_vram_ratio"] = (
            self.config.auto_full_pin_min_model_to_vram_ratio
        )
        self.state.planner_decisions["auto_full_pin_resident_budget_requested_gb"] = round(
            requested_full_pin_resident_bytes / 1024**3, 4
        )

        if self.resident_module_budget_bytes <= 0 and candidate_module_bytes > 0 and not can_auto_full_pin:
            budget = int(candidate_module_bytes * 0.25)
            if total_vram_gb > 0:
                vram_headroom_gb = self.config.auto_vram_headroom_gb
                if vram_headroom_gb <= 0:
                    if total_vram_gb <= 8.5:
                        vram_headroom_gb = 3.0 if model_to_vram_ratio <= 2.5 else 2.0
                    elif total_vram_gb <= 12.5:
                        vram_headroom_gb = 2.5
                    else:
                        vram_headroom_gb = max(3.0, total_vram_gb * 0.15)
                    self.state.planner_decisions["auto_model_to_vram_ratio"] = round(model_to_vram_ratio, 4)
                vram_target_gb = max(1.0, total_vram_gb - vram_headroom_gb)
                vram_target_bytes = int(vram_target_gb * 1024**3)
                candidate_exceeds_vram = candidate_module_bytes > int(total_vram_gb * 1024**3)
                if candidate_exceeds_vram:
                    budget = max(budget, vram_target_bytes)
                    budget = min(budget, candidate_module_bytes)
                    self.state.planner_decisions["auto_resident_module_budget_vram_target_gb"] = round(
                        vram_target_gb, 4
                    )
                    self.state.planner_decisions["auto_vram_headroom_gb"] = round(vram_headroom_gb, 4)
                if budget > vram_target_bytes:
                    budget = vram_target_bytes
                    self.state.planner_decisions["auto_resident_module_budget_vram_cap_gb"] = round(vram_target_gb, 4)
                    self.state.planner_decisions["auto_vram_headroom_gb"] = round(vram_headroom_gb, 4)
                    self.state.planner_decisions["auto_resident_module_budget_vram_cap_reason"] = (
                        "cuda_vram_headroom"
                    )
            if self.max_resident_module_budget_bytes > 0:
                budget = min(budget, self.max_resident_module_budget_bytes)
            self.resident_module_budget_bytes = budget
            self.state.planner_decisions["auto_resident_module_budget_gb"] = round(budget / 1024**3, 4)
        elif can_auto_full_pin:
            if can_auto_full_pin_resident:
                self.resident_module_budget_bytes = full_pin_resident_budget_bytes
                self.state.planner_decisions["auto_resident_module_budget_gb"] = round(
                    full_pin_resident_budget_bytes / 1024**3, 4
                )
                self.state.planner_decisions["auto_resident_module_decision"] = "full_pin_residual_vram"
            else:
                self.state.planner_decisions["auto_resident_module_budget_gb"] = 0.0
                self.state.planner_decisions["auto_resident_module_decision"] = "zero_for_full_pin"
                if requested_full_pin_resident_bytes > 0:
                    self.state.planner_decisions["auto_full_pin_resident_skip_reason"] = "insufficient_ram"

        if self.pin_weight_budget_bytes <= 0 and self.pin_weight_budget_ratio <= 0 and candidate_weight_bytes > 0:
            if can_auto_full_pin:
                ratio = 1.0
                budget = candidate_weight_bytes
                self.state.planner_decisions["auto_pin_weight_decision"] = (
                    "full_pin_residual_vram" if can_auto_full_pin_resident else "full_pin_zero_resident"
                )
            elif candidate_weight_bytes <= 8 * 1024**3:
                ratio = 1.0
                budget = int(candidate_weight_bytes * ratio)
            elif candidate_weight_bytes <= 16 * 1024**3:
                ratio = 0.85
                budget = int(candidate_weight_bytes * ratio)
            else:
                ratio = 0.75
                budget = int(candidate_weight_bytes * ratio)
            if self.max_pin_weight_budget_bytes > 0:
                budget = min(budget, self.max_pin_weight_budget_bytes)
            if self.available_system_ram_bytes > 0:
                required_ram_bytes = candidate_weight_bytes + self.resident_module_budget_bytes
                self.state.planner_decisions["auto_required_system_ram_gb"] = round(required_ram_bytes / 1024**3, 4)
                if usable_ram_bytes < required_ram_bytes:
                    budget = 0
                    ratio = 0.0
                    self._skip_pin_weights = True
                    self.state.planner_decisions["auto_pin_weight_decision"] = "skip_insufficient_ram"
                elif not can_auto_full_pin:
                    self.state.planner_decisions["auto_pin_weight_decision"] = "budgeted"
            elif not can_auto_full_pin:
                self.state.planner_decisions["auto_pin_weight_decision"] = "budgeted"
            self.pin_weight_budget_bytes = budget
            self.state.planner_decisions["auto_pin_weight_budget_gb"] = round(budget / 1024**3, 4)
            self.state.planner_decisions["auto_pin_weight_budget_ratio"] = ratio

    def _select_resident_modules(
        self,
        module: nn.Module,
        skip_patterns: tuple[re.Pattern[str], ...],
        module_patterns: tuple[re.Pattern[str], ...],
    ) -> int:
        candidates: list[tuple[str, nn.Module, int]] = []
        for module_name, submodule in module.named_modules():
            if module_name == "":
                continue
            if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
                continue
            if not any(pattern.search(module_name) for pattern in module_patterns):
                continue
            if any(_is_module_descendant(module_name, selected_name) for selected_name in self._resident_module_names):
                continue
            candidates.append((module_name, submodule, _module_size_bytes(submodule)))

        ordered_candidates = candidates
        if self.resident_module_selection == "spread":
            ordered_candidates = _spread_order(candidates, self.resident_module_budget_bytes)
        elif self.resident_module_selection == "largest":
            ordered_candidates = _largest_first_order(candidates)

        selected_bytes = 0
        for module_name, _, module_bytes in ordered_candidates:
            if selected_bytes + module_bytes > self.resident_module_budget_bytes:
                continue
            self._resident_module_names.add(module_name)
            self.state.selected_resident_modules.append(module_name)
            selected_bytes += module_bytes
        return selected_bytes

    def _is_descendant_of_resident_module(self, module_name: str) -> bool:
        return any(_is_module_descendant(module_name, resident_name) for resident_name in self._resident_module_names)

    def _move_resident_module_to_execution_device(self, _module_name: str, module: nn.Module) -> int:
        return self._move_module_tensors_to_execution_device(module)

    def _move_module_tensors_to_execution_device(self, module: nn.Module) -> int:
        moved_bytes = 0
        for parameter in module.parameters(recurse=True):
            tensor_bytes = _tensor_size_bytes(parameter.data)
            moved_bytes += tensor_bytes
            if parameter.device != self.execution_device:
                parameter.data = parameter.data.to(self.execution_device)
        for buffer in module.buffers(recurse=True):
            tensor_bytes = _tensor_size_bytes(buffer.data)
            moved_bytes += tensor_bytes
            if buffer.device != self.execution_device:
                buffer.data = buffer.data.to(self.execution_device)
        return moved_bytes

    def _move_root_local_tensors_to_device(self, module: nn.Module) -> None:
        start = time.perf_counter()
        moved_bytes = 0
        for parameter in module.parameters(recurse=False):
            if parameter.device == self.execution_device:
                continue
            tensor_bytes = _tensor_size_bytes(parameter.data)
            parameter.data = parameter.data.to(self.execution_device)
            moved_bytes += tensor_bytes
        for buffer in module.buffers(recurse=False):
            if buffer.device == self.execution_device:
                continue
            tensor_bytes = _tensor_size_bytes(buffer.data)
            buffer.data = buffer.data.to(self.execution_device)
            moved_bytes += tensor_bytes
        if moved_bytes:
            self.state.add_setup("root_tensors_to_device", time.perf_counter() - start, moved_bytes)

    def _move_small_local_tensors_to_device(self, module: nn.Module) -> int:
        moved_bytes = 0
        for parameter in module.parameters(recurse=False):
            if parameter.device == self.execution_device:
                continue
            tensor_bytes = _tensor_size_bytes(parameter.data)
            if tensor_bytes > self.config.small_tensor_threshold_bytes:
                continue
            start = time.perf_counter()
            parameter.data = parameter.data.to(self.execution_device)
            self.state.add_setup("small_parameters_to_device", time.perf_counter() - start, tensor_bytes)
            moved_bytes += tensor_bytes
        for buffer in module.buffers(recurse=False):
            if buffer.device == self.execution_device:
                continue
            tensor_bytes = _tensor_size_bytes(buffer.data)
            if tensor_bytes > self.config.small_tensor_threshold_bytes:
                continue
            start = time.perf_counter()
            buffer.data = buffer.data.to(self.execution_device)
            self.state.add_setup("small_buffers_to_device", time.perf_counter() - start, tensor_bytes)
            moved_bytes += tensor_bytes
        return moved_bytes

    def _pin_linear_weights(self, linears: list[tuple[str, nn.Module]]) -> int:
        pinned_bytes = 0
        assignment_lock = threading.Lock()

        def pin_linear(module_name: str, linear: nn.Module) -> tuple[int, bool]:
            weight = linear.weight.data
            cached = self._get_cached_pinned_weight(module_name, weight)
            if cached is not None:
                pinned = cached
                cache_hit = True
            else:
                pinned = weight.pin_memory()
                cache_hit = False
            pinned = self._store_cached_pinned_weight(module_name, pinned)
            tensor_bytes = _tensor_size_bytes(pinned)
            with assignment_lock:
                linear.weight.data = pinned
            return tensor_bytes, cache_hit

        def record_pinned(result: tuple[int, bool]) -> None:
            nonlocal pinned_bytes
            tensor_bytes, cache_hit = result
            pinned_bytes += tensor_bytes
            if cache_hit:
                self.state.add_setup("pinned_weight_cache_hit", 0.0, tensor_bytes)

        if self.pin_cpu_workers == 1:
            for module_name, linear in linears:
                try:
                    record_pinned(pin_linear(module_name, linear))
                except _PIN_MEMORY_ERRORS as exc:
                    if not self.config.allow_pin_memory_fallback:
                        raise
                    self._disable_pin_memory("pin_linear_weights_failed", exc)
                    break
            return pinned_bytes

        linears_iter = iter(linears)
        with ThreadPoolExecutor(max_workers=self.pin_cpu_workers) as executor:
            futures = set()

            def submit_next() -> bool:
                try:
                    module_name, linear = next(linears_iter)
                except StopIteration:
                    return False
                futures.add(executor.submit(pin_linear, module_name, linear))
                return True

            for _ in range(self.pin_cpu_workers):
                if not submit_next():
                    break

            while futures:
                for future in as_completed(futures):
                    futures.remove(future)
                    try:
                        record_pinned(future.result())
                    except _PIN_MEMORY_ERRORS as exc:
                        if not self.config.allow_pin_memory_fallback:
                            raise
                        self._disable_pin_memory("pin_linear_weights_failed", exc)
                        for pending in futures:
                            pending.cancel()
                        return pinned_bytes
                    if not self._pin_memory_disabled:
                        submit_next()
                    break
        return pinned_bytes

    def _get_cached_pinned_weight(self, module_name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        if not self.config.cache_pinned_weights:
            return None
        cache_key = self._pinned_weight_cache_key(module_name, tensor)
        with _DYNAMIC_OFFLOAD_PINNED_TENSOR_CACHE_LOCK:
            cached = _DYNAMIC_OFFLOAD_PINNED_TENSOR_CACHE.get(cache_key)
        if cached is None:
            return None
        if cached.shape != tensor.shape or cached.dtype != tensor.dtype:
            return None
        return cached

    def _store_cached_pinned_weight(self, module_name: str, tensor: torch.Tensor) -> torch.Tensor:
        if not self.config.cache_pinned_weights or not tensor.is_pinned():
            return tensor
        cache_key = self._pinned_weight_cache_key(module_name, tensor)
        with _DYNAMIC_OFFLOAD_PINNED_TENSOR_CACHE_LOCK:
            cached = _DYNAMIC_OFFLOAD_PINNED_TENSOR_CACHE.setdefault(cache_key, tensor)
        return cached

    def _pinned_weight_cache_key(self, module_name: str, tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            "v1",
            self._pinned_weight_cache_namespace,
            module_name,
            tuple(tensor.shape),
            str(tensor.dtype),
            tensor.numel(),
            tensor.element_size(),
        )

    def _disable_pin_memory(self, action: str, exc: BaseException) -> None:
        if self._pin_memory_disabled:
            return
        self._pin_memory_disabled = True
        self.state.add_setup(action, 0.0, 0)
        if self.config.verbose:
            print(
                f"  [dynamic-offload] {action}: disabling pinned CPU memory fallback after {type(exc).__name__}: {exc}",
                flush=True,
            )

    @staticmethod
    def _can_patch_linear(linear: nn.Linear) -> bool:
        weight = getattr(linear, "weight", None)
        in_features = getattr(linear, "in_features", None)
        out_features = getattr(linear, "out_features", None)
        return (
            isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and in_features is not None
            and out_features is not None
            and tuple(weight.shape) == (out_features, in_features)
        )

    def _patch_linear(self, linear: nn.Linear) -> None:
        original_forward = linear.forward
        self._patched_modules.append((linear, original_forward))
        self.state.patched_module_count += 1

        def dynamic_linear_forward(patched_linear, input):
            weight = self._to_input_device(patched_linear.weight, input, "linear_weight")
            bias = self._to_input_device(patched_linear.bias, input, "linear_bias")
            if self.linear_runtime_strategy == "swap":
                source_weight = patched_linear.weight.data
                source_bias = None if patched_linear.bias is None else patched_linear.bias.data
                patched_linear.weight.data = weight
                if patched_linear.bias is not None:
                    patched_linear.bias.data = bias
                try:
                    return original_forward(input)
                finally:
                    patched_linear.weight.data = source_weight
                    if patched_linear.bias is not None:
                        patched_linear.bias.data = source_bias
            return F.linear(input, weight, bias)

        linear.forward = dynamic_linear_forward.__get__(linear, linear.__class__)

    def _patch_embedding(self, embedding: nn.Embedding) -> None:
        original_forward = embedding.forward
        self._patched_modules.append((embedding, original_forward))
        self.state.patched_module_count += 1

        def dynamic_embedding_forward(patched_embedding, input):
            weight = self._to_input_device(patched_embedding.weight, input, "embedding_weight", cast_to_input_dtype=False)
            if self.linear_runtime_strategy == "swap":
                source_weight = patched_embedding.weight.data
                patched_embedding.weight.data = weight
                try:
                    return original_forward(input)
                finally:
                    patched_embedding.weight.data = source_weight
            return F.embedding(
                input,
                weight,
                patched_embedding.padding_idx,
                patched_embedding.max_norm,
                patched_embedding.norm_type,
                patched_embedding.scale_grad_by_freq,
                patched_embedding.sparse,
            )

        embedding.forward = dynamic_embedding_forward.__get__(embedding, embedding.__class__)

    def _to_input_device(
        self,
        tensor: torch.Tensor | None,
        input: torch.Tensor,
        name: str,
        *,
        cast_to_input_dtype: bool = True,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.device == input.device and (not cast_to_input_dtype or tensor.dtype == input.dtype):
            return tensor
        start = time.perf_counter()
        dtype = input.dtype if cast_to_input_dtype else tensor.dtype
        moved = tensor.to(device=input.device, dtype=dtype, non_blocking=True)
        self.state.add_copy(name, time.perf_counter() - start, _tensor_size_bytes(moved))
        return moved

    def print_profile_summary(self, *, top_n: int = 8) -> None:
        summary = self.state.as_dict()
        setup_runtime = summary["setup_runtime"]
        setup_seconds = sum(item["seconds"] for item in setup_runtime.values())
        setup_gb = sum(item["gb"] for item in setup_runtime.values())
        copy_runtime = summary["copy_runtime"]
        copy_seconds = sum(item["seconds"] for item in copy_runtime.values())
        copy_gb = sum(item["gb"] for item in copy_runtime.values())
        print(
            "  [dynamic-offload-profile] "
            f"summary: mode={self.execution_mode} modules={summary['module_count']} "
            f"patched={summary['patched_module_count']} setup_seconds={setup_seconds:.4f} "
            f"setup_gb={setup_gb:.4f} copy_seconds={copy_seconds:.4f} "
            f"copy_gb={copy_gb:.4f}",
            flush=True,
        )

        selected_modules = summary["selected_resident_modules"]
        resolved_patterns = summary["resolved_resident_module_patterns"]
        if resolved_patterns:
            print(
                "  [dynamic-offload-profile] "
                f"resolved_resident_module_patterns={resolved_patterns}",
                flush=True,
            )

        planner_decisions = summary["planner_decisions"]
        if planner_decisions:
            print(
                "  [dynamic-offload-profile] "
                f"planner_decisions={planner_decisions}",
                flush=True,
            )

        if selected_modules:
            print(
                "  [dynamic-offload-profile] "
                f"resident_modules={selected_modules}",
                flush=True,
            )

        selected_linear_weights = summary["selected_resident_linear_weights"]
        if selected_linear_weights:
            print(
                "  [dynamic-offload-profile] "
                f"resident_linear_weights={selected_linear_weights[:top_n]} "
                f"count={len(selected_linear_weights)}",
                flush=True,
            )

        selected_pinned_linear_weights = summary["selected_pinned_linear_weights"]
        if selected_pinned_linear_weights:
            print(
                "  [dynamic-offload-profile] "
                f"pinned_linear_weights={selected_pinned_linear_weights[:top_n]} "
                f"count={len(selected_pinned_linear_weights)}",
                flush=True,
            )

        if setup_runtime:
            print("  [dynamic-offload-profile] setup_runtime_by_type:", flush=True)
            for key, item in sorted(setup_runtime.items(), key=lambda pair: pair[1]["seconds"], reverse=True)[:top_n]:
                print(
                    f"    {key}: seconds={item['seconds']:.4f} gb={item['gb']:.4f}",
                    flush=True,
                )

        if copy_runtime:
            print("  [dynamic-offload-profile] copy_runtime_by_type:", flush=True)
            for name, item in sorted(copy_runtime.items(), key=lambda pair: pair[1]["seconds"], reverse=True)[:top_n]:
                print(
                    f"    {name}: calls={item['calls']} seconds={item['seconds']:.4f} gb={item['gb']:.4f}",
                    flush=True,
                )


def load_dynamic_offload_settings_from_env(
    *,
    execution_device: str | torch.device = "cuda:0",
    offload_device: str | torch.device = "cpu",
    target_module_classes: tuple[type[nn.Module], ...] = _DEFAULT_TARGET_MODULE_CLASSES,
    skip_modules_pattern: tuple[str, ...] = (),
    always_resident_modules_pattern: tuple[str, ...] | None = None,
    running_on_wsl: bool | None = None,
    environ: Mapping[str, str] | None = None,
    default_preset: str = "",
    component_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> DynamicOffloadSettings:
    env = os.environ if environ is None else environ
    is_wsl = is_wsl_environment() if running_on_wsl is None else running_on_wsl
    requested_preset = dynamic_offload_env_value("DDO_PRESET", default_preset, env).strip().lower()
    effective_preset = resolve_dynamic_offload_preset(requested_preset, running_on_wsl=is_wsl)

    def preset_env(name: str, default: str = "") -> str:
        return dynamic_offload_preset_env_value(
            name,
            default,
            requested_preset=requested_preset,
            effective_preset=effective_preset,
            running_on_wsl=is_wsl,
            environ=env,
        )

    def preset_bool(name: str, default: str = "0") -> bool:
        return _parse_bool_value(preset_env(name, default))

    default_always_resident_patterns = _DEFAULT_ALWAYS_RESIDENT_MODULE_PATTERNS
    if always_resident_modules_pattern is not None:
        default_always_resident_patterns = always_resident_modules_pattern

    available_system_ram_override = preset_env("DDO_AVAILABLE_SYSTEM_RAM_GB", "").strip()
    available_system_ram_gb = (
        float(available_system_ram_override)
        if available_system_ram_override
        else round(get_available_system_ram_gb(), 4)
    )

    plan = preset_bool("DDO_PLAN")
    execution_mode = preset_env("DDO_EXECUTION_MODE", "plan").lower()
    requested_pin_cpu_memory = preset_bool("DDO_PIN_CPU_MEMORY")
    allow_pin_memory_fallback = preset_bool("DDO_ALLOW_PIN_MEMORY_FALLBACK", "1")
    disable_pin_on_wsl = preset_bool("DDO_DISABLE_PIN_ON_WSL", "1")
    effective_pin_cpu_memory = requested_pin_cpu_memory and not (is_wsl and disable_pin_on_wsl)
    load_safetensors_backend = preset_env("DDO_LOAD_SAFETENSORS_BACKEND", "").strip().lower()
    if not load_safetensors_backend:
        load_safetensors_backend = preset_env("DDO_SAFETENSORS_BACKEND", "").strip().lower()

    config = DynamicOffloadConfig(
        execution_device=execution_device,
        offload_device=offload_device,
        target_module_classes=target_module_classes,
        skip_modules_pattern=skip_modules_pattern,
        always_resident_modules_pattern=_parse_pattern_list_value(
            preset_env(
                "DDO_ALWAYS_RESIDENT_MODULE_PATTERNS",
                ",".join(default_always_resident_patterns),
            )
        ),
        small_tensor_threshold_bytes=int(preset_env("DDO_SMALL_TENSOR_THRESHOLD_KB", "1024")) * 1024,
        execution_mode=execution_mode,
        linear_runtime_strategy=preset_env("DDO_LINEAR_RUNTIME_STRATEGY", "functional").lower(),
        pin_cpu_memory=effective_pin_cpu_memory,
        allow_pin_memory_fallback=allow_pin_memory_fallback,
        pin_cpu_workers=int(preset_env("DDO_PIN_CPU_WORKERS", "4")),
        cache_plan=preset_bool("DDO_CACHE_PLAN", "1"),
        cache_pinned_weights=preset_bool("DDO_CACHE_PINNED_WEIGHTS", "0"),
        pinned_weight_cache_namespace=preset_env("DDO_PINNED_WEIGHT_CACHE_NAMESPACE", ""),
        auto_budget_policy=preset_env("DDO_AUTO_BUDGET_POLICY", "off").lower(),
        max_resident_module_budget_gb=float(
            preset_env("DDO_MAX_RESIDENT_MODULE_BUDGET_GB", "6.0")
        ),
        auto_vram_headroom_gb=float(preset_env("DDO_AUTO_VRAM_HEADROOM_GB", "0.0")),
        auto_full_pin_min_model_to_vram_ratio=float(
            preset_env("DDO_AUTO_FULL_PIN_MIN_MODEL_TO_VRAM_RATIO", "4.0")
        ),
        auto_full_pin_resident_budget_gb=float(
            preset_env("DDO_AUTO_FULL_PIN_RESIDENT_BUDGET_GB", "0.0")
        ),
        max_pin_weight_budget_gb=float(preset_env("DDO_MAX_PIN_WEIGHT_BUDGET_GB", "0.0")),
        available_system_ram_gb=available_system_ram_gb,
        system_ram_headroom_gb=float(preset_env("DDO_SYSTEM_RAM_HEADROOM_GB", "6.0")),
        pin_weight_budget_gb=float(preset_env("DDO_PIN_WEIGHT_BUDGET_GB", "0.0")),
        pin_weight_budget_ratio=float(preset_env("DDO_PIN_WEIGHT_BUDGET_RATIO", "0.0")),
        pin_weight_selection=preset_env("DDO_PIN_WEIGHT_SELECTION", "spread").lower(),
        resident_module_budget_gb=float(preset_env("DDO_RESIDENT_MODULE_BUDGET_GB", "0.0")),
        resident_module_patterns=_parse_pattern_list_value(preset_env("DDO_RESIDENT_MODULE_PATTERNS", "")),
        resident_module_selection=preset_env("DDO_RESIDENT_MODULE_SELECTION", "spread").lower(),
        load_safetensors_backend=load_safetensors_backend,
        verbose=preset_bool("DDO_VERBOSE", "0"),
        show_profile=preset_bool("DDO_SHOW_PROFILE", "0"),
    )
    return DynamicOffloadSettings(
        requested_preset=requested_preset,
        effective_preset=effective_preset,
        enabled=plan or execution_mode != "plan",
        plan=plan,
        config=config,
        requested_pin_cpu_memory=requested_pin_cpu_memory,
        effective_pin_cpu_memory=effective_pin_cpu_memory,
        disable_pin_on_wsl=disable_pin_on_wsl,
        running_on_wsl=is_wsl,
        component_policies=component_policies or {},
    )


def apply_dynamic_offload(module: nn.Module, config: DynamicOffloadConfig | None = None) -> DynamicOffloadHook:
    config = config or DynamicOffloadConfig()
    registry = HookRegistry.check_if_exists_or_initialize(module)
    existing_hook = registry.get_hook(_DYNAMIC_OFFLOAD_HOOK)
    if existing_hook is not None:
        registry.remove_hook(_DYNAMIC_OFFLOAD_HOOK)
    hook = DynamicOffloadHook(config)
    registry.register_hook(hook, _DYNAMIC_OFFLOAD_HOOK)
    return hook


def _component_prefix(component: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", component.upper()).strip("_") or "MODEL"


def _component_policy(settings: DynamicOffloadSettings, component: str) -> Mapping[str, Any]:
    prefix = _component_prefix(component)
    candidates = (
        component,
        component.lower(),
        prefix,
        prefix.lower(),
    )
    for candidate in candidates:
        policy = settings.component_policies.get(candidate)
        if policy is not None:
            return policy
    return {}


def _policy_value(policy: Mapping[str, Any], name: str) -> Any:
    key = name.lower()
    candidates = (
        key,
        name,
        name.upper(),
        key.replace("_", "-"),
    )
    for candidate in candidates:
        if candidate in policy:
            return policy[candidate]
    return None


def _component_preset_value(
    settings: DynamicOffloadSettings,
    component: str,
    name: str,
    default: str = "",
    environ: Mapping[str, str] | None = None,
) -> str:
    policy_item = _policy_value(_component_policy(settings, component), name)
    if policy_item is not None:
        return str(policy_item)

    prefix = _component_prefix(component)
    for candidate in (f"DDO_{prefix}_{name}", f"DDO_RUNNER_{prefix}_{name}"):
        value = settings.preset_value(candidate, "", environ)
        if value != "":
            return value
    return default


def _component_preset_bool(
    settings: DynamicOffloadSettings,
    component: str,
    name: str,
    default: str = "0",
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _parse_bool_value(_component_preset_value(settings, component, name, default, environ))


def enable_diffusers_group_offload(
    module: nn.Module,
    *,
    onload_device: str | torch.device = "cuda:0",
    offload_device: str | torch.device = "cpu",
    offload_type: str = "leaf_level",
    use_stream: bool = True,
    record_stream: bool = False,
    low_cpu_mem_usage: bool = True,
    num_blocks_per_group: int = 1,
    apply_hook: bool = True,
    record_event: Any | None = None,
    event_name: str = "setup_diffusers_group_offload",
) -> DiffusersGroupOffloadResult:
    """Apply the official Diffusers group offload helper with DDO-style reporting."""
    start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "onload_device": torch.device(onload_device),
        "offload_device": torch.device(offload_device),
        "offload_type": offload_type,
        "use_stream": use_stream,
        "record_stream": record_stream,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if offload_type == "block_level":
        kwargs["num_blocks_per_group"] = num_blocks_per_group

    if apply_hook:
        from diffusers.hooks import apply_group_offloading

        apply_group_offloading(module, **kwargs)

    event_payload = {
        "offload_type": offload_type,
        "use_stream": use_stream,
        "record_stream": record_stream,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if offload_type == "block_level":
        event_payload["num_blocks_per_group"] = num_blocks_per_group

    if record_event is not None:
        record_event(event_name, time.perf_counter() - start, **event_payload)

    return DiffusersGroupOffloadResult(module=module, event_payload=event_payload)


def enable_offload(
    module: nn.Module,
    *,
    settings: DynamicOffloadSettings | None = None,
    config: DynamicOffloadConfig | None = None,
    component: str = "model",
    preset: str = "auto",
    route: str | None = None,
    group_offload: bool | None = None,
    dynamic_offload: bool | None = None,
    offload_type: str | None = None,
    use_stream: bool | None = None,
    record_stream: bool | None = None,
    group_low_cpu_mem_usage: bool | None = None,
    num_blocks_per_group: int | None = None,
    execution_device: str | torch.device = "cuda:0",
    offload_device: str | torch.device = "cpu",
    low_cpu_mem_usage: bool = True,
    running_on_wsl: bool | None = None,
    environ: Mapping[str, str] | None = None,
    use_environment: bool = True,
    apply_hook: bool = True,
    record_event: Any | None = None,
    dynamic_event_name: str | None = None,
    group_event_name: str | None = None,
    **config_overrides: Any,
) -> DynamicOffloadApplyResult:
    """Enable the offload route requested by DDO settings for one component."""
    if settings is None:
        settings = load_dynamic_offload_settings_from_env(
            execution_device=execution_device,
            offload_device=offload_device,
            running_on_wsl=running_on_wsl,
            environ=environ if use_environment else {},
            default_preset=preset,
        )

    route_value = route or _component_preset_value(settings, component, "ROUTE", "auto", environ)
    route_value = route_value.strip().lower()
    if route_value in {"diffusers", "diffusers_group", "group", "group_offload"}:
        route_value = "diffusers_group_offload"
    elif route_value in {"dynamic", "ddo"}:
        route_value = "dynamic_offload"
    elif route_value in {"cuda", "standard", "none", "off"}:
        route_value = "none"

    group_enabled = (
        group_offload
        if group_offload is not None
        else _component_preset_bool(settings, component, "GROUP_OFFLOAD", "0", environ)
    )
    dynamic_setting = _component_preset_value(settings, component, "DYNAMIC_OFFLOAD", "", environ)
    dynamic_enabled = (
        dynamic_offload
        if dynamic_offload is not None
        else (_parse_bool_value(dynamic_setting) if dynamic_setting != "" else settings.enabled)
    )
    quantized_backend_payload = detect_quantized_backend_modules(module)

    if route_value == "auto":
        if group_enabled or (quantized_backend_payload and dynamic_enabled):
            route_value = "diffusers_group_offload"
        elif dynamic_enabled:
            route_value = "dynamic_offload"
        else:
            route_value = "none"

    if route_value == "diffusers_group_offload":
        group_use_stream = (
            use_stream
            if use_stream is not None
            else _component_preset_bool(settings, component, "OFFLOAD_STREAM", "1", environ)
        )
        group_record_stream = (
            record_stream
            if record_stream is not None
            else _component_preset_bool(settings, component, "OFFLOAD_RECORD_STREAM", "0", environ)
        )
        if settings.running_on_wsl and settings.disable_pin_on_wsl and group_use_stream:
            group_use_stream = False
            group_record_stream = False

        resolved_offload_type = offload_type or _component_preset_value(settings, component, "OFFLOAD_TYPE", "leaf_level", environ)
        resolved_low_cpu_mem_usage = (
            group_low_cpu_mem_usage
            if group_low_cpu_mem_usage is not None
            else _component_preset_bool(
                settings,
                component,
                "OFFLOAD_LOW_CPU_MEM_USAGE",
                "1" if low_cpu_mem_usage else "0",
                environ,
            )
        )
        resolved_num_blocks = (
            num_blocks_per_group
            if num_blocks_per_group is not None
            else int(_component_preset_value(settings, component, "NUM_BLOCKS_PER_GROUP", "1", environ))
        )
        result = enable_diffusers_group_offload(
            module,
            onload_device=execution_device,
            offload_device=offload_device,
            offload_type=resolved_offload_type,
            use_stream=group_use_stream,
            record_stream=group_record_stream,
            low_cpu_mem_usage=resolved_low_cpu_mem_usage,
            num_blocks_per_group=resolved_num_blocks,
            apply_hook=apply_hook,
            record_event=record_event,
            event_name=group_event_name or f"setup_{component}_group_offload",
        )
        if quantized_backend_payload:
            result.event_payload["quantized_backend"] = quantized_backend_payload
        return DynamicOffloadApplyResult(
            module=result.module,
            hook=None,
            settings=settings,
            should_move_to_execution_device=False,
            event_payload=result.event_payload,
            route="diffusers_group_offload",
        )

    if route_value == "dynamic_offload":
        result = enable_dynamic_offload(
            module,
            settings=settings,
            config=config,
            execution_device=execution_device,
            offload_device=offload_device,
            apply_hook=apply_hook,
            record_event=record_event,
            event_name=dynamic_event_name or f"setup_{component}_dynamic_offload",
            **config_overrides,
        )
        result.route = "dynamic_offload"
        return result

    if route_value != "none":
        raise ValueError(
            f"Invalid offload route {route_value!r}. "
            "Valid values: auto, dynamic_offload, diffusers_group_offload, none."
        )

    event_payload: dict[str, Any] = {}
    should_move = True
    if apply_hook:
        move_start = time.perf_counter()
        module.to(execution_device)
        move_seconds = time.perf_counter() - move_start
        should_move = False
        event_payload["unmanaged_module_moved_to_execution_device"] = str(execution_device)
        if record_event is not None:
            record_event(
                f"move_{component}_to_execution_device",
                move_seconds,
                route="none",
                execution_device=str(execution_device),
            )

    return DynamicOffloadApplyResult(
        module=module,
        hook=None,
        settings=settings,
        should_move_to_execution_device=should_move,
        event_payload=event_payload,
        route="none",
    )


def enable_pipeline_offload(
    pipeline: Any,
    *,
    settings: DynamicOffloadSettings | None = None,
    components: Mapping[str, nn.Module] | tuple[str, ...] | list[str] | None = None,
    preset: str = "auto",
    execution_device: str | torch.device = "cuda:0",
    offload_device: str | torch.device = "cpu",
    low_cpu_mem_usage: bool = True,
    running_on_wsl: bool | None = None,
    environ: Mapping[str, str] | None = None,
    use_environment: bool = True,
    apply_hook: bool = True,
    record_event: Any | None = None,
    **config_overrides: Any,
) -> dict[str, DynamicOffloadApplyResult]:
    """Apply DDO routing to modules exposed by a Diffusers-style pipeline.

    Each component is resolved through `enable_offload(...)`, so global presets,
    component policies, environment overrides, and quantized-backend compatibility
    all follow the same path. If a component resolves to the unmanaged `none`
    route, this high-level helper moves it to the execution device automatically.
    """
    if settings is None:
        settings = load_dynamic_offload_settings_from_env(
            execution_device=execution_device,
            offload_device=offload_device,
            running_on_wsl=running_on_wsl,
            environ=environ if use_environment else {},
            default_preset=preset,
        )

    if isinstance(components, Mapping):
        selected_components = components.items()
    else:
        component_names: tuple[str, ...]
        if components is not None:
            component_names = tuple(components)
        else:
            pipeline_components = getattr(pipeline, "components", None)
            if isinstance(pipeline_components, Mapping):
                component_names = tuple(pipeline_components)
            else:
                component_names = (
                    "text_encoder",
                    "text_encoder_2",
                    "transformer",
                    "unet",
                    "vae",
                )

        selected_components = (
            (name, getattr(pipeline, name))
            for name in component_names
            if isinstance(getattr(pipeline, name, None), nn.Module)
        )

    results: dict[str, DynamicOffloadApplyResult] = {}
    for component_name, component_module in selected_components:
        if not isinstance(component_module, nn.Module):
            continue
        result = enable_offload(
            component_module,
            settings=settings,
            component=component_name,
            execution_device=execution_device,
            offload_device=offload_device,
            low_cpu_mem_usage=low_cpu_mem_usage,
            environ=environ if use_environment else {},
            apply_hook=apply_hook,
            record_event=record_event,
            **config_overrides,
        )
        if apply_hook and result.should_move_to_execution_device:
            result.module.to(execution_device)
            result.should_move_to_execution_device = False
            result.event_payload["unmanaged_module_moved_to_execution_device"] = str(execution_device)
        results[component_name] = result
    return results


def enable_dynamic_offload(
    module: nn.Module,
    *,
    settings: DynamicOffloadSettings | None = None,
    config: DynamicOffloadConfig | None = None,
    preset: str = "auto",
    execution_device: str | torch.device = "cuda:0",
    offload_device: str | torch.device = "cpu",
    target_module_classes: tuple[type[nn.Module], ...] = _DEFAULT_TARGET_MODULE_CLASSES,
    skip_modules_pattern: tuple[str, ...] = (),
    always_resident_modules_pattern: tuple[str, ...] | None = None,
    running_on_wsl: bool | None = None,
    environ: Mapping[str, str] | None = None,
    use_environment: bool = True,
    apply_hook: bool = True,
    record_event: Any | None = None,
    event_name: str = "setup_dynamic_offload",
    **config_overrides: Any,
) -> DynamicOffloadApplyResult:
    """Resolve DDO settings, optionally apply the hook, and return the prepared module.

    This is the high-level, Diffusers-style entry point. Environment variables are
    read by default for CLI/runner use, while explicit keyword overrides always
    take precedence over preset and environment values.
    """
    start = time.perf_counter()
    if settings is None:
        settings = load_dynamic_offload_settings_from_env(
            execution_device=execution_device,
            offload_device=offload_device,
            target_module_classes=target_module_classes,
            skip_modules_pattern=skip_modules_pattern,
            always_resident_modules_pattern=always_resident_modules_pattern,
            running_on_wsl=running_on_wsl,
            environ=environ if use_environment else {},
            default_preset=preset,
        )

    explicit_config = config is not None or bool(config_overrides)
    effective_config = config or settings.config
    if config_overrides:
        unknown = sorted(set(config_overrides) - set(DynamicOffloadConfig.__dataclass_fields__))
        if unknown:
            raise TypeError(f"Unknown DynamicOffloadConfig override(s): {', '.join(unknown)}")
        effective_config = replace(effective_config, **config_overrides)

    if effective_config is not settings.config:
        settings = replace(
            settings,
            config=effective_config,
            enabled=explicit_config or settings.plan or effective_config.execution_mode.lower() != "plan",
            effective_pin_cpu_memory=effective_config.pin_cpu_memory,
        )

    hook = None
    event_payload: dict[str, Any] = {}
    quantized_backend_payload = detect_quantized_backend_modules(module)
    if apply_hook and settings.enabled:
        hook = apply_dynamic_offload(module, effective_config)
        event_payload = build_dynamic_offload_event_payload(settings, hook.state, effective_config)
    if quantized_backend_payload:
        event_payload["quantized_backend"] = quantized_backend_payload

    elapsed = time.perf_counter() - start
    if record_event is not None:
        record_event(event_name, elapsed, **event_payload)

    return DynamicOffloadApplyResult(
        module=module,
        hook=hook,
        settings=settings,
        should_move_to_execution_device=hook is None or effective_config.execution_mode.lower() == "plan",
        event_payload=event_payload,
    )


@contextlib.contextmanager
def _temporary_safetensors_backend(backend: str):
    backend = backend.strip().lower()
    if not backend:
        yield
        return
    if backend not in {"mmap", "pread"}:
        raise ValueError("DynamicOffloadConfig.load_safetensors_backend must be '', 'mmap', or 'pread'")

    try:
        import safetensors
        import safetensors.torch
    except ImportError:
        yield
        return

    original_safe_open = safetensors.safe_open
    original_torch_load_file = safetensors.torch.load_file

    def safe_open_with_backend(filename, framework, device=None, **kwargs):
        kwargs.setdefault("backend", backend)
        if device is None:
            return original_safe_open(filename, framework=framework, **kwargs)
        return original_safe_open(filename, framework=framework, device=device, **kwargs)

    def torch_load_file_with_backend(filename, device="cpu", **kwargs):
        kwargs.setdefault("backend", backend)
        return original_torch_load_file(filename, device=device, **kwargs)

    safetensors.safe_open = safe_open_with_backend
    safetensors.torch.load_file = torch_load_file_with_backend

    patched_modules: list[tuple[Any, str, Any]] = []
    for module_name in (
        "accelerate.utils.modeling",
        "transformers.modeling_utils",
        "diffusers.models.model_loading_utils",
    ):
        try:
            imported_module = __import__(module_name, fromlist=["_"])
        except ImportError:
            continue
        if getattr(imported_module, "safe_open", None) is original_safe_open:
            imported_module.safe_open = safe_open_with_backend
            patched_modules.append((imported_module, "safe_open", original_safe_open))
        if getattr(imported_module, "safe_load_file", None) is original_torch_load_file:
            imported_module.safe_load_file = torch_load_file_with_backend
            patched_modules.append((imported_module, "safe_load_file", original_torch_load_file))

    try:
        yield
    finally:
        for imported_module, attr_name, original_value in reversed(patched_modules):
            setattr(imported_module, attr_name, original_value)
        safetensors.safe_open = original_safe_open
        safetensors.torch.load_file = original_torch_load_file


def from_pretrained_with_dynamic_offload(
    pretrained_model_name_or_path: str | os.PathLike[str],
    *,
    model_loader: Any | None = None,
    dynamic_offload_config: DynamicOffloadConfig | None = None,
    apply_dynamic: bool = False,
    **from_pretrained_kwargs: Any,
) -> DynamicOffloadLoadResult:
    if model_loader is None:
        from diffusers import AutoModel

        model_loader = AutoModel
    from_pretrained = getattr(model_loader, "from_pretrained", model_loader)
    config = dynamic_offload_config or DynamicOffloadConfig()
    with _temporary_safetensors_backend(config.load_safetensors_backend):
        module = from_pretrained(pretrained_model_name_or_path, **from_pretrained_kwargs)
    if not apply_dynamic:
        return DynamicOffloadLoadResult(module=module)

    hook = apply_dynamic_offload(module, config)
    return DynamicOffloadLoadResult(
        module=module,
        hook=hook,
        should_move_to_execution_device=config.execution_mode.lower() == "plan",
    )


def remove_dynamic_offload(module: nn.Module, recurse: bool = True) -> None:
    if hasattr(module, "_diffusers_hook"):
        module._diffusers_hook.remove_hook(_DYNAMIC_OFFLOAD_HOOK, recurse=recurse)


def get_dynamic_offload_state(module: nn.Module) -> DynamicOffloadState | None:
    if not hasattr(module, "_diffusers_hook"):
        return None
    hook = module._diffusers_hook.get_hook(_DYNAMIC_OFFLOAD_HOOK)
    if hook is None:
        return None
    return hook.state


def clear_dynamic_offload_plan_cache() -> None:
    with _DYNAMIC_OFFLOAD_PLAN_CACHE_LOCK:
        _DYNAMIC_OFFLOAD_PLAN_CACHE.clear()


def clear_dynamic_offload_pinned_tensor_cache() -> None:
    with _DYNAMIC_OFFLOAD_PINNED_TENSOR_CACHE_LOCK:
        _DYNAMIC_OFFLOAD_PINNED_TENSOR_CACHE.clear()


def build_dynamic_offload_plan(module: nn.Module, config: DynamicOffloadConfig) -> DynamicOffloadState:
    if not config.cache_plan:
        state = _build_dynamic_offload_plan_uncached(module, config)
        state.planner_decisions["plan_cache"] = "disabled"
        return state

    cache_key = _dynamic_offload_plan_cache_key(module, config)
    with _DYNAMIC_OFFLOAD_PLAN_CACHE_LOCK:
        cached_state = _DYNAMIC_OFFLOAD_PLAN_CACHE.get(cache_key)
    if cached_state is not None:
        state = cached_state.clone_for_runtime()
        state.planner_decisions["plan_cache"] = "hit"
        return state

    state = _build_dynamic_offload_plan_uncached(module, config)
    state.resolved_resident_module_patterns = [
        pattern.pattern for pattern in _resolve_resident_module_patterns(module, config)
    ]
    state.planner_decisions["plan_cache"] = "miss"
    with _DYNAMIC_OFFLOAD_PLAN_CACHE_LOCK:
        _DYNAMIC_OFFLOAD_PLAN_CACHE[cache_key] = state.clone_for_runtime()
    return state


def _build_dynamic_offload_plan_uncached(module: nn.Module, config: DynamicOffloadConfig) -> DynamicOffloadState:
    state = DynamicOffloadState()
    skip_patterns = tuple(re.compile(pattern) for pattern in config.skip_modules_pattern)
    resident_patterns = tuple(re.compile(pattern) for pattern in config.always_resident_modules_pattern)

    for module_name, submodule in module.named_modules():
        if module_name == "":
            continue
        if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
            continue
        if not isinstance(submodule, config.target_module_classes):
            continue

        state.module_count += 1
        module_resident = bool(resident_patterns and any(pattern.search(module_name) for pattern in resident_patterns))
        for tensor_name, tensor in _iter_local_tensors(submodule):
            placement = _classify_tensor(tensor, module_resident, config.small_tensor_threshold_bytes)
            entry = DynamicOffloadPlanEntry(
                module_name=module_name,
                module_type=submodule.__class__.__name__,
                tensor_name=tensor_name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                bytes=_tensor_size_bytes(tensor),
                placement=placement,
            )
            state.entries.append(entry)
            state.total_bytes += entry.bytes
            state.bytes_by_placement[placement] = state.bytes_by_placement.get(placement, 0) + entry.bytes

    return state


def _dynamic_offload_plan_cache_key(module: nn.Module, config: DynamicOffloadConfig) -> tuple[Any, ...]:
    return (
        module.__class__.__module__,
        module.__class__.__qualname__,
        tuple(cls.__module__ + "." + cls.__qualname__ for cls in config.target_module_classes),
        tuple(config.skip_modules_pattern),
        tuple(config.always_resident_modules_pattern),
        tuple(config.resident_module_patterns),
        int(config.small_tensor_threshold_bytes),
        _dynamic_offload_plan_structure_signature(module, config),
    )


def _dynamic_offload_plan_structure_signature(
    module: nn.Module,
    config: DynamicOffloadConfig,
) -> tuple[tuple[Any, ...], ...]:
    skip_patterns = tuple(re.compile(pattern) for pattern in config.skip_modules_pattern)
    signature: list[tuple[Any, ...]] = []
    for module_name, submodule in module.named_modules():
        if module_name == "":
            continue
        if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
            continue
        if not isinstance(submodule, config.target_module_classes):
            continue
        tensor_signature = tuple(
            (
                tensor_name,
                tuple(tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                tensor.numel(),
                tensor.element_size(),
            )
            for tensor_name, tensor in _iter_local_tensors(submodule)
        )
        signature.append(
            (
                module_name,
                submodule.__class__.__module__,
                submodule.__class__.__qualname__,
                tensor_signature,
            )
        )
    return tuple(signature)


def _iter_local_tensors(module: nn.Module):
    for name, parameter in module.named_parameters(recurse=False):
        yield name, parameter.data
    for name, buffer in module.named_buffers(recurse=False):
        yield name, buffer.data


def _resolve_resident_module_patterns(
    module: nn.Module,
    config: DynamicOffloadConfig,
) -> tuple[re.Pattern[str], ...]:
    raw_patterns = tuple(pattern.strip() for pattern in config.resident_module_patterns if pattern.strip())
    explicit_patterns = tuple(pattern for pattern in raw_patterns if pattern.lower() != "auto")
    compiled_patterns = [re.compile(pattern) for pattern in explicit_patterns]
    if not any(pattern.lower() == "auto" for pattern in raw_patterns):
        return tuple(compiled_patterns)

    inferred_patterns = _infer_repeated_resident_module_patterns(module, config)
    return (*compiled_patterns, *inferred_patterns)


def _infer_repeated_resident_module_patterns(
    module: nn.Module,
    config: DynamicOffloadConfig,
) -> tuple[re.Pattern[str], ...]:
    skip_patterns = tuple(re.compile(pattern) for pattern in config.skip_modules_pattern)
    repeated_module_pattern = re.compile(r"^(.+)\.(\d+)$")
    groups: dict[str, list[tuple[str, int]]] = {}

    for module_name, submodule in module.named_modules():
        match = repeated_module_pattern.match(module_name)
        if match is None:
            continue
        if skip_patterns and any(pattern.search(module_name) for pattern in skip_patterns):
            continue
        target_bytes = _target_module_size_bytes(submodule, config.target_module_classes)
        if target_bytes <= 0:
            continue
        groups.setdefault(match.group(1), []).append((module_name, target_bytes))

    candidates = [
        (prefix, members, sum(member_bytes for _, member_bytes in members))
        for prefix, members in groups.items()
        if len(members) >= 2
    ]
    if not candidates:
        return ()

    prefix, _, _ = max(candidates, key=lambda item: (item[2], len(item[1])))
    return (re.compile(rf"^{re.escape(prefix)}\.\d+$"),)


def _classify_tensor(tensor: torch.Tensor, module_resident: bool, small_tensor_threshold_bytes: int) -> str:
    if module_resident:
        return "resident_module"
    if _tensor_size_bytes(tensor) <= small_tensor_threshold_bytes:
        return "resident_small"
    return "streamed_large"


def _module_size_bytes(module: nn.Module) -> int:
    size = sum(_tensor_size_bytes(parameter.data) for parameter in module.parameters(recurse=True))
    size += sum(_tensor_size_bytes(buffer.data) for buffer in module.buffers(recurse=True))
    return size


def _target_module_size_bytes(module: nn.Module, target_module_classes: tuple[type[nn.Module], ...]) -> int:
    size = 0
    for submodule in module.modules():
        if not isinstance(submodule, target_module_classes):
            continue
        size += sum(_tensor_size_bytes(parameter.data) for parameter in submodule.parameters(recurse=False))
        size += sum(_tensor_size_bytes(buffer.data) for buffer in submodule.buffers(recurse=False))
    return size


def _is_module_descendant(module_name: str, parent_name: str) -> bool:
    return module_name.startswith(f"{parent_name}.")


def _spread_order(items: list[tuple[str, Any, int]], budget_bytes: int) -> list[tuple[str, Any, int]]:
    if len(items) <= 2:
        return items
    average_bytes = max(1, sum(item[2] for item in items) // len(items))
    target_count = max(1, min(len(items), budget_bytes // average_bytes))
    if target_count == 1:
        spread_indices = [len(items) // 2]
    else:
        spread_indices = [
            round(position * (len(items) - 1) / (target_count - 1))
            for position in range(target_count)
        ]
    seen = set()
    ordered: list[tuple[str, nn.Linear, int]] = []
    for index in spread_indices:
        if index not in seen:
            seen.add(index)
            ordered.append(items[index])
    for index, item in enumerate(items):
        if index not in seen:
            ordered.append(item)
    return ordered


def _largest_first_order(items: list[tuple[str, Any, int]]) -> list[tuple[str, Any, int]]:
    return sorted(items, key=lambda item: item[2], reverse=True)


def _tensor_size_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
