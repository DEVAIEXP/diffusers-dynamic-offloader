"""Opt-in native idle device-cache release at completed phase boundaries."""

import gc
import time

import torch

from .metrics import DynamicOffloadMetrics


def _installed_native():
    from ddo_vmm_allocator.native import NativeVmmLibrary
    from ddo_vmm_allocator.torch_allocator import installed_library_path

    path = installed_library_path()
    return NativeVmmLibrary(path) if path is not None else None


def release_native_vmm_idle_cache(*, enabled=False, phase="", device=None, clear_cublas_workspaces=False):
    """Release only whole-idle backing, never live tensors or model weights.

    Caller must detach previous components and drop their references first.
    Synchronizes once at this explicit phase boundary. Does not initialize CUDA,
    alter budgets, purge CPU standby or promise all process VRAM becomes free.
    Native proof/release failures propagate; unsupported is not success.
    Optional cuBLAS cleanup uses a guarded private PyTorch API, restricted to
    a quiescent single-GPU process on the compute thread without graph capture.
    No concurrent compute may run during this explicit boundary operation.
    """
    started = time.perf_counter()
    observation = {"phase": phase, "status": "disabled", "trim_executed": False}
    if enabled:
        if not torch.cuda.is_initialized():
            observation["status"] = "cuda_not_initialized"
        else:
            native = _installed_native()
            if native is None:
                observation["status"] = "native_allocator_not_installed"
            else:
                clear_blas = None
                if clear_cublas_workspaces:
                    clear_blas = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
                    if not callable(clear_blas):
                        raise RuntimeError("PyTorch cuBLAS workspace cleanup is unsupported")
                    if torch.cuda.device_count() != 1 or torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("cuBLAS cleanup requires a single GPU without graph capture")
                gc.collect()
                torch.cuda.synchronize(device)
                if clear_blas is not None:
                    observation["pool_before_cublas_clear"] = native.torch_pool_stats()
                    clear_blas()
                    observation["cublas_workspaces_cleared"] = True
                observation.update(native.trim_idle_torch_pool())
                observation["library"] = str(native.path)
    observation["elapsed_seconds"] = time.perf_counter() - started
    return DynamicOffloadMetrics(
        route="native_vmm_idle_cache_release",
        configuration={"enabled": bool(enabled), "clear_cublas_workspaces": bool(clear_cublas_workspaces)},
        setup={"device_cache_release": observation},
    )
