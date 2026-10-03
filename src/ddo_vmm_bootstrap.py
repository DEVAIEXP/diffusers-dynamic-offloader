"""Fresh-process bootstrap for the experimental native VMM backend.

This module deliberately has no dependency on :mod:`torch` or the DDO package.
Import it as the first executable action in a VMM-fast runner, so the child can
install the native allocator before the runner imports Torch or Diffusers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

_CHILD_MARKER = "DDO_VMM_FAST_CHILD"
_BOOTSTRAP = """
from ddo_vmm_allocator.torch_allocator import install_pytorch_vmm_allocator
import runpy
import sys
from pathlib import Path

install_pytorch_vmm_allocator()
target = sys.argv[1]
sys.path.insert(0, str(Path(target).resolve().parent))
sys.argv = [target, *sys.argv[2:]]
runpy.run_path(target, run_name='__main__')
"""


def vmm_fast_requested(argv: Sequence[str] | None = None, *, default_backend: str = "linear") -> bool:
    """Return whether a runner invocation selects the native VMM route.

    It reads only the two public runner arguments before CUDA-owning imports.
    The protected NVMe baseline and linear routes stay in the original process.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    backend = default_backend
    preset = "auto"
    for index, argument in enumerate(arguments):
        if argument.startswith("--ddo-backend="):
            backend = argument.partition("=")[2]
        elif argument == "--ddo-backend" and index + 1 < len(arguments):
            backend = arguments[index + 1]
        elif argument.startswith("--ddo-preset="):
            preset = argument.partition("=")[2]
        elif argument == "--ddo-preset" and index + 1 < len(arguments):
            preset = arguments[index + 1]
    return backend.strip().lower() == "vmm" and preset.strip().lower() == "auto"


def _default_allocator_root() -> Path | None:
    """Find the checked-out optional allocator without importing it."""
    candidate = Path(__file__).resolve().parents[1] / "ddo-vmm-allocator"
    return candidate if candidate.is_dir() else None


def ensure_vmm_fast_process(
    *,
    library_path: str | os.PathLike[str] | None = None,
    allocator_cache_mib: int = 512,
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Run this script under the native allocator or exit the parent process.

    Call this as the first executable statement of a VMM-fast runner. In the
    child this function is a no-op. ``library_path`` is optional only when
    ``DDO_VMM_LIBRARY_PATH`` identifies the platform library.
    """
    environment = os.environ if environ is None else environ
    if environment.get(_CHILD_MARKER) == "1":
        return
    if allocator_cache_mib < 0:
        raise ValueError("allocator_cache_mib must be non-negative")
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        raise RuntimeError("vmm_fast bootstrap must run before CUDA is initialized")

    arguments = list(sys.argv if argv is None else argv)
    if not arguments or not arguments[0]:
        raise RuntimeError("VMM bootstrap requires a runner script in argv[0]")
    target = Path(arguments[0]).resolve(strict=True)
    library = library_path or environment.get("DDO_VMM_LIBRARY_PATH")
    if not library:
        root = _default_allocator_root()
        if root is not None:
            candidates = (
                (root / "build" / "Release" / "ddo_vmm.dll",)
                if sys.platform == "win32" else
                (root / "build-linux" / "libddo_vmm.so", root / "build" / "libddo_vmm.so")
                if sys.platform == "linux" else ()
            )
            for candidate in candidates:
                if candidate.is_file():
                    library = str(candidate)
                    break
    if not library:
        raise RuntimeError("vmm_fast requires DDO_VMM_LIBRARY_PATH or an explicit library_path")
    library = str(Path(library).resolve(strict=True))

    root = _default_allocator_root()
    if root is None:
        raise RuntimeError("vmm_fast requires the ddo-vmm-allocator submodule or installed package")
    child_environment = dict(environment)
    child_environment[_CHILD_MARKER] = "1"
    child_environment["DDO_VMM_LIBRARY_PATH"] = library
    child_environment["DDO_VMM_TORCH_CACHE_MIB"] = str(allocator_cache_mib)
    source_root = str(root / "src")
    existing_path = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = source_root if not existing_path else source_root + os.pathsep + existing_path
    result = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP, str(target), *arguments[1:]],
        env=child_environment,
        check=False,
    )
    raise SystemExit(result.returncode)
