"""Isolated native-wheel discovery smoke test; never load DLLs or execute CUDA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import ddo_vmm_bootstrap as bootstrap


def main():
    if sys.platform != "win32":
        raise RuntimeError("This selected delivery smoke test requires Windows")
    if bootstrap._default_allocator_root() is not None:
        raise RuntimeError("Smoke test must use an installed DDO wheel, not its source checkout")
    package = bootstrap._installed_allocator_root()
    if package is None:
        raise RuntimeError("Native adapter package is not installed")
    library = bootstrap._discover_library(None, package)
    if library is None:
        raise RuntimeError("Installed adapter has no platform native binary")
    manifest = json.loads((library.parent / "artifact.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    if manifest["sha256"] != digest or manifest["platform"] != "win_amd64":
        raise RuntimeError("Packaged artifact manifest does not match")
    with tempfile.TemporaryDirectory(prefix="ddo-wheel-smoke-") as directory:
        runner = Path(directory) / "runner.py"
        runner.write_text("raise AssertionError('CUDA runner must never execute in this probe')\n")
        with patch.object(bootstrap.subprocess, "run") as launch:
            launch.return_value.returncode = 0
            try:
                bootstrap.ensure_vmm_fast_process(argv=[str(runner), "--ddo-backend", "vmm"], environ={})
            except SystemExit as error:
                if error.code != 0:
                    raise
            else:
                raise RuntimeError("Bootstrap did not request a child process")
        environment = launch.call_args.kwargs["env"]
        if environment["DDO_VMM_LIBRARY_PATH"] != str(library.resolve()):
            raise RuntimeError("Bootstrap selected a different binary")
        if "PYTHONPATH" in environment:
            raise RuntimeError("Installed bootstrap unexpectedly depends on checkout PYTHONPATH")
    if any(name in sys.modules for name in ("torch", "cuda", "ddo_vmm_allocator")):
        raise RuntimeError("CPU parent imported a CUDA-owning dependency or allocator initializer")
    print(json.dumps({
        "status": "passed",
        "scope": "installed-wheel metadata/hash/discovery and mocked child launch only",
        "library": str(library), "sha256": digest,
        "cuda_executed": False, "dll_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
