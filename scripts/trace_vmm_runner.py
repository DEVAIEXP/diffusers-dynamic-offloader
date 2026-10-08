"""Opt-in diagnostic timeline for a VMM-fast runner; never a benchmark gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

from ddo_vmm_bootstrap import ensure_vmm_fast_process, vmm_fast_requested


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("target", type=Path)
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def run(args):
    target = args.target.resolve(strict=True)
    output = args.trace_output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite trace: {output}")
    if not vmm_fast_requested(args.target_args):
        raise ValueError("diagnostic requires explicit --ddo-backend vmm --ddo-preset auto")
    # Relaunch this wrapper, not the target: allocator installation must
    # precede the profiler and all CUDA-owning imports in the fresh child.
    ensure_vmm_fast_process()
    import torch

    if not torch.profiler.kineto_available():
        raise RuntimeError("CUDA timeline requires a Kineto-enabled PyTorch build")
    original_argv, original_path = sys.argv, list(sys.path)
    sys.argv = [str(target), *args.target_args]
    sys.path.insert(0, str(target.parent))
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            runpy.run_path(str(target), run_name="__main__")
        output.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(output))
        print(f"[ddo-trace] diagnostic only; profiled timings are not benchmark results: {output}")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


if __name__ == "__main__":
    run(parse_args())
