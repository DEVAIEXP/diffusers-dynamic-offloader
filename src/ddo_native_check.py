"""Small opt-in installed-allocator check, without model loading or benchmarking."""

from __future__ import annotations

import argparse


def _run_cuda() -> None:
    # Bootstrap must have installed the allocator before these imports.
    import torch
    from ddo_vmm_allocator.torch_allocator import installed_library_path

    library = installed_library_path()
    if library is None:
        raise RuntimeError("DDO native allocator was not installed")

    x = torch.ones(1024, device="cuda:0", dtype=torch.int32)
    y = x + 1
    torch.cuda.synchronize()
    if y.sum().item() != 2048:
        raise RuntimeError("Native CUDA result mismatch")

    print("Native library:", library)
    print("DDO native CUDA: PASS")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-cuda", action="store_true",
        help="Explicitly run a small CUDA operation through the native allocator.",
    )
    args = parser.parse_args(argv)
    if not args.run_cuda:
        parser.error("pass --run-cuda to authorize GPU execution")

    from ddo_vmm_bootstrap import ensure_vmm_fast_process

    ensure_vmm_fast_process(allocator_cache_mib=32, argv=[__file__, "--run-cuda"])
    _run_cuda()


if __name__ == "__main__":
    main()
