"""Build-only recipe for an isolated external kernel, outside the DDO package."""
import argparse
import hashlib
import json
import os
from pathlib import Path


REVISION = "400ff39852d08df4572cea7fca4a800b0a99cc36"
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Pinned external checkout root, including its original LICENSE.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--describe", action="store_true", help="No imports of Torch, build or CUDA.")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    build_dir = args.build_dir.resolve()
    cuda = source_root / "kernel/qk_norm_rope_cuda/qk_norm_rope.cu"
    header = source_root / "kernel/torch-ext/torch_binding.h"
    license_path = source_root / "LICENSE"
    recipe = dict(source_revision=REVISION, sources=[str(ROOT / "bindings.cpp"), str(cuda)],
                  include_paths=[str(header.parent)], build_dir=str(build_dir),
                  architecture="8.6", max_jobs=1, cuda_execution=False,
                  namespace="external_qk_rope_bench", external_license="Apache-2.0")
    if args.describe:
        print(json.dumps(recipe, indent=2))
        return
    if os.name != "nt":
        raise RuntimeError("This recipe is scoped to the Windows benchmark environment")
    for path in (cuda, header, license_path, source_root / "SOURCE_REVISION"):
        if not path.is_file():
            raise RuntimeError(f"Missing attributed pinned source: {path}")
    if (source_root / "SOURCE_REVISION").read_text().strip() != REVISION:
        raise RuntimeError("External source revision differs from the reviewed recipe")
    # Refuse reusing an old binary directory or modifying the native allocator.
    build_dir.mkdir(parents=True, exist_ok=False)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    os.environ["MAX_JOBS"] = "1"
    import torch
    from torch.utils.cpp_extension import load
    if torch.cuda.is_initialized():
        raise RuntimeError("Build-only process must not initialize CUDA")
    library = load(name="external_qk_rope_bench", sources=recipe["sources"],
                   extra_include_paths=recipe["include_paths"],
                   extra_cflags=["/O2", "/Zc:preprocessor"],
                   extra_cuda_cflags=["-O3", "-Xcompiler", "/Zc:preprocessor"],
                   build_directory=str(build_dir), with_cuda=True,
                   is_python_module=False, verbose=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization during build")
    recipe.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  library=str(library), library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
                  source_hashes={str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (cuda, header, license_path)})
    # Generated build evidence, not repository documentation.
    (build_dir / "build_manifest.json").write_text(json.dumps(recipe, indent=2))
    print(json.dumps(recipe, indent=2))


if __name__ == "__main__":
    main()
