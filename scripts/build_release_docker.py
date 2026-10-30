"""Orchestrate a GPU-free Ubuntu 22.04 candidate build from Windows or Linux.

Source is mounted read-only, the image context contains only release tools, and
existing candidates are refused. Nothing is published, tagged in Git or promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def commands(root, version, output, image, jobs, allow_dirty):
    """Return argv lists, not shell commands; mounts preserve paths with spaces."""
    if not re.fullmatch(r"\d+\.\d+\.\d+rc\d+", version):
        raise ValueError("Select an explicit prerelease version, for example 0.1.0rc2")
    if not 1 <= jobs <= 8:
        raise ValueError("Docker build jobs must be between 1 and 8")
    for path in (root, output.parent):
        if "," in str(path):
            raise ValueError("Docker mount paths cannot contain commas")
    build = ["docker", "build", "--platform", "linux/amd64", "--tag", image,
             "--file", str(root / "tools/release/Dockerfile"), str(root / "tools/release")]
    run = ["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
           "--cpus", "2", "--memory", "6g", "--memory-swap", "6g", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--env", "GIT_CONFIG_COUNT=2",
           "--env", "GIT_CONFIG_KEY_0=safe.directory", "--env", "GIT_CONFIG_VALUE_0=/src",
           "--env", "GIT_CONFIG_KEY_1=safe.directory",
           "--env", "GIT_CONFIG_VALUE_1=/src/ddo-vmm-allocator",
           "--mount", f"type=bind,source={root},target=/src,readonly",
           "--mount", f"type=bind,source={output.parent},target=/out",
           image, "python", "-B", "/src/scripts/build_release.py", "--version", version,
           "--cuda-toolkit", "/usr/local/cuda", "--cuda-driver-stub",
           "/usr/local/cuda/lib64/stubs/libcuda.so", "--abi-baseline", "ubuntu22.04",
           "--jobs", str(jobs), "--output", f"/out/{output.name}"]
    if allow_dirty:
        run.append("--allow-dirty")
    return build, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root / ".internal/releases" / args.version / "docker-linux-x86_64").resolve()
    if output.exists():
        parser.error("Candidate output exists; choose a new version/path, never overwrite")
    image = f"ddo-release-ubuntu2204:{args.version}"
    build, run = commands(root, args.version, output, image, args.jobs, args.allow_dirty)
    print(json.dumps({"build": build, "run": run, "cuda": False, "publication": False}, indent=2), flush=True)
    if args.dry_run:
        return 0
    engine = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True))
    if engine.get("OSType") != "linux" or engine.get("Architecture") not in {"x86_64", "amd64"}:
        parser.error("A Linux x86-64 Docker engine is required")
    subprocess.run(build, check=True)
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    # Use the exact built image, even if its human-readable tag changes later.
    run[run.index(image)] = image_id
    output.parent.mkdir(parents=True, exist_ok=True)
    record = output.parent / f"{output.name}-docker-build.json"
    with record.open("x", encoding="utf-8") as stream:
        json.dump({"image_id": image_id, "build": build, "run": run,
                   "recipe_sha256": hashlib.sha256((root / "tools/release/Dockerfile").read_bytes()).hexdigest(),
                   "lock_sha256": hashlib.sha256((root / "tools/release/uv.lock").read_bytes()).hexdigest(),
                   "installed_cuda_validation": "not_run", "publication_ready": False}, stream, indent=2)
    subprocess.run(run, check=True)
    # Keep reviewed image provenance with downloadable distribution artifacts.
    shutil.copyfile(record, output / "dist/docker-build.json")
    print(f"Candidate: {output / 'dist'}; separate installed Linux/WSL validation remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
