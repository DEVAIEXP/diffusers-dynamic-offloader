"""Generate a local platform prerelease pair; never publish, tag, or execute CUDA.

Use the isolated locked tools/release project. Rebuilds stage source snapshots,
preserve the checkouts and previous candidates, and require a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib

from packaging.version import Version

FEATURES = {
    "DDO_VMM_NATIVE_SPAN_POOL": "ON",
    "DDO_VMM_NATIVE_JOINED_SPANS": "OFF",
    "DDO_VMM_BUILD_PROBES": "OFF",
    "DDO_VMM_BUILD_EXECUTION_OPS": "OFF",
    "DDO_VMM_NATIVE_SOURCE_DISPATCHER": "OFF",
    "DDO_VMM_SOURCE_TEST_HOOKS": "OFF",
    "DDO_VMM_ALLOCATION_DIAGNOSTICS": "OFF",
}
REQUIRED_EXPORTS = {
    "ddo_vmm_torch_malloc", "ddo_vmm_torch_free", "ddo_vmm_source_abi_version",
    "ddo_vmm_source_cache_create", "ddo_vmm_model_arena_create",
}


def platform_profile(target):
    """Keep reviewed Windows features separate from the Linux ABI profile."""
    if target == "win32":
        return "win_amd64", {**FEATURES, "DDO_VMM_NATIVE_JOINED_SPANS": "ON",
                              "DDO_VMM_ALLOCATION_DIAGNOSTICS": "ON"}
    if target == "linux":
        return "linux_x86_64", dict(FEATURES)
    raise ValueError("Only Windows AMD64 and Linux x86-64 profiles are reviewed")


def audit_windows(library, cwd, dumpbin):
    """Inspect PE dependencies/exports without loading the DLL or driver."""
    headers = capture([str(dumpbin), "/headers", str(library)], cwd)
    if not re.search(r"8664 machine \(x64\)", headers) or "DLL" not in headers:
        raise ValueError("Expected an AMD64 PE DLL")
    dependencies = capture([str(dumpbin), "/dependents", str(library)], cwd)
    needed = sorted(set(re.findall(r"(?im)^\s+([\w.-]+\.dll)\s*$", dependencies)))
    allowed = {"kernel32.dll", "nvcuda.dll", "dxgi.dll", "advapi32.dll", "user32.dll"}
    if not needed or {name.lower() for name in needed} - allowed:
        raise ValueError(f"Unreviewed PE dependencies: {needed}")
    symbols = capture([str(dumpbin), "/exports", str(library)], cwd)
    exports = set(re.findall(r"(?m)^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\w+)", symbols))
    if REQUIRED_EXPORTS - exports:
        raise ValueError(f"Missing native exports: {sorted(REQUIRED_EXPORTS - exports)}")
    return {"schema_version": 1, "platform": "win_amd64", "sha256": sha256(library),
            "needed": needed, "required_exports": sorted(REQUIRED_EXPORTS),
            "driver_link": "external_system_nvcuda", "machine": "AMD64",
            "compatibility": "PE audit only; separate installed Windows CUDA validation required"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def staged_metadata(text, version, allocator_version=None):
    """Change only staged project metadata, never the user's pyproject."""
    if not Version(version).is_prerelease:
        raise ValueError("This generator accepts prerelease versions only (for example 0.1.0rc1)")
    original = tomllib.loads(text)
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
    if project is None:
        raise ValueError("Missing [project] metadata")
    body, count = re.subn(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{version}"', project[1])
    if count != 1:
        raise ValueError("Expected one static project version")
    text = text[:project.start(1)] + body + text[project.end(1):]
    if allocator_version:
        if not Version(allocator_version).is_prerelease:
            raise ValueError("Allocator version must also be a prerelease")
        extra = re.search(r"(?ms)^vmm\s*=\s*\[(.*?)\]", text)
        if extra is None or "ddo-vmm-allocator" in extra[1]:
            raise ValueError("Expected an unpinned allocator-free vmm extra before staging")
        text = text[:extra.start(1)] + f'\n    "ddo-vmm-allocator=={allocator_version}",' + text[extra.start(1):]
    result = tomllib.loads(text)
    assert result["project"]["name"] == original["project"]["name"]
    assert result["project"]["version"] == version
    return text


def snapshot(root, destination, entries):
    """Copy only public build inputs, refusing escaping symlinks and caches."""
    hashes = {}
    for entry in entries:
        source = root / entry
        if not source.exists():
            raise ValueError(f"Missing build input: {entry}")
        files = source.rglob("*") if source.is_dir() else [source]
        for file in sorted(files):
            if not file.is_file():
                continue
            relative = file.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" or
                   part.endswith((".egg-info", ".dist-info")) for part in relative.parts):
                continue
            if file.suffix in {".pyc", ".dll", ".so", ".safetensors", ".o", ".obj", ".a", ".lib"}:
                continue
            if not file.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Escaping source symlink: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            hashes[relative.as_posix()] = sha256(target)
    return hashes


def capture(command, cwd):
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def git_identity(root):
    # Local Git warnings are diagnostics, not part of the commit identifier.
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True,
                                       stderr=subprocess.PIPE).strip()

    commit = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Invalid source commit identity")
    return {
        "commit": commit,
        "dirty": bool(git("status", "--porcelain")),
    }


def audit_linux(library, cwd, *, driver_stub=False, abi_baseline=None):
    """Observe host ELF requirements; never assert manylinux compatibility."""
    dynamic = capture(["readelf", "-d", str(library)], cwd)
    if re.search(r"\((?:RPATH|RUNPATH)\)", dynamic):
        raise ValueError("Unexpected ELF RPATH/RUNPATH; refusing a path-dependent wheel")
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
    allowed = {"libcuda.so.1", "libstdc++.so.6", "libgcc_s.so.1", "libc.so.6",
               "libm.so.6", "libpthread.so.0", "librt.so.1", "libdl.so.2", "ld-linux-x86-64.so.2"}
    if set(needed) - allowed:
        raise ValueError(f"Unreviewed ELF dependencies: {sorted(set(needed) - allowed)}")
    symbols = capture(["nm", "-D", "--defined-only", str(library)], cwd)
    exports = {line.split()[-1] for line in symbols.splitlines() if line.split()}
    if REQUIRED_EXPORTS - exports:
        raise ValueError(f"Missing native exports: {sorted(REQUIRED_EXPORTS - exports)}")
    versions = capture(["readelf", "--version-info", str(library)], cwd)
    requirements = sorted(set(re.findall(r"(?:GLIBCXX|GLIBC|CXXABI)_[0-9.]+", versions)))
    linkage = capture(["ldd", str(library)], cwd)
    missing = re.findall(r"^\s*(\S+)\s+=>\s+not found", linkage, re.MULTILINE)
    if "not found" in linkage and (not missing or set(missing) -
                                  ({"libcuda.so.1"} if driver_stub else set())):
        raise ValueError("Native dependency is unresolved on the build host")
    if abi_baseline == "ubuntu22.04":
        ceilings = {"GLIBC": (2, 35), "GLIBCXX": (3, 4, 30), "CXXABI": (1, 3, 13)}
        for requirement in requirements:
            family, version = requirement.split("_", 1)
            if tuple(map(int, version.split("."))) > ceilings[family]:
                raise ValueError(f"Symbol exceeds Ubuntu 22.04 ABI ceiling: {requirement}")
    return {"schema_version": 1, "platform": "linux_x86_64", "sha256": sha256(library),
            "needed": needed, "required_symbol_versions": requirements,
            "required_exports": sorted(REQUIRED_EXPORTS), "manylinux_validated": False,
            "abi_baseline": abi_baseline or "build_host",
            "driver_link": "toolkit_stub_external_driver_required" if driver_stub else "build_host",
            "unresolved_external_driver": bool(missing),
            "compatibility": "ABI audit only; installed native Linux/WSL CUDA validation required"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--allocator-version")
    parser.add_argument("--cuda-toolkit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-driver-stub", type=Path,
                        help="Explicit toolkit link stub for GPU-free builds; never bundled or loaded")
    parser.add_argument("--abi-baseline", choices=["ubuntu22.04"])
    parser.add_argument("--dumpbin", type=Path, help="Reviewed MSVC dumpbin.exe, required on Windows")
    parser.add_argument("--allow-dirty", action="store_true", help="Record a development snapshot, never publication-ready")
    parser.add_argument("--dry-run", action="store_true", help="Read metadata/print plan only; no source copy, build or CUDA")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    allocator = root / "ddo-vmm-allocator"
    version = str(Version(args.version))
    allocator_version = str(Version(args.allocator_version or version))
    staged_metadata((root / "pyproject.toml").read_text(), version, allocator_version)
    staged_metadata((allocator / "pyproject.toml").read_text(), allocator_version)
    if sys.platform not in {"linux", "win32"} or platform.machine().lower() not in {"x86_64", "amd64"}:
        parser.error("Requires Linux x86-64 or Windows AMD64")
    windows = sys.platform == "win32"
    tag, features = platform_profile(sys.platform)
    if windows and (not args.dumpbin or args.cuda_driver_stub or args.abi_baseline):
        parser.error("Windows requires --dumpbin and refuses Linux stub/ABI options")
    dumpbin = args.dumpbin.resolve(strict=True) if args.dumpbin else None
    if not 1 <= args.jobs <= 64:
        parser.error("jobs must be between 1 and 64")
    identities = {"ddo": git_identity(root), "allocator": git_identity(allocator)}
    if any(item["dirty"] for item in identities.values()) and not args.allow_dirty:
        parser.error("Dirty checkout: review/commit it, or explicitly use --allow-dirty for a development candidate")
    toolkit = args.cuda_toolkit.resolve(strict=True)
    driver_stub = args.cuda_driver_stub.resolve(strict=True) if args.cuda_driver_stub else None
    if driver_stub and (not driver_stub.is_relative_to(toolkit) or
                        "stubs" not in driver_stub.parts or driver_stub.name != "libcuda.so"):
        parser.error("Driver stub must be toolkit-local stubs/libcuda.so")
    output = (args.output or root / ".internal/releases" / version /
              ("windows-amd64" if windows else "linux-x86_64")).resolve()
    if output.exists():
        parser.error("Candidate directory already exists; choose a new version/output, never overwrite")
    plan = {"version": version, "allocator_version": allocator_version, "output": str(output),
            "features": features, "sources": identities, "cuda_tests": False,
            "publication": False, "tag": f"py3-none-{tag}", "manylinux": False,
            "dumpbin": str(dumpbin) if dumpbin else None,
            "abi_baseline": args.abi_baseline, "cuda_driver_stub": str(driver_stub) if driver_stub else None}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    output.mkdir(parents=True, exist_ok=False)
    dist = output / "dist"
    dist.mkdir()
    logs = output / "logs"
    logs.mkdir()
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "PYTHONHOME"} and not key.startswith("DDO_VMM_WHEEL_")}
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    manifest = {**plan, "status": "building", "publication_ready": False,
                "tools": {name: importlib.metadata.version(name) for name in
                          ("build", "hatchling", "setuptools", "wheel", "packaging")}}

    def run(label, command, cwd=root, env=None):
        print(f"[{label}] {' '.join(map(str, command))}", flush=True)
        with (logs / f"{label}.log").open("w") as log:
            subprocess.run(list(map(str, command)), cwd=cwd, env=env or environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        sources = output / "sources"
        ddo_stage, allocator_stage = sources / "ddo", sources / "allocator"
        manifest["source_files"] = {
            "ddo": snapshot(root, ddo_stage, ["pyproject.toml", "README.md", "LICENSE", "src"]),
            "allocator": snapshot(allocator, allocator_stage,
                ["pyproject.toml", "README.md", "LICENSE", "hatch_build.py", "CMakeLists.txt", "src", "native"]),
        }
        for stage, selected_version, pair in ((ddo_stage, version, allocator_version),
                                              (allocator_stage, allocator_version, None)):
            metadata = stage / "pyproject.toml"
            metadata.write_text(staged_metadata(metadata.read_text(), selected_version, pair), encoding="utf-8")
        manifest["staged_metadata_sha256"] = {"ddo": sha256(ddo_stage / "pyproject.toml"),
                                               "allocator": sha256(allocator_stage / "pyproject.toml")}
        native_build, cpu_build = output / "build-native", output / "build-cpu"
        generator = ["-G", "Visual Studio 17 2022", "-A", "x64"] if windows else []
        configuration = ["--config", "Release"] if windows else []
        run("configure-cpu", ["cmake", "-S", allocator_stage / "native/tests/cpu", "-B", cpu_build,
                              "-DCMAKE_BUILD_TYPE=Release", *generator])
        run("build-cpu", ["cmake", "--build", cpu_build, *configuration, "--parallel", args.jobs])
        run("test-cpu", ["ctest", "--test-dir", cpu_build, *(["-C", "Release"] if windows else []),
                         "--output-on-failure", "--no-tests=error"])
        run("configure-native", ["cmake", "-S", allocator_stage, "-B", native_build,
                                  "-DCMAKE_BUILD_TYPE=Release", *generator, f"-DCUDAToolkit_ROOT={toolkit}",
                                  *([f"-DCUDA_cuda_driver_LIBRARY={driver_stub}", "-DCMAKE_SKIP_RPATH=ON"]
                                    if driver_stub else []),
                                  *[f"-D{name}={value}" for name, value in features.items()]])
        run("build-native", ["cmake", "--build", native_build, *configuration, "--parallel", args.jobs])
        library = native_build / "Release/ddo_vmm.dll" if windows else native_build / "libddo_vmm.so"
        abi = (audit_windows(library, root, dumpbin) if windows else
               audit_linux(library, root, driver_stub=bool(driver_stub), abi_baseline=args.abi_baseline))
        manifest["native"] = abi
        abi_path = output / "native-abi.json"
        abi_path.write_text(json.dumps(abi, indent=2) + "\n", encoding="utf-8")
        manifest["toolchain"] = {"cmake": capture(["cmake", "--version"], root).splitlines()[0],
                                   "compiler": ("\n".join(path.read_text() for path in sorted(
                                       (native_build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake")))
                                                if windows else capture(["c++", "--version"], root).splitlines()[0]),
                                   "toolkit": capture([str(toolkit / ("bin/nvcc.exe" if windows else "bin/nvcc")),
                                                       "--version"], root),
                                   "libc": platform.libc_ver()}
        native_env = {**environment, "DDO_VMM_WHEEL_LIBRARY": str(library),
                      "DDO_VMM_WHEEL_SHA256": abi["sha256"], "DDO_VMM_WHEEL_ABI_REPORT": str(abi_path)}
        run("wheel-allocator", [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist, allocator_stage], env=native_env)
        run("wheel-ddo", [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist, ddo_stage])
        wheels = sorted(dist.glob("*.whl"))
        if len(wheels) != 2:
            raise ValueError("Expected exactly two candidate wheels")
        sys.path.insert(0, str(root / "scripts"))
        from verify_native_delivery_cpu import verify
        manifest["wheels"] = []
        for wheel in wheels:
            native = wheel.name.startswith("ddo_vmm_allocator-")
            result = verify(wheel, abi["sha256"] if native else None,
                            source_root=(allocator_stage if native else ddo_stage) / "src")
            manifest["wheels"].append({"file": wheel.name, "sha256": result["sha256"]})
        clean = output / "installed-cpu"
        run("create-clean-environment", ["uv", "venv", "--python", sys.executable, clean])
        installed_python = clean / ("Scripts/python.exe" if windows else "bin/python")
        run("install-candidate-wheels", ["uv", "pip", "install", "--python", installed_python,
                                          "--no-deps", *wheels])
        run("installed-cpu-discovery", [installed_python, "-I", root / "scripts/probe_installed_native_cpu.py"], cwd=output)
        manifest.update(status="cpu_candidate_passed", installed_cuda_validation="not_run",
                        publication_ready=False)
        (dist / "SHA256SUMS").write_text("".join(f"{sha256(wheel)}  {wheel.name}\n" for wheel in wheels), encoding="utf-8")
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(output / "release-manifest.json", dist / "release-manifest.json")
    print(f"Candidate ready for separate installed CUDA review: {dist}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
