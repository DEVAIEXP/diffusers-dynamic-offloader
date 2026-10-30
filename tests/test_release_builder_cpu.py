"""Release staging/admission contracts without builds, native loads or CUDA."""
import importlib.util
import json
from pathlib import Path
import tomllib

import pytest

spec = importlib.util.spec_from_file_location("ddo_release_builder", Path(__file__).parents[1] / "scripts/build_release.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

METADATA = '''[project]
name = "test-ddo"
version = "0.1.0"
dependencies = ["torch"]
[project.optional-dependencies]
vmm = ["cuda-core"]
'''


def test_stage_versions_and_optional_pair_without_changing_base_dependencies():
    staged = tomllib.loads(builder.staged_metadata(METADATA, "0.1.0rc1", "0.2.0rc2"))
    assert staged["project"]["version"] == "0.1.0rc1"
    assert staged["project"]["dependencies"] == ["torch"]
    assert staged["project"]["optional-dependencies"]["vmm"] == ["ddo-vmm-allocator==0.2.0rc2", "cuda-core"]
    assert tomllib.loads(METADATA)["project"]["version"] == "0.1.0"


def test_windows_profile_does_not_mutate_linux_features():
    tag, features = builder.platform_profile("win32")
    assert tag == "win_amd64"
    assert features["DDO_VMM_NATIVE_JOINED_SPANS"] == "ON"
    assert features["DDO_VMM_ALLOCATION_DIAGNOSTICS"] == "ON"
    assert builder.FEATURES["DDO_VMM_NATIVE_JOINED_SPANS"] == "OFF"
    assert builder.platform_profile("linux")[0] == "linux_x86_64"


def test_git_identity_keeps_warning_channel_out_of_commit(monkeypatch, tmp_path):
    def output(command, **kwargs):
        assert kwargs["stderr"] == builder.subprocess.PIPE
        return "a" * 40 + "\n" if "rev-parse" in command else " M README.md\n"

    monkeypatch.setattr(builder.subprocess, "check_output", output)
    assert builder.git_identity(tmp_path) == {"commit": "a" * 40, "dirty": True}


@pytest.mark.parametrize("dependency,headers,missing,passes", [
    ("nvcuda.dll", "8664 machine (x64) DLL", False, True),
    ("cudart64_13.dll", "8664 machine (x64) DLL", False, False),
    ("nvcuda.dll", "14C machine (x86) DLL", False, False),
    ("nvcuda.dll", "8664 machine (x64) DLL", True, False),
])
def test_windows_audit_never_loads_native_code(tmp_path, monkeypatch, dependency, headers, missing, passes):
    library = tmp_path / "fixture.dll"
    library.write_bytes(b"never loaded")

    def capture(command, cwd):
        if command[1] == "/headers":
            return headers
        if command[1] == "/dependents":
            return f"    KERNEL32.dll\n    {dependency}\n"
        return "" if missing else "\n".join(
            f"   {i} A 00001000 {name}" for i, name in enumerate(builder.REQUIRED_EXPORTS))

    monkeypatch.setattr(builder, "capture", capture)
    if passes:
        result = builder.audit_windows(library, tmp_path, Path("dumpbin.exe"))
        assert result["sha256"] == builder.sha256(library)
        assert result["platform"] == "win_amd64"
    else:
        with pytest.raises(ValueError):
            builder.audit_windows(library, tmp_path, Path("dumpbin.exe"))


@pytest.mark.parametrize("version", ["0.1.0", "1.0", "not-a-version"])
def test_no_silent_stable_release(version):
    with pytest.raises(ValueError):
        builder.staged_metadata(METADATA, version)


def test_existing_pair_is_not_silently_overwritten():
    with pytest.raises(ValueError, match="allocator-free"):
        builder.staged_metadata(METADATA.replace('"cuda-core"', '"ddo-vmm-allocator==0.1.0"'), "0.1.0rc1", "0.1.0rc1")


def test_snapshot_preserves_source_bytes_and_excludes_generated_files(tmp_path):
    root, output = tmp_path / "repo", tmp_path / "stage"
    (root / "src/__pycache__").mkdir(parents=True)
    (root / "src/module.py").write_bytes(b"# unchanged\r\n")
    (root / "src/library.so").write_bytes(b"not a release input")
    (root / "src/__pycache__/module.pyc").write_bytes(b"cache")
    (root / "src/generated.egg-info").mkdir()
    (root / "src/generated.egg-info/PKG-INFO").write_text("stale metadata")
    (root / "src/object.o").write_bytes(b"not source")
    hashes = builder.snapshot(root, output, ["src"])
    assert set(hashes) == {"src/module.py"}
    assert (output / "src/module.py").read_bytes() == b"# unchanged\r\n"


def test_escaping_source_symlink_is_refused(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("private = True")
    (root / "src/escape.py").symlink_to(outside)
    with pytest.raises(ValueError, match="Escaping"):
        builder.snapshot(root, tmp_path / "stage", ["src"])


def test_elf_audit_records_requirements_without_manylinux_claim(tmp_path, monkeypatch):
    library = tmp_path / "library.so"
    library.write_bytes(b"CPU fixture, never loaded")

    def capture(command, cwd):
        if command[0] == "nm":
            return "\n".join(f"0000 T {name}" for name in builder.REQUIRED_EXPORTS)
        if command[:2] == ["readelf", "-d"]:
            return "(NEEDED) Shared library: [libcuda.so.1]\n(NEEDED) Shared library: [libc.so.6]"
        if command[0] == "readelf":
            return "GLIBC_2.38 GLIBCXX_3.4.31 CXXABI_1.3.13"
        return "libcuda.so.1 => /system/libcuda.so.1"

    monkeypatch.setattr(builder, "capture", capture)
    result = builder.audit_linux(library, tmp_path)
    assert result["manylinux_validated"] is False
    assert "GLIBC_2.38" in result["required_symbol_versions"]
    assert result["sha256"] == builder.sha256(library)
    json.dumps(result)


@pytest.mark.parametrize("dynamic", ["(RUNPATH) Library runpath: [/build/private]",
                                     "(NEEDED) Shared library: [libunexpected.so]"])
def test_path_dependent_or_unreviewed_elf_is_refused(tmp_path, monkeypatch, dynamic):
    monkeypatch.setattr(builder, "capture", lambda *args: dynamic)
    with pytest.raises(ValueError):
        builder.audit_linux(tmp_path / "unused.so", tmp_path)


@pytest.mark.parametrize("missing,stub,versions,passes", [
    ("libcuda.so.1", True, "GLIBC_2.35 GLIBCXX_3.4.29 CXXABI_1.3.13", True),
    ("libcuda.so.1", False, "GLIBC_2.35", False),
    ("libstdc++.so.6", True, "GLIBC_2.35", False),
    ("libcuda.so.1", True, "GLIBC_2.38", False),
    ("libcuda.so.1", True, "GLIBCXX_3.4.31", False),
    ("libcuda.so.1", True, "CXXABI_1.3.15", False),
])
def test_container_audit_keeps_non_driver_dependencies_and_abi_bounds(
        tmp_path, monkeypatch, missing, stub, versions, passes):
    library = tmp_path / "fixture.so"
    library.write_bytes(b"never loaded")

    def capture(command, cwd):
        if command[0] == "nm":
            return "\n".join(f"0000 T {name}" for name in builder.REQUIRED_EXPORTS)
        if command[:2] == ["readelf", "-d"]:
            return "(NEEDED) Shared library: [libcuda.so.1]\n(NEEDED) Shared library: [libc.so.6]"
        if command[0] == "readelf":
            return versions
        return f"{missing} => not found\nlibc.so.6 => /system/libc.so.6"

    monkeypatch.setattr(builder, "capture", capture)
    if not passes:
        with pytest.raises(ValueError):
            builder.audit_linux(library, tmp_path, driver_stub=stub, abi_baseline="ubuntu22.04")
    else:
        result = builder.audit_linux(library, tmp_path, driver_stub=stub, abi_baseline="ubuntu22.04")
        assert result["unresolved_external_driver"] is True
        assert result["manylinux_validated"] is False
