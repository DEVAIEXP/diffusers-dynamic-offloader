from unittest.mock import patch
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ddo_vmm_bootstrap import ensure_vmm_fast_process, vmm_fast_requested


def test_public_backend_arguments_select_only_vmm_fast():
    assert vmm_fast_requested([], default_backend="vmm")
    assert vmm_fast_requested(["--ddo-backend", "vmm", "--ddo-preset", "auto"])
    assert not vmm_fast_requested(["--ddo-backend=vmm", "--ddo-preset=nvme_baseline"])
    assert not vmm_fast_requested(["--ddo-backend", "linear", "--ddo-preset", "auto"])


def test_child_process_is_a_noop(tmp_path):
    ensure_vmm_fast_process(
        argv=[str(tmp_path / "runner.py")],
        environ={"DDO_VMM_FAST_CHILD": "1"},
    )


def test_parent_relaunches_script_with_allocator_first(tmp_path):
    runner = tmp_path / "runner.py"
    library = tmp_path / "ddo_vmm.dll"
    runner.touch()
    library.touch()
    with (
        patch("ddo_vmm_bootstrap._default_allocator_root", return_value=tmp_path),
        patch("ddo_vmm_bootstrap.subprocess.run") as run,
        patch("ddo_vmm_bootstrap.sys.modules", {}),
    ):
        run.return_value.returncode = 0
        with pytest.raises(SystemExit, match="0"):
            ensure_vmm_fast_process(
                library_path=library,
                allocator_cache_mib=512,
                argv=[str(runner), "--preset", "auto"],
                environ={},
            )

    command = run.call_args.args[0]
    environment = run.call_args.kwargs["env"]
    assert command[-3:] == [str(runner.resolve()), "--preset", "auto"]
    assert environment["DDO_VMM_FAST_CHILD"] == "1"
    assert environment["DDO_VMM_LIBRARY_PATH"] == str(library.resolve())
    assert environment["DDO_VMM_TORCH_CACHE_MIB"] == "512"
    assert environment["PYTHONPATH"].startswith(str(tmp_path / "src"))


def test_parent_rejects_negative_cache(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        ensure_vmm_fast_process(argv=[str(tmp_path / "runner.py")], allocator_cache_mib=-1, environ={})


@pytest.mark.parametrize("platform,relative", [
    ("win32", "build/Release/ddo_vmm.dll"),
    ("linux", "build-linux/libddo_vmm.so"),
    ("linux", "build/libddo_vmm.so"),
])
def test_platform_discovery_relaunches_only_with_matching_library(tmp_path, platform, relative):
    runner = tmp_path / "runner.py"
    runner.touch()
    library = tmp_path / relative
    library.parent.mkdir(parents=True)
    library.touch()
    with patch("ddo_vmm_bootstrap._default_allocator_root", return_value=tmp_path), \
            patch("ddo_vmm_bootstrap.sys.platform", platform), \
            patch("ddo_vmm_bootstrap.sys.modules", {}), \
            patch("ddo_vmm_bootstrap.subprocess.run") as run:
        run.return_value.returncode = 0
        with pytest.raises(SystemExit, match="0"):
            ensure_vmm_fast_process(argv=[str(runner)], environ={})
    assert run.call_args.kwargs["env"]["DDO_VMM_LIBRARY_PATH"] == str(library.resolve())


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_non_windows_discovery_never_loads_the_windows_dll(tmp_path, platform):
    runner = tmp_path / "runner.py"
    runner.touch()
    dll = tmp_path / "build" / "Release" / "ddo_vmm.dll"
    dll.parent.mkdir(parents=True)
    dll.touch()
    with patch("ddo_vmm_bootstrap._default_allocator_root", return_value=tmp_path), \
            patch("ddo_vmm_bootstrap.sys.platform", platform), \
            patch("ddo_vmm_bootstrap.sys.modules", {}), \
            patch("ddo_vmm_bootstrap.subprocess.run") as run:
        with pytest.raises(RuntimeError, match="requires DDO_VMM_LIBRARY_PATH"):
            ensure_vmm_fast_process(argv=[str(runner)], environ={})
    run.assert_not_called()


@pytest.mark.parametrize("platform,relative", [
    ("win32", "native/win32/ddo_vmm.dll"),
    ("linux", "native/linux/libddo_vmm.so"),
])
def test_installed_package_auto_discovers_binary_without_checkout(tmp_path, platform, relative):
    runner = tmp_path / "runner.py"
    runner.touch()
    package = tmp_path / "ddo_vmm_allocator"
    library = package / relative
    library.parent.mkdir(parents=True)
    library.touch()
    with (
        patch("ddo_vmm_bootstrap._default_allocator_root", return_value=None),
        patch("ddo_vmm_bootstrap._installed_allocator_root", return_value=package),
        patch("ddo_vmm_bootstrap.sys.platform", platform),
        patch("ddo_vmm_bootstrap.sys.modules", {}),
        patch("ddo_vmm_bootstrap.subprocess.run") as run,
    ):
        run.return_value.returncode = 7
        with pytest.raises(SystemExit, match="7"):
            ensure_vmm_fast_process(argv=[str(runner), "--ddo-backend", "vmm"],
                                    environ={"PYTHONPATH": "existing-host-path"})
    environment = run.call_args.kwargs["env"]
    assert environment["DDO_VMM_LIBRARY_PATH"] == str(library.resolve())
    assert environment["PYTHONPATH"] == "existing-host-path"
    assert run.call_args.args[0][-3:] == [str(runner), "--ddo-backend", "vmm"]


def test_installed_package_supports_explicit_diagnostic_library(tmp_path):
    runner = tmp_path / "runner.py"
    library = tmp_path / "diagnostic.dll"
    runner.touch()
    library.touch()
    with (
        patch("ddo_vmm_bootstrap._default_allocator_root", return_value=None),
        patch("ddo_vmm_bootstrap._installed_allocator_root", return_value=tmp_path),
        patch("ddo_vmm_bootstrap.sys.modules", {}),
        patch("ddo_vmm_bootstrap.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        with pytest.raises(SystemExit, match="0"):
            ensure_vmm_fast_process(argv=[str(runner)], library_path=library, environ={})
    assert "PYTHONPATH" not in run.call_args.kwargs["env"]


def test_package_discovery_uses_metadata_without_importing_allocator(tmp_path):
    import ddo_vmm_bootstrap as bootstrap

    spec = SimpleNamespace(origin=str(tmp_path / "__init__.py"), submodule_search_locations=[str(tmp_path)])
    with patch.object(bootstrap, "find_spec", return_value=spec) as lookup:
        assert bootstrap._installed_allocator_root() == tmp_path.resolve()
    lookup.assert_called_once_with("ddo_vmm_allocator")


@pytest.mark.parametrize("spec", [None, SimpleNamespace(origin=None, submodule_search_locations=[])])
def test_missing_or_nonfilesystem_package_is_unavailable(spec):
    import ddo_vmm_bootstrap as bootstrap

    with patch.object(bootstrap, "find_spec", return_value=spec):
        assert bootstrap._installed_allocator_root() is None


def test_discovery_preserves_checkout_precedence_and_never_scans_experimental_builds(tmp_path):
    import ddo_vmm_bootstrap as bootstrap

    checkout = tmp_path / "checkout"
    package = tmp_path / "package"
    standard = checkout / "build/Release/ddo_vmm.dll"
    experimental = checkout / "build-experimental/Release/ddo_vmm.dll"
    packaged = package / "native/win32/ddo_vmm.dll"
    for library in (standard, experimental, packaged):
        library.parent.mkdir(parents=True)
        library.touch()
    with patch.object(bootstrap.sys, "platform", "win32"):
        assert bootstrap._discover_library(checkout, package) == standard
        standard.unlink()
        assert bootstrap._discover_library(checkout, package) == packaged
        packaged.unlink()
        assert bootstrap._discover_library(checkout, package) is None


def test_explicit_library_precedes_environment_and_discovery(tmp_path):
    runner = tmp_path / "runner.py"
    explicit = tmp_path / "explicit.dll"
    runner.touch()
    explicit.touch()
    with (
        patch("ddo_vmm_bootstrap._default_allocator_root", return_value=tmp_path),
        patch("ddo_vmm_bootstrap.sys.modules", {}),
        patch("ddo_vmm_bootstrap._discover_library") as discover,
        patch("ddo_vmm_bootstrap.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        with pytest.raises(SystemExit, match="0"):
            ensure_vmm_fast_process(argv=[str(runner)], library_path=explicit,
                                    environ={"DDO_VMM_LIBRARY_PATH": "missing-environment.dll"})
    discover.assert_not_called()
    assert run.call_args.kwargs["env"]["DDO_VMM_LIBRARY_PATH"] == str(explicit.resolve())


def test_invalid_override_does_not_silently_select_another_binary(tmp_path):
    runner = tmp_path / "runner.py"
    runner.touch()
    with (
        patch("ddo_vmm_bootstrap._default_allocator_root", return_value=tmp_path),
        patch("ddo_vmm_bootstrap._discover_library") as discover,
        patch("ddo_vmm_bootstrap.sys.modules", {}),
        patch("ddo_vmm_bootstrap.subprocess.run") as run,
    ):
        with pytest.raises(FileNotFoundError):
            ensure_vmm_fast_process(argv=[str(runner)],
                                    environ={"DDO_VMM_LIBRARY_PATH": str(tmp_path / "absent.dll")})
    discover.assert_not_called()
    run.assert_not_called()


def test_no_adapter_refuses_before_launch_even_with_binary(tmp_path):
    runner = tmp_path / "runner.py"
    library = tmp_path / "ddo_vmm.dll"
    runner.touch()
    library.touch()
    with (
        patch("ddo_vmm_bootstrap._default_allocator_root", return_value=None),
        patch("ddo_vmm_bootstrap._installed_allocator_root", return_value=None),
        patch("ddo_vmm_bootstrap.sys.modules", {}),
        patch("ddo_vmm_bootstrap.subprocess.run") as run,
    ):
        with pytest.raises(RuntimeError, match="submodule or installed package"):
            ensure_vmm_fast_process(argv=[str(runner)], library_path=library, environ={})
    run.assert_not_called()


def test_real_package_metadata_discovery_never_executes_package_or_torch(tmp_path):
    package = tmp_path / "ddo_vmm_allocator"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('allocator imported in parent')\n")
    environment = dict(os.environ)
    source = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(source)))
    code = (
        "import sys; from pathlib import Path; import ddo_vmm_bootstrap as b; "
        "assert b._installed_allocator_root() == Path(sys.argv[1]); "
        "assert 'ddo_vmm_allocator' not in sys.modules; "
        "assert 'torch' not in sys.modules; assert 'cuda' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code, str(package)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
