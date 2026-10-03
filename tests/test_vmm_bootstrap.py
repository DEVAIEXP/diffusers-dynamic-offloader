from unittest.mock import patch

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
