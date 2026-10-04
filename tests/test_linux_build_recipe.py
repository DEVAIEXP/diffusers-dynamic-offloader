"""Build recipe contracts only: no compiler, CUDA or Linux process is launched."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ddo_linux_build_recipe", ROOT / "ddo-vmm-allocator" / "scripts" / "build_linux.py"
)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def test_cpu_recipe_is_isolated_from_cuda_and_windows_builds():
    commands = recipe.build_commands(ROOT / "ddo-vmm-allocator", "cpu", 2)
    assert len(commands) == 3
    assert commands[0][2].endswith(str(Path("native/tests/cpu")))
    assert commands[0][4].endswith("build-linux-cpu")
    assert commands[-1][0] == "ctest"
    assert "--no-tests=error" in commands[-1]
    assert all("cuda" not in part.lower() for command in commands for part in command)


def test_library_recipe_preserves_explicit_feature_selection_and_never_runs_probes():
    commands = recipe.build_commands(ROOT / "ddo-vmm-allocator", "library", 2)
    assert len(commands) == 2
    assert commands[0][4].endswith("build-linux")
    assert "-DDDO_VMM_BUILD_PROBES=OFF" in commands[0]
    assert "-DDDO_VMM_NATIVE_SPAN_POOL=ON" in commands[0]
    assert "-DDDO_VMM_NATIVE_SOURCE_DISPATCHER=OFF" in commands[0]
    assert "-DDDO_VMM_SOURCE_TEST_HOOKS=OFF" in commands[0]
    # CMake consumes the first -D. The variable itself begins with DDO, so
    # two D's after '-' would silently configure unused DO_* variables.
    assert all(part[2:].startswith("DDO_") for part in commands[0] if part.startswith("-D") and "VMM" in part)


@pytest.mark.parametrize("jobs", [0, -1, 65])
def test_invalid_parallelism_rejected(jobs):
    with pytest.raises(ValueError):
        recipe.build_commands(ROOT, "cpu", jobs)


def test_dry_run_has_no_subprocesses(capsys):
    with patch.object(recipe.subprocess, "run") as run:
        assert recipe.main(["--mode", "library", "--dry-run"]) == 0
    run.assert_not_called()
    assert "DDO_VMM_NATIVE_SPAN_POOL=ON" in capsys.readouterr().out


def test_windows_execution_refused_before_running_commands():
    with (
        patch.object(recipe.sys, "platform", "win32"),
        patch.object(recipe.subprocess, "run") as run,
        pytest.raises(SystemExit),
    ):
        recipe.main([])
    run.assert_not_called()


def test_first_failed_cpu_gate_stops_sequence():
    with patch.object(recipe.sys, "platform", "linux"), patch.object(recipe.subprocess, "run") as run:
        run.return_value.returncode = 7
        assert recipe.main([]) == 7
    assert run.call_count == 1
