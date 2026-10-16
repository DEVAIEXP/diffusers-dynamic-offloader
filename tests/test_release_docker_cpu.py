"""Docker orchestration contracts; no daemon, container, native load or CUDA."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "release_docker", Path(__file__).parents[1] / "scripts/build_release_docker.py")
docker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docker)


def test_bounded_gpu_free_readonly_source_and_separate_output(tmp_path):
    root = tmp_path / "repo with spaces"
    output = tmp_path / "candidates/rc2/linux"
    build, run = docker.commands(root, "0.1.0rc2", output, "builder:rc2", 2, True)
    assert build[-1] == str(root / "tools/release")
    assert f"type=bind,source={root},target=/src,readonly" in run
    assert f"type=bind,source={output.parent},target=/out" in run
    assert run[run.index("--network") + 1] == "none"
    assert run[run.index("--memory") + 1] == "6g"
    assert "--gpus" not in run and "--privileged" not in run
    assert "--allow-dirty" in run
    assert "--cuda-driver-stub" in run and "ubuntu22.04" in run


@pytest.mark.parametrize("version,jobs", [("0.1.0", 2), ("main", 2), ("0.1.0rc2", 0), ("0.1.0rc2", 9)])
def test_no_implicit_stable_version_or_unbounded_jobs(tmp_path, version, jobs):
    with pytest.raises(ValueError):
        docker.commands(tmp_path, version, tmp_path / "candidate", "builder:rc2", jobs, False)


def test_dry_run_never_calls_docker_or_creates_output(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not execute anything")
    monkeypatch.setattr(docker.subprocess, "run", forbidden)
    monkeypatch.setattr(docker.subprocess, "check_output", forbidden)
    output = tmp_path / "new"
    assert docker.main(["--version", "0.1.0rc2", "--output", str(output), "--dry-run"]) == 0
    assert not output.exists()


def test_existing_candidate_is_never_overwritten(tmp_path):
    with pytest.raises(SystemExit):
        docker.main(["--version", "0.1.0rc2", "--output", str(tmp_path), "--dry-run"])


def test_recipe_context_is_allowlisted_and_base_digest_pinned():
    root = Path(__file__).parents[1]
    recipe = (root / "tools/release/Dockerfile").read_text()
    assert "ubuntu22.04@sha256:689268c255aad87459d7a8f561a4e264632c43aee333b3176e1ea2fb01665694" in recipe
    ignore = (root / "tools/release/.dockerignore").read_text().splitlines()
    assert ignore == ["*", "!Dockerfile", "!pyproject.toml", "!uv.lock"]


def test_successful_container_build_retains_image_identity_in_dist(tmp_path, monkeypatch):
    output = tmp_path / "candidate"

    def capture(command, **kwargs):
        return '{"OSType":"linux","Architecture":"x86_64"}' if command[1] == "info" else "sha256:fixture-image\n"

    def run(command, **kwargs):
        assert kwargs["check"] is True
        if command[1] == "run":
            (output / "dist").mkdir(parents=True)

    monkeypatch.setattr(docker.subprocess, "check_output", capture)
    monkeypatch.setattr(docker.subprocess, "run", run)
    assert docker.main(["--version", "0.1.0rc3", "--output", str(output)]) == 0
    assert (output / "dist/docker-build.json").read_bytes() == (tmp_path / "candidate-docker-build.json").read_bytes()
