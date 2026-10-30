"""Static release-workflow safety contracts; no Actions, network or CUDA execution."""
from pathlib import Path
import re


WORKFLOW = Path(__file__).parents[1] / ".github/workflows/release-candidates.yml"


def test_candidate_workflow_is_manual_read_only_and_not_publication():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    for forbidden in ("pull_request:", "pull_request_target:", "push:", "contents: write",
                      "id-token: write", "gh release", "twine", "pypa/gh-action-pypi-publish",
                      "--allow-dirty", "--gpus", "vmm_idle_trim_probe.py"):
        assert forbidden not in text
    assert "contents: read" in text
    assert "persist-credentials: false" in text
    assert "CUDA_VISIBLE_DEVICES: '-1'" in text


def test_actions_are_hash_pinned_and_upload_is_distribution_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    actions = re.findall(r"uses: ([^\s]+)", text)
    assert len(actions) == 2
    assert all(re.fullmatch(r"actions/[\w-]+@[0-9a-f]{40}", item) for item in actions)
    assert "path: ${{ env.CANDIDATE_OUTPUT }}/dist/*" in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 14" in text


def test_version_uses_environment_not_shell_interpolation_and_profiles_are_shared():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "CANDIDATE_VERSION: ${{ inputs.version }}" in text
    assert "--version $env:CANDIDATE_VERSION" in text
    assert "--version ${{ inputs.version }}" not in text
    assert "scripts/build_release.py" in text
    assert "scripts/build_release_docker.py" in text
    assert "runs-on: [self-hosted, Windows, X64, ddo-release]" in text
    assert "--jobs 2" in text
