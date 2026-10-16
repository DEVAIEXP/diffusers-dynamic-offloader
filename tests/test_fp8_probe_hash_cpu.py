"""Probe artifact pinning remains strict without importing Torch or CUDA."""
import ast
from pathlib import Path

import pytest


def validator():
    path = Path(__file__).parents[1] / "scripts/probe_fp8_public_cuda.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "validate_native_hash")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name], tree


def test_expected_hash_is_case_insensitive_but_mandatory():
    check, _ = validator()
    check("AB" * 32, "ab" * 32)
    with pytest.raises(ValueError, match="does not match"):
        check("00" * 32, "ab" * 32)


@pytest.mark.parametrize("expected", ["", "ab", "gg" * 32])
def test_invalid_expected_hash_is_refused(expected):
    check, _ = validator()
    with pytest.raises(ValueError, match="64 hexadecimal"):
        check("00" * 32, expected)


def test_windows_run_case_default_is_preserved():
    _, tree = validator()
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_case")
    assert run.args.defaults[-1].value == "EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6"
