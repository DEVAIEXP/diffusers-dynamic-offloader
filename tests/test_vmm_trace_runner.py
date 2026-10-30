"""CPU-only launch/lifecycle contracts for the optional diagnostic wrapper."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def wrapper():
    path = Path(__file__).parents[1] / "scripts" / "trace_vmm_runner.py"
    spec = importlib.util.spec_from_file_location("trace_vmm_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(wrapper, tmp_path):
    target = tmp_path / "runner.py"
    target.write_text("# mock target", encoding="utf-8")
    return wrapper.parse_args([
        "--trace-output", str(tmp_path / "trace.json"), str(target),
        "--ddo-backend", "vmm", "--ddo-preset", "auto", "--steps", "8",
    ])


def test_trace_bootstrap_precedes_profiler_and_preserves_arguments(wrapper, tmp_path, monkeypatch):
    args = arguments(wrapper, tmp_path)
    trace = []
    original_argv, original_path = sys.argv, list(sys.path)
    monkeypatch.setattr(wrapper, "ensure_vmm_fast_process", lambda: trace.append("bootstrap"))

    class Profiler:
        def __enter__(self):
            trace.append("profiler_start")
            return self
        def __exit__(self, *exc):
            trace.append("profiler_stop")
        def export_chrome_trace(self, path):
            trace.append(("export", path))

    def profile(**kwargs):
        assert trace == ["bootstrap"]
        assert not kwargs["record_shapes"] and not kwargs["profile_memory"] and not kwargs["with_stack"]
        assert kwargs["activities"] == ["cpu", "cuda"]
        return Profiler()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(profiler=SimpleNamespace(
        kineto_available=lambda: True, profile=profile,
        ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
    )))

    def target(path, run_name):
        assert sys.argv == [str(args.target.resolve()), *args.target_args]
        assert sys.path[0] == str(args.target.parent)
        assert run_name == "__main__"
        trace.append("target")

    monkeypatch.setattr(wrapper.runpy, "run_path", target)
    wrapper.run(args)
    assert trace[:4] == ["bootstrap", "profiler_start", "target", "profiler_stop"]
    assert trace[-1] == ("export", str(args.trace_output.resolve()))
    assert sys.argv is original_argv and sys.path == original_path


def test_existing_trace_rejected_before_bootstrap(wrapper, tmp_path, monkeypatch):
    args = arguments(wrapper, tmp_path)
    args.trace_output.write_text("preserved", encoding="utf-8")
    monkeypatch.setattr(wrapper, "ensure_vmm_fast_process", lambda: pytest.fail("must not launch"))
    with pytest.raises(FileExistsError):
        wrapper.run(args)
    assert args.trace_output.read_text() == "preserved"


@pytest.mark.parametrize("backend,preset", [("linear", "auto"), ("vmm", "nvme_baseline")])
def test_protected_routes_are_not_profiled_by_native_wrapper(wrapper, tmp_path, monkeypatch, backend, preset):
    args = arguments(wrapper, tmp_path)
    args.target_args = ["--ddo-backend", backend, "--ddo-preset", preset]
    monkeypatch.setattr(wrapper, "ensure_vmm_fast_process", lambda: pytest.fail("must not launch"))
    with pytest.raises(ValueError):
        wrapper.run(args)


def test_no_kineto_fails_without_running_target(wrapper, tmp_path, monkeypatch):
    args = arguments(wrapper, tmp_path)
    monkeypatch.setattr(wrapper, "ensure_vmm_fast_process", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(profiler=SimpleNamespace(kineto_available=lambda: False)))
    monkeypatch.setattr(wrapper.runpy, "run_path", lambda *a, **k: pytest.fail("must not execute target"))
    with pytest.raises(RuntimeError, match="Kineto"):
        wrapper.run(args)


def test_target_failure_restores_process_state_without_export(wrapper, tmp_path, monkeypatch):
    args = arguments(wrapper, tmp_path)
    original_argv, original_path = sys.argv, list(sys.path)
    monkeypatch.setattr(wrapper, "ensure_vmm_fast_process", lambda: None)

    class Profiler:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def export_chrome_trace(self, path): pytest.fail("failed target must not be labelled successful")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(profiler=SimpleNamespace(
        kineto_available=lambda: True, profile=lambda **kwargs: Profiler(),
        ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
    )))
    def fail(*args, **kwargs):
        raise RuntimeError("target failed")
    monkeypatch.setattr(wrapper.runpy, "run_path", fail)
    with pytest.raises(RuntimeError, match="target failed"):
        wrapper.run(args)
    assert sys.argv is original_argv and sys.path == original_path
    assert not args.trace_output.exists()
