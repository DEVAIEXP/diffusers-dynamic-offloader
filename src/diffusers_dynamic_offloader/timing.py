"""Opt-in compute-stream timing; step waits are a separate diagnostic control."""

from __future__ import annotations

from dataclasses import replace


class DynamicOffloadExecutionTimer:
    """Record CUDA boundary markers; resolve only after completion is proven.

    Intervals include compute-stream dependencies and host launch gaps, not
    kernel-only work. They do not establish H2D/compute overlap. Disabled mode
    creates no events and does not inspect CUDA. Each instance measures one run.
    Explicit diagnostic_step_wait changes submission cadence and disqualifies
    the report as a performance benchmark. It is never enabled by a preset.
    """

    def __init__(self, *, enabled=False, device="cuda", native_factory=None,
                 diagnostic_step_wait=False):
        if diagnostic_step_wait and not enabled:
            raise ValueError("diagnostic step wait requires enabled GPU timing")
        self.enabled = enabled
        self.diagnostic_step_wait = bool(diagnostic_step_wait)
        self._step_wait_calls = 0
        self._step_wait_seconds = 0.0
        self.device = device
        self._events = []
        self._stream = None
        self._end = None
        self._active = False
        self._failed = False
        self._used = False
        self._native_factory = native_factory
        self._native = None
        self._detach_native = None

    def __enter__(self):
        if self._used:
            raise RuntimeError("Execution timer instances cannot be reused")
        self._used = True
        if self.enabled:
            import torch
            with torch.cuda.device(self.device):
                self._stream = torch.cuda.current_stream(self.device)
                self._event_type = torch.cuda.Event
                if self._native_factory is not None:
                    self._native, self._detach_native = self._native_factory(int(self._stream.cuda_stream))
                    self._native.mark_boundary(0, int(self._stream.cuda_stream))
                try:
                    event = self._event_type(enable_timing=True)
                    event.record(self._stream)
                    self._events.append(event)
                except Exception:
                    self._failed = True
                    if self._native is not None:
                        try:
                            self._native.seal()
                        finally:
                            self._detach_native()
                    raise
        self._active = True
        return self

    def mark_step(self):
        if not self.enabled:
            return
        if not self._active:
            raise RuntimeError("Step marker requires an active execution timer")
        if self._native is not None:
            self._native.mark_boundary(len(self._events), int(self._stream.cuda_stream))
        event = self._event_type(enable_timing=True)
        event.record(self._stream)
        self._events.append(event)

        if self.diagnostic_step_wait:
            # Explicit causal control only, never an offload/residency policy.
            # Wait for this compute boundary, not the device or upload stream.
            from time import perf_counter
            start = perf_counter()
            try:
                self._step_wait_calls += 1
                event.synchronize()
            finally:
                self._step_wait_seconds += perf_counter() - start

    def __exit__(self, exc_type, exc_value, traceback):
        self._failed = exc_type is not None
        self._active = False
        if self._native is not None:
            try:
                self._native.mark_boundary(2**64 - 1, int(self._stream.cuda_stream))
                self._native.seal()
            finally:
                self._detach_native()
        if self.enabled:
            self._end = self._event_type(enable_timing=True)
            self._end.record(self._stream)

    def snapshot(self, *, wait=False):
        """Default never waits. Optional wait drains just the final marker."""
        report = {"schema_version": 1, "clock": "cuda_event", "unit": "seconds",
                  "scope": "compute_stream_with_dependencies", "failed": self._failed}
        if self.diagnostic_step_wait:
            report["diagnostic_step_wait"] = {
                "enabled": True, "scope": "compute_boundary_event_host_wait",
                "calls": self._step_wait_calls, "host_seconds": self._step_wait_seconds,
                "benchmark_eligible": False,
            }
        if not self.enabled:
            return {**report, "status": "disabled"}
        if self._end is None:
            return {**report, "status": "not_finished"}
        if wait:
            self._end.synchronize()
        if not self._end.query():
            return {**report, "status": "pending", "step_count": len(self._events) - 1}
        steps = [a.elapsed_time(b) / 1000 for a, b in zip(self._events, self._events[1:])]
        return {**report, "status": "complete", "stream": int(self._stream.cuda_stream),
                "phase_seconds": self._events[0].elapsed_time(self._end) / 1000,
                "step_seconds": steps,
                "tail_seconds": self._events[-1].elapsed_time(self._end) / 1000,
                "steady_state_seconds_per_step": sum(steps[1:]) / len(steps[1:]) if len(steps) > 1 else None}

    def attach(self, report):
        """Combine with the public DDO report; hosts need no field knowledge."""
        runtime = {**report.runtime, "execution_timing": self.snapshot()}
        if self._native is not None:
            runtime["native_timeline"] = self._native.snapshot()
        return replace(report, runtime=runtime)
