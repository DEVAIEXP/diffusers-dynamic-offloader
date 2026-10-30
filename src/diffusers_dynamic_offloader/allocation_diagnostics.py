"""Explicit native allocation diagnosis, without changing allocation policy."""
from dataclasses import replace
from time import perf_counter


class DynamicOffloadAllocationDiagnostics:
    """Process-wide native host counters sampled at CPU submission boundaries.

    Driver timers include host blocking inside the call, not GPU kernel time.
    Boundaries do not wait for CUDA: work in flight can cross snapshots. Timers
    are cumulative/delta observations, not disjoint partitions of GPU duration.
    Pool peak is a process-wide high-water mark observed on new backing creation,
    excludes model arenas and is not an absolute driver-memory peak.
    """

    def __init__(self, *, enabled=False, native=None):
        self.enabled, self._native = bool(enabled), native
        self._used = self._active = self._failed = False
        self._steps = []
        self._error = None
        self._last = self._baseline = self._final = None

    def _read(self):
        snapshot = self._native.allocation_diagnostics_snapshot()
        if not snapshot:
            raise RuntimeError("native allocation diagnostics are unavailable")
        return snapshot

    def __enter__(self):
        if self._used:
            raise RuntimeError("Allocation diagnostic scopes cannot be reused")
        self._used = True
        if self.enabled:
            if self._native is None:
                raise ValueError("Allocation diagnostics require an active native component")
            self._native.set_allocation_diagnostics(True)
            try:
                self._baseline = self._last = self._read()
                self._started = self._last_time = perf_counter()
            except BaseException:
                self._native.set_allocation_diagnostics(False)
                raise
        self._active = True
        return self

    @staticmethod
    def _delta(before, after):
        return {key: after[key] - value for key, value in before.items()
                if key.endswith(("_calls", "_host_seconds", "_backings"))}

    def mark_step(self):
        if not self.enabled:
            return
        if not self._active:
            raise RuntimeError("Allocation step marker requires an active scope")
        current = self._read()
        now = perf_counter()
        self._steps.append({"step_index": len(self._steps),
                            "host_submission_seconds": now - self._last_time,
                            "driver_delta": self._delta(self._last, current),
                            "pool_observed_peak_mapped_bytes": current["pool_observed_peak_mapped_bytes"]})
        self._last, self._last_time = current, now

    def __exit__(self, exc_type, exc_value, traceback):
        self._failed, self._active = exc_type is not None, False
        if self.enabled:
            try:
                try:
                    self._final = self._read()
                    self._elapsed = perf_counter() - self._started
                finally:
                    self._native.set_allocation_diagnostics(False)
            except BaseException as error:
                self._failed, self._error = True, repr(error)
                if exc_value is None:
                    raise
                if hasattr(exc_value, "add_note"):
                    exc_value.add_note("Allocation diagnostic cleanup also failed: " + repr(error))

    def snapshot(self):
        result = {"schema_version": 1, "scope": "process_native_driver_host_calls",
                  "boundary": "cpu_submission_no_cuda_wait", "benchmark_eligible": False,
                  "failed": self._failed}
        if not self.enabled:
            return {**result, "status": "disabled"}
        if self._error is not None:
            return {**result, "status": "failed", "error": self._error}
        if self._final is None:
            return {**result, "status": "not_finished"}
        return {**result, "status": "complete", "native_schema_version": self._final["schema_version"],
                "host_scope_seconds": self._elapsed,
                "steps": list(self._steps), "total_delta": self._delta(self._baseline, self._final),
                "pool_observed_peak_mapped_bytes": self._final["pool_observed_peak_mapped_bytes"]}

    def attach(self, report):
        return replace(report, runtime={**report.runtime, "allocation_diagnostics": self.snapshot()})
