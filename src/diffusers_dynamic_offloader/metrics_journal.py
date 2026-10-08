"""Opt-in durable partial reports without CUDA timing/completion collection."""
import json
import os
import time
import uuid
from pathlib import Path


class DynamicOffloadMetricsJournal:
    """Persist public snapshots; step boundaries do not prove GPU completion.

    Each record is immutable and separately published. Killing the process may
    leave a temporary file, but cannot corrupt earlier published records. Native
    statistics may contain pending work; the journal never collects or waits.
    """

    def __init__(self, result, directory, *, enabled=False):
        self.enabled = enabled
        self._result = result if enabled else None
        self._directory = Path(directory) if enabled else None
        self._started = time.perf_counter()
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Metrics journal is closed")
        return self

    def __exit__(self, *exc):
        self._result = None  # Do not retain an unloaded model through diagnostics.
        self._closed = True

    def record(self, step_index=None):
        """Return the published path, or None when disabled; never replace it."""
        if self._closed:
            raise RuntimeError("Metrics journal is closed")
        if not self.enabled:
            return None
        if step_index is not None and (type(step_index) is not int or step_index < 0):
            raise ValueError("Step index must be a non-negative integer or None")
        name = 'before_denoise.json' if step_index is None else f'step_{step_index:04d}.json'
        target = self._directory / name
        if target.exists():
            raise FileExistsError(target)
        report = self._result.metrics_snapshot(collect_completed=False)
        payload = {
            'schema_version': 1, 'observation': 'partial_host_step_boundary',
            'step_index': step_index, 'host_elapsed_seconds': time.perf_counter() - self._started,
            'gpu_completion_proven': False, 'completion_collection_requested': False,
            'metrics': report.as_dict(),
        }
        serialized = json.dumps(payload, indent=2, allow_nan=False)
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = self._directory / (name + '.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows rename refuses existing destinations. POSIX hard-link creation
        # supplies the same no-replacement, atomic publication guarantee.
        if os.name == 'nt':
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        return target
