"""Bounded, opt-in operator tracing; never an allocation or offload policy."""
from dataclasses import replace
from pathlib import Path
from threading import Lock
from time import perf_counter


_owner = Lock()


class DynamicOffloadOperatorDiagnostics:
    """Profile selected invocations of one module and label its direct children.

    Profiling may synchronize at start/stop and perturbs submission cadence.
    It cannot establish uninstrumented performance or prove driver paging.
    No input shapes, stacks, memory profiles or model tensors are retained.
    Single-threaded, nonrecursive inference only; no concurrent profiler.
    """

    def __init__(self, *, enabled=False, module=None, module_path=None,
                 invocations=(0, 2), output_dir=None, use_cuda=True):
        self.enabled, self.module, self.module_path = bool(enabled), module, module_path
        self.invocations, self.output_dir = tuple(invocations), output_dir
        self.use_cuda = bool(use_cuda)
        self._used = self._finished = self._failed = False
        self._hooks, self._samples = [], []
        self._profiler = None
        self._count = 0
        self._root_active = False
        self._ranges = {}
        self._owns = False

    def __enter__(self):
        if self._used:
            raise RuntimeError('Operator diagnostic scopes cannot be reused')
        self._used = True
        if not self.enabled:
            return self
        if (not self.module_path or not self.output_dir or not self.invocations
                or len(self.invocations) > 4
                or any(type(i) is not int or not 0 <= i <= 31 for i in self.invocations)
                or len(set(self.invocations)) != len(self.invocations)):
            raise ValueError('Specify one module path, output directory and 1..4 unique invocation indices in 0..31')
        target = self.module.get_submodule(self.module_path)
        if not _owner.acquire(blocking=False):
            raise RuntimeError('Operator diagnostics are already owned')
        self._owns = True
        try:
            import torch
            self._torch = torch
            self._activities = [torch.profiler.ProfilerActivity.CPU]
            if self.use_cuda:
                if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                    raise RuntimeError('CUDA operator tracing unavailable; refusing legacy event-timing fallback')
                self._activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._directory = Path(self.output_dir).resolve()
            self._directory.mkdir(parents=True, exist_ok=True)
            for index in self.invocations:
                if (self._directory / f'operators_invocation_{index:02d}.json').exists():
                    raise FileExistsError('Operator trace directory already contains a requested capture')
            self._hooks.append(target.register_forward_pre_hook(self._begin))
            self._hooks.append(target.register_forward_hook(self._end, always_call=True))
            for name, child in target.named_children():
                self._hooks.append(child.register_forward_pre_hook(self._child_begin(name)))
                self._hooks.append(child.register_forward_hook(self._child_end(name), always_call=True))
        except BaseException:
            self._remove()
            raise
        return self

    def _child_begin(self, name):
        def begin(module, args):
            if self._profiler is not None:
                scope = self._torch.profiler.record_function('ddo.forward.' + name)
                scope.__enter__()
                self._ranges.setdefault(name, []).append(scope)
        return begin

    def _child_end(self, name):
        def end(module, args, output):
            stack = self._ranges.get(name)
            if stack:
                stack.pop().__exit__(None, None, None)
        return end

    def _begin(self, module, args):
        if self._root_active:
            raise RuntimeError('Recursive/concurrent selected forwards are unsupported')
        self._root_active = True
        index, self._count = self._count, self._count + 1
        if index not in self.invocations:
            return
        self._index = index
        profiler = self._torch.profiler.profile(activities=self._activities,
            record_shapes=False, profile_memory=False, with_stack=False)
        profiler.__enter__()
        self._profiler = profiler
        self._start = perf_counter()

    def _end(self, module, args, output):
        self._root_active = False
        if self._profiler is None:
            return
        host_forward = perf_counter() - self._start
        profiler, self._profiler = self._profiler, None
        start = perf_counter()
        profiler.__exit__(None, None, None)
        stop_seconds = perf_counter() - start
        path = self._directory / f'operators_invocation_{self._index:02d}.json'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite operator trace: {path}')
        profiler.export_chrome_trace(str(path))
        rows = [{'name': e.key, 'calls': e.count,
                 'self_host_seconds': e.self_cpu_time_total / 1e6,
                 'host_seconds': e.cpu_time_total / 1e6,
                 'self_device_seconds': e.self_device_time_total / 1e6,
                 'device_seconds': e.device_time_total / 1e6}
                for e in profiler.key_averages()]
        device_events = sum(str(e.device_type).endswith('CUDA') for e in profiler.events())
        self._samples.append({'invocation_index': self._index,
            'host_forward_seconds': host_forward, 'profiler_stop_seconds': stop_seconds,
            'trace_path': str(path), 'device_activity_events': device_events,
            'operator_count': len(rows),
            'top_host': sorted(rows, key=lambda e: e['self_host_seconds'], reverse=True)[:40],
            'top_device': sorted(rows, key=lambda e: e['self_device_seconds'], reverse=True)[:40],
            'child_ranges': [e for e in rows if e['name'].startswith('ddo.forward.')]})
        if self.use_cuda and not device_events:
            raise RuntimeError('Operator capture has no CUDA activity; stop rather than continue an unobservable diagnostic')

    def _remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.module = None  # A completed diagnostic must not retain the model.
        if self._owns:
            _owner.release()
            self._owns = False

    def __exit__(self, exc_type, exc_value, traceback):
        self._failed = exc_type is not None
        try:
            if self._profiler is not None:
                profiler, self._profiler = self._profiler, None
                profiler.__exit__(exc_type, exc_value, traceback)
        except BaseException as error:
            self._failed = True
            if exc_value is None:
                raise
            if hasattr(exc_value, 'add_note'):
                exc_value.add_note('Operator profiler cleanup also failed: ' + repr(error))
        finally:
            self._remove()
            self._finished = True

    def snapshot(self):
        status = 'disabled' if not self.enabled else 'not_finished' if not self._finished else (
            'failed' if self._failed else 'incomplete' if len(self._samples) != len(self.invocations)
            else 'cuda_activity_unavailable' if self.use_cuda and any(
                not s['device_activity_events'] for s in self._samples) else 'complete')
        return {'schema_version': 1, 'status': status, 'benchmark_eligible': False,
                'scope': 'selected_forward_operators', 'may_synchronize': True,
                'use_cuda': self.use_cuda, 'module_path': self.module_path,
                'requested_invocations': list(self.invocations), 'observed_invocations': self._count,
                'samples': list(self._samples)}

    def attach(self, report):
        return replace(report, runtime={**report.runtime, 'operator_diagnostics': self.snapshot()})
