"""Explicit CUDA context ownership for low-level probes, not runner setup."""

from contextlib import contextmanager


def _checked(result, operation):
    status, *values = result
    if status != 0:
        raise RuntimeError(f"{operation} failed: {status}")
    return values


@contextmanager
def primary_context_for_probe(driver, device=0):
    """Retain/push a primary context, then restore the thread and release it.

    The injected driver lets CPU tests verify ownership without calling CUDA.
    This is test-only; native source-cache callers must satisfy its documented
    current-context precondition, not rely on PyTorch cached allocations.
    """
    context, = _checked(driver.cuDevicePrimaryCtxRetain(device), "retain primary context")
    pushed = False
    try:
        _checked(driver.cuCtxPushCurrent(context), "push primary context")
        pushed = True
        yield
    finally:
        try:
            if pushed:
                _checked(driver.cuCtxPopCurrent(), "pop primary context")
        finally:
            _checked(driver.cuDevicePrimaryCtxRelease(device), "release primary context")


@contextmanager
def detached_context_for_probe(driver):
    """Test-only: remove the thread's current context, restore it in finally.

    No context is created, reset or destroyed. Do not run Torch operations while
    detached; only native calls expected to reject the missing owner context.
    """
    context, = _checked(driver.cuCtxGetCurrent(), "get current context")
    if not context:
        raise RuntimeError("probe requires an existing current CUDA context")
    _checked(driver.cuCtxSetCurrent(driver.CUcontext(0)), "detach current context")
    try:
        yield context
    finally:
        _checked(driver.cuCtxSetCurrent(context), "restore current context")
