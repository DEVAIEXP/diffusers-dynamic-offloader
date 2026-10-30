"""CPU-only ownership tests for the test-side CUDA ledger probe guard."""

import pytest
from vmm_test_helpers import primary_context_for_probe, detached_context_for_probe


class Driver:
    def __init__(self, fail=None):
        self.calls = []
        self.current = "previous"
        self.fail = fail

    def cuDevicePrimaryCtxRetain(self, device):
        self.calls.append("retain")
        return (1, None) if self.fail == "retain" else (0, "primary")

    def cuCtxPushCurrent(self, context):
        self.calls.append("push")
        if self.fail == "push":
            return (1,)
        self.current = context
        return (0,)

    def cuCtxPopCurrent(self):
        self.calls.append("pop")
        if self.fail == "pop":
            return (1, None)
        self.current = "previous"
        return (0, "primary")

    def cuDevicePrimaryCtxRelease(self, device):
        self.calls.append("release")
        return (0,)


def test_probe_guard_restores_context_and_balances_retain():
    driver = Driver()
    with primary_context_for_probe(driver):
        assert driver.current == "primary"
    assert driver.current == "previous"
    assert driver.calls == ["retain", "push", "pop", "release"]


def test_probe_guard_restores_context_when_probe_raises():
    driver = Driver()
    with pytest.raises(ValueError, match="probe failed"), primary_context_for_probe(driver):
        raise ValueError("probe failed")
    assert driver.current == "previous"
    assert driver.calls == ["retain", "push", "pop", "release"]


@pytest.mark.parametrize("fail,expected", [
    ("retain", ["retain"]),
    ("push", ["retain", "push", "release"]),
    ("pop", ["retain", "push", "pop", "release"]),
])
def test_probe_guard_releases_only_owned_resources_on_driver_error(fail, expected):
    driver = Driver(fail)
    with pytest.raises(RuntimeError), primary_context_for_probe(driver):
        pass
    assert driver.calls == expected


class DetachDriver:
    CUcontext = staticmethod(lambda value: value)

    def __init__(self, fail=None):
        self.current = 123
        self.calls = []
        self.fail = fail

    def cuCtxGetCurrent(self):
        self.calls.append("get")
        return (1 if self.fail == "get" else 0, self.current)

    def cuCtxSetCurrent(self, context):
        self.calls.append(context)
        if self.fail == ("detach" if context == 0 else "restore"):
            return (1,)
        self.current = context
        return (0,)


@pytest.mark.parametrize("raise_body", [False, True])
def test_detach_probe_restores_owner_even_after_body_failure(raise_body):
    driver = DetachDriver()
    def probe():
        with detached_context_for_probe(driver) as owner:
            assert owner == 123 and driver.current == 0
            if raise_body:
                raise ValueError("probe failed")
    if raise_body:
        with pytest.raises(ValueError, match="probe failed"):
            probe()
    else:
        probe()
    assert driver.current == 123
    assert driver.calls == ["get", 0, 123]


@pytest.mark.parametrize("fail,calls", [
    ("get", ["get"]), ("detach", ["get", 0]),
    ("restore", ["get", 0, 123]),
])
def test_detach_probe_reports_driver_failure(fail, calls):
    driver = DetachDriver(fail)
    with pytest.raises(RuntimeError):
        with detached_context_for_probe(driver):
            pass
    assert driver.calls == calls


def test_detach_probe_refuses_missing_initial_context():
    driver = DetachDriver()
    driver.current = 0
    with pytest.raises(RuntimeError, match="existing current"):
        with detached_context_for_probe(driver):
            pass
    assert driver.calls == ["get"]
