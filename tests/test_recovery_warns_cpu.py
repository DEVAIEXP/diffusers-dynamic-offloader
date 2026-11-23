"""Public recovery-notification settings and early narrow logging filters."""

import logging

import pytest


def test_recovery_flag_defaults_and_explicit_boolean():
    from diffusers_dynamic_offloader import DynamicOffloadConfig

    assert DynamicOffloadConfig().show_recovery_warns is False
    assert DynamicOffloadConfig(show_recovery_warns=True).show_recovery_warns is True
    with pytest.raises(ValueError, match="boolean"):
        DynamicOffloadConfig(show_recovery_warns="false")


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True)])
def test_recovery_flag_environment_and_public_metrics(value, expected):
    from diffusers_dynamic_offloader import DynamicOffloadSettings

    settings = DynamicOffloadSettings.from_env(execution_device="cpu", running_on_wsl=False,
                                              environ={"DDO_PRESET": "one_shot_fast",
                                                       "DDO_SHOW_RECOVERY_WARNS": value})
    assert settings.config.show_recovery_warns is expected
    assert settings.as_metrics()["dynamic_offload_show_recovery_warns"] is expected


def test_bootstrap_filter_is_narrow_and_keeps_errors():
    from ddo_vmm_bootstrap import _TorchImportNoticeFilter, _BOOTSTRAP

    quiet = _TorchImportNoticeFilter()
    for name in ("KernelPreference", "ScaleCalculationMode"):
        message = (f"<enum '{name}'> is an Enum subclass and is now natively supported by torch.compile "
                   "as an opaque value type. Calling register_constant() on Enum subclasses is deprecated "
                   "and will be an error in a future release.")
        assert not quiet.filter(logging.LogRecord("torch.utils._pytree", logging.WARNING, "", 0, message, (), None))
        assert quiet.filter(logging.LogRecord("torch.utils._pytree", logging.ERROR, "", 0, message, (), None))
    assert quiet.filter(logging.LogRecord("torch.utils._pytree", logging.WARNING, "", 0,
                                         "A different warning", (), None))
    assert _BOOTSTRAP.index("configure_bootstrap_logging()") < _BOOTSTRAP.index("from ddo_vmm_allocator")
