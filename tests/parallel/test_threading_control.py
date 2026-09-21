"""Tests for evodevo.parallel.threading_control.

Deliberately never imports numpy/torch itself at module level, and controls
"is numpy/torch already imported" via monkeypatching sys.modules directly
rather than relying on the real, invocation-order-dependent state of the
test process (which already has numpy imported by the time this file runs
as part of the full suite, since earlier-collected test modules import it)
-- so these tests behave identically whether run alone or as part of the
full suite.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

from evodevo.parallel.threading_control import (
    THREAD_ENV_VARS,
    ThreadingConfiguredTooLateError,
    ThreadingReport,
    configure_threading,
    describe_threading_environment,
)

_DYNAMIC_VARS = {"MKL_DYNAMIC", "OMP_DYNAMIC"}


def _clear_thread_env(monkeypatch):
    for var in THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _hide_from_sys_modules(monkeypatch, *names):
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)


def _force_present_in_sys_modules(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))


def test_configure_threading_sets_every_env_var_individually(monkeypatch):
    _clear_thread_env(monkeypatch)
    _hide_from_sys_modules(monkeypatch, "numpy", "torch")

    report = configure_threading(n_threads=3, strict=True)

    assert isinstance(report, ThreadingReport)
    for var in THREAD_ENV_VARS:
        expected = "FALSE" if var in _DYNAMIC_VARS else "3"
        assert os.environ[var] == expected, f"{var} was not set correctly"
        assert report.env_after[var] == expected


def test_configure_threading_records_env_before_and_after(monkeypatch):
    _clear_thread_env(monkeypatch)
    _hide_from_sys_modules(monkeypatch, "numpy", "torch")
    monkeypatch.setenv("OMP_NUM_THREADS", "17")

    report = configure_threading(n_threads=1, strict=True)

    assert report.env_before["OMP_NUM_THREADS"] == "17"
    assert report.env_before["MKL_NUM_THREADS"] is None
    assert report.env_after["OMP_NUM_THREADS"] == "1"
    assert set(report.env_before) == set(THREAD_ENV_VARS)
    assert set(report.env_after) == set(THREAD_ENV_VARS)


def test_guard_raises_when_numpy_already_imported_and_strict(monkeypatch):
    _clear_thread_env(monkeypatch)
    _force_present_in_sys_modules(monkeypatch, "numpy")

    with pytest.raises(ThreadingConfiguredTooLateError) as excinfo:
        configure_threading(n_threads=1, strict=True)

    assert "numpy" in str(excinfo.value)
    # The guard must fail BEFORE mutating anything -- no partial effect.
    for var in THREAD_ENV_VARS:
        assert var not in os.environ


def test_guard_raises_when_torch_already_imported_and_strict(monkeypatch):
    _clear_thread_env(monkeypatch)
    _hide_from_sys_modules(monkeypatch, "numpy")
    _force_present_in_sys_modules(monkeypatch, "torch")

    with pytest.raises(ThreadingConfiguredTooLateError) as excinfo:
        configure_threading(n_threads=1, strict=True)

    assert "torch" in str(excinfo.value)


def test_guard_does_not_raise_when_strict_is_false(monkeypatch):
    _clear_thread_env(monkeypatch)
    _force_present_in_sys_modules(monkeypatch, "numpy")

    report = configure_threading(n_threads=2, strict=False)

    assert report.guard_passed is False
    assert "numpy" in report.already_imported
    # Even though the guard "failed", strict=False still applies the env vars.
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_guard_passes_when_neither_numpy_nor_torch_imported(monkeypatch):
    _clear_thread_env(monkeypatch)
    _hide_from_sys_modules(monkeypatch, "numpy", "torch")

    report = configure_threading(n_threads=4, strict=True)

    assert report.guard_passed is True
    assert report.already_imported == ()


def test_torch_num_threads_only_populated_when_torch_already_imported(monkeypatch):
    _clear_thread_env(monkeypatch)
    _hide_from_sys_modules(monkeypatch, "numpy", "torch")

    report = configure_threading(n_threads=1, strict=True)

    assert report.torch_num_threads is None


def test_describe_threading_environment_is_read_only(monkeypatch):
    _clear_thread_env(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "5")

    env_before = dict(os.environ)
    result = describe_threading_environment()
    env_after = dict(os.environ)

    assert env_before == env_after
    assert isinstance(result, dict)
    assert result["env"]["OMP_NUM_THREADS"] == "5"


def test_describe_threading_environment_reports_none_when_torch_and_numpy_absent(monkeypatch):
    _hide_from_sys_modules(monkeypatch, "numpy", "torch")

    result = describe_threading_environment()

    assert result["torch_num_threads"] is None
    assert result["torch_num_interop_threads"] is None
    assert result["numpy_backend"] is None


def test_configure_threading_rejects_nonpositive_n_threads(monkeypatch):
    _hide_from_sys_modules(monkeypatch, "numpy", "torch")
    with pytest.raises(ValueError):
        configure_threading(n_threads=0, strict=True)
