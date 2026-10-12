from __future__ import annotations

from types import SimpleNamespace

import pytest

from mjrl.mjx_pipeline import cli
from mjrl.mjx_pipeline.factory import AcceleratorFactory
from mjrl.mjx_pipeline.strategy import MJXAcceleratorStrategy


def test_given_probe_command_when_mjx_is_unavailable_then_report_mentions_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: MJX cannot be imported on this machine.
    strategy = MJXAcceleratorStrategy()

    def _boom(_name: str):
        raise ImportError("module not found")

    monkeypatch.setattr(strategy, "_import_module", _boom)
    factory = AcceleratorFactory(mjx_strategy=strategy)

    # When: the probe CLI runs with auto preference and event output enabled.
    exit_code = cli.main(["--prefer", "auto", "--payload-size", "4", "--show-events"], factory=factory)
    out = capsys.readouterr().out

    # Then: a concise report includes the selected backend and fallback trace.
    assert exit_code == 0
    assert "requested=auto selected=cpu" in out
    assert "probe.mjx.unavailable" in out
    assert "selection.fallback" in out


def test_given_probe_command_when_mjx_is_available_then_report_mentions_mjx(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: imports and devices look valid for MJX.
    strategy = MJXAcceleratorStrategy()

    def _import(name: str):
        if name == "jax":
            return SimpleNamespace(devices=lambda: ["gpu:0", "gpu:1"])
        if name == "mujoco.mjx":
            return object()
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(strategy, "_import_module", _import)
    factory = AcceleratorFactory(mjx_strategy=strategy)

    # When: MJX is requested directly.
    exit_code = cli.main(["--prefer", "mjx"], factory=factory)
    out = capsys.readouterr().out

    # Then: report selects MJX and includes concise probe stats.
    assert exit_code == 0
    assert "requested=mjx selected=mjx" in out
    assert "events=5" in out
