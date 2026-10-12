from __future__ import annotations

from types import SimpleNamespace

import pytest

from mjrl.mjx_pipeline.factory import AcceleratorFactory
from mjrl.mjx_pipeline.observer import ListEventSink
from mjrl.mjx_pipeline.pipeline import MJXAccelerationPipeline
from mjrl.mjx_pipeline.strategy import MJXAcceleratorStrategy


def test_given_auto_preference_when_mjx_probe_fails_then_cpu_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: MJX probing fails because imports are unavailable.
    sink = ListEventSink()
    strategy = MJXAcceleratorStrategy()

    def _boom(_name: str):
        raise ImportError("module not found")

    monkeypatch.setattr(strategy, "_import_module", _boom)
    factory = AcceleratorFactory(mjx_strategy=strategy)
    pipeline = MJXAccelerationPipeline(factory=factory, sink=sink)

    # When: the pipeline runs with automatic selection.
    report = pipeline.run([1.0, 2.0], preference="auto")

    # Then: it falls back to the CPU strategy and records why.
    assert report.selected == "cpu"
    assert report.result["backend"] == "cpu"
    assert [event.name for event in sink.events] == [
        "pipeline.started",
        "probe.mjx.started",
        "probe.mjx.unavailable",
        "selection.fallback",
        "selection.chosen",
        "pipeline.finished",
    ]
    assert sink.events[2].payload["error_type"] == "ImportError"


def test_given_mjx_preference_when_mjx_is_available_then_mjx_is_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: both jax and mujoco.mjx can be imported.
    sink = ListEventSink()
    strategy = MJXAcceleratorStrategy()

    def _import(name: str):
        if name == "jax":
            return SimpleNamespace(devices=lambda: ["gpu:0"])
        if name == "mujoco.mjx":
            return object()
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(strategy, "_import_module", _import)
    factory = AcceleratorFactory(mjx_strategy=strategy)
    pipeline = MJXAccelerationPipeline(factory=factory, sink=sink)

    # When: the caller prefers MJX.
    report = pipeline.run([1.0, 2.0, 3.0], preference="mjx")

    # Then: MJX is selected and the probe metadata is emitted.
    assert report.selected == "mjx"
    assert report.result["backend"] == "mjx"
    assert sink.events[1].name == "probe.mjx.started"
    assert sink.events[2].name == "probe.mjx.available"
    assert sink.events[2].payload["device_count"] == 1


def test_given_unknown_preference_when_selecting_then_it_fails_fast() -> None:
    # Given/When/Then: invalid strategy names are rejected at the boundary.
    pipeline = MJXAccelerationPipeline()
    with pytest.raises(ValueError, match="unknown accelerator preference"):
        pipeline.run([1.0], preference="bogus")
