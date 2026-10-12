from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mjrl.mjx_pipeline.factory import AcceleratorFactory
from mjrl.mjx_pipeline.observer import ProbeEvent
from mjrl.mjx_pipeline.strategy import CPUFallbackStrategy, MJXAcceleratorStrategy
from mjrl.mjx_pipeline.trainer import MJXTrainReport
from mjrl.mjx_pipeline.trainer import validate_trainable_model
from tools import mjx_train_clutter


class _FakeMJX(MJXAcceleratorStrategy):
    name = "mjx"

    def probe(self, sink) -> bool:  # noqa: ANN001
        sink.on_event(type("E", (), {"name": "probe.mjx.available", "payload": {}})())
        return True


class _FakeTrainer:
    called = False

    def __init__(self, *, sink=None) -> None:  # noqa: ANN001
        _ = sink

    def train(self, *, model_path: Path, out_dir: Path, cfg):  # noqa: ANN001
        _FakeTrainer.called = True
        return MJXTrainReport(
            iterations=cfg.iterations,
            final_loss=1.23,
            checkpoint=out_dir / "fake.npz",
            model_path=model_path,
        )


class _FakeTrainerWithEvents:
    called = False

    def __init__(self, *, sink=None) -> None:  # noqa: ANN001
        self._sink = sink

    def train(self, *, model_path: Path, out_dir: Path, cfg):  # noqa: ANN001
        _FakeTrainerWithEvents.called = True
        if self._sink is not None:
            self._sink.on_event(
                ProbeEvent(
                    name="training.stage",
                    payload={"stage": "model_loaded", "seconds": 0.1},
                )
            )
            self._sink.on_event(
                ProbeEvent(
                    name="training.iteration",
                    payload={"step": 1, "loss": 0.5, "avg_reward": 0.6},
                )
            )
        return MJXTrainReport(
            iterations=cfg.iterations,
            final_loss=0.5,
            checkpoint=out_dir / "fake.npz",
            model_path=model_path,
        )


class _CaptureCfgTrainer:
    seen_cfg = None

    def __init__(self, *, sink=None) -> None:  # noqa: ANN001
        _ = sink

    def train(self, *, model_path: Path, out_dir: Path, cfg):  # noqa: ANN001
        _CaptureCfgTrainer.seen_cfg = cfg
        return MJXTrainReport(
            iterations=cfg.iterations,
            final_loss=0.0,
            checkpoint=out_dir / "fake.npz",
            model_path=model_path,
        )


def test_given_cpu_fallback_when_require_mjx_then_cli_exits_with_2(tmp_path: Path) -> None:
    # Given: strategy selection resolves to CPU fallback only.
    factory = AcceleratorFactory(
        mjx_strategy=CPUFallbackStrategy(), cpu_strategy=CPUFallbackStrategy()
    )

    # When: CLI requires MJX.
    code = mjx_train_clutter.main(
        ["--prefer", "auto", "--out-dir", str(tmp_path), "--require-mjx"],
        factory=factory,
        trainer_cls=_FakeTrainer,
    )

    # Then: it fails fast without entering training.
    assert code == 2
    assert _FakeTrainer.called is False


def test_given_mjx_available_when_require_mjx_then_training_starts(tmp_path: Path) -> None:
    # Given: MJX strategy is available through factory.
    _FakeTrainer.called = False
    factory = AcceleratorFactory(
        mjx_strategy=_FakeMJX(), cpu_strategy=CPUFallbackStrategy()
    )

    # When: CLI runs with MJX required.
    code = mjx_train_clutter.main(
        [
            "--prefer",
            "mjx",
            "--out-dir",
            str(tmp_path),
            "--iterations",
            "5",
            "--horizon",
            "4",
        ],
        factory=factory,
        trainer_cls=_FakeTrainer,
    )

    # Then: trainer is invoked and exits successfully.
    assert code == 0
    assert _FakeTrainer.called is True


def test_given_tensorboard_enabled_when_training_runs_then_events_file_is_written(tmp_path: Path) -> None:
    if importlib.util.find_spec("tensorboard") is None:
        return
    # Given: available MJX strategy and TensorBoard log target.
    _FakeTrainerWithEvents.called = False
    factory = AcceleratorFactory(
        mjx_strategy=_FakeMJX(), cpu_strategy=CPUFallbackStrategy()
    )
    tb_dir = tmp_path / "tb"

    # When: CLI runs with TensorBoard enabled.
    code = mjx_train_clutter.main(
        [
            "--prefer",
            "mjx",
            "--out-dir",
            str(tmp_path / "run"),
            "--tb-logdir",
            str(tb_dir),
            "--tensorboard",
            "--iterations",
            "1",
            "--horizon",
            "1",
        ],
        factory=factory,
        trainer_cls=_FakeTrainerWithEvents,
    )

    # Then: training succeeds and tfevents artifacts are emitted.
    assert code == 0
    assert _FakeTrainerWithEvents.called is True
    assert list(tb_dir.glob("events.out.tfevents*"))


def test_given_es_batch_cli_when_mjx_training_starts_then_cfg_carries_batch_size(tmp_path: Path) -> None:
    _CaptureCfgTrainer.seen_cfg = None
    factory = AcceleratorFactory(
        mjx_strategy=_FakeMJX(), cpu_strategy=CPUFallbackStrategy()
    )
    code = mjx_train_clutter.main(
        [
            "--prefer",
            "mjx",
            "--out-dir",
            str(tmp_path),
            "--optimizer",
            "es",
            "--es-samples",
            "9",
            "--es-batch-size",
            "3",
            "--strip-visual-meshes",
            "--strip-noncolliding-geoms",
            "--collision-keep",
            "LM_foot_tip_link_meshcol,RM_foot_tip_link_meshcol",
            "--iterations",
            "1",
            "--horizon",
            "1",
        ],
        factory=factory,
        trainer_cls=_CaptureCfgTrainer,
    )
    assert code == 0
    assert _CaptureCfgTrainer.seen_cfg is not None
    assert _CaptureCfgTrainer.seen_cfg.es_samples == 9
    assert _CaptureCfgTrainer.seen_cfg.es_batch_size == 3
    assert _CaptureCfgTrainer.seen_cfg.strip_visual_meshes is True
    assert _CaptureCfgTrainer.seen_cfg.strip_noncolliding_geoms is True
    assert _CaptureCfgTrainer.seen_cfg.collision_keep_patterns == (
        "LM_foot_tip_link_meshcol",
        "RM_foot_tip_link_meshcol",
    )


def test_given_model_without_actuators_when_training_starts_then_it_is_rejected() -> None:
    class Model:
        nu = 0

    with pytest.raises(ValueError, match="at least one actuator"):
        validate_trainable_model(Model())
