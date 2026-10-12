#!/usr/bin/env python3
"""Start the isolated MJX training track for AMD bring-up."""

from __future__ import annotations

import argparse
from numbers import Real
from pathlib import Path
import time
from typing import Any

from mjrl.mjx_pipeline import (
    AcceleratorFactory,
    EventSink,
    ListEventSink,
    MJXTrainConfig,
    PrintEventSink,
    ProbeEvent,
    TeeEventSink,
)
from mjrl.mjx_pipeline.trainer import MJXLinearPolicyTrainer


class TensorBoardEventSink:
    """Write MJX training/probe events into TensorBoard scalar streams."""

    def __init__(self, log_dir: Path) -> None:
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._writer = EventFileWriter(str(log_dir))
        self._event_idx = 0

    def on_event(self, event: ProbeEvent) -> None:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        self._event_idx += 1
        payload = event.payload or {}
        step = int(payload.get("step", self._event_idx))
        for key, value in payload.items():
            if key == "step":
                continue
            if isinstance(value, bool):
                scalar_value = float(int(value))
            elif isinstance(value, Real):
                scalar_value = float(value)
            else:
                continue
            summary = Summary(
                value=[
                    Summary.Value(
                        tag=self._tag(event.name, key),
                        simple_value=scalar_value,
                    )
                ]
            )
            self._writer.add_event(Event(wall_time=time.time(), step=step, summary=summary))
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()

    @staticmethod
    def _tag(event_name: str, key: str) -> str:
        return f"MJX/{event_name.replace('.', '/')}/{key}"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mjx_train_clutter.py", description=__doc__)
    p.add_argument("--prefer", choices=("auto", "mjx", "cpu"), default="auto")
    p.add_argument(
        "--model",
        type=Path,
        default=Path("assets/jumper/jumper.xml"),
        help="MuJoCo XML model used for the MJX starter trainer",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/mjx_clutter"),
        help="output directory for checkpoints",
    )
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--optimizer", choices=("grad", "es"), default="grad")
    p.add_argument("--es-sigma", type=float, default=0.02)
    p.add_argument("--es-samples", type=int, default=8)
    p.add_argument("--es-batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--strip-visual-meshes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="drop non-colliding visual meshes before MJX compile (dynamics preserved)",
    )
    p.add_argument(
        "--strip-noncolliding-geoms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="drop all non-colliding geoms before MJX compile (dynamics preserved)",
    )
    p.add_argument(
        "--collision-keep",
        type=str,
        default="",
        help="comma-separated glob patterns of colliding geoms to keep active; others disabled",
    )
    p.add_argument("--require-mjx", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--show-events", action="store_true")
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--tb-logdir",
        type=Path,
        default=None,
        help="TensorBoard event directory (defaults to --out-dir when enabled)",
    )
    p.add_argument(
        "--stream-events",
        action="store_true",
        help="print events as they happen (useful for remote JIT/debug visibility)",
    )
    return p


def main(
    argv: list[str] | None = None,
    *,
    factory: AcceleratorFactory | None = None,
    trainer_cls: type[MJXLinearPolicyTrainer] = MJXLinearPolicyTrainer,
) -> int:
    args = _parser().parse_args(argv)
    collector = ListEventSink()
    sinks: list[EventSink] = [collector]
    if args.stream_events:
        sinks.append(PrintEventSink())
    tb_sink = None
    if args.tensorboard:
        tb_logdir = args.tb_logdir or args.out_dir
        tb_logdir.mkdir(parents=True, exist_ok=True)
        try:
            tb_sink = TensorBoardEventSink(tb_logdir)
            sinks.append(tb_sink)
        except ModuleNotFoundError:
            print("TensorBoard module unavailable; continuing without event-file logging.")
    sink = TeeEventSink(*sinks) if len(sinks) > 1 else collector
    fact = factory or AcceleratorFactory()
    try:
        strategy = fact.create(args.prefer, sink)

        if args.require_mjx and strategy.name != "mjx":
            print(
                "MJX unavailable on this machine; selected fallback="
                f"{strategy.name}. Re-run with --no-require-mjx to continue on CPU."
            )
            if args.show_events:
                _print_events(collector.events)
            return 2

        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        keep_patterns = tuple(
            p.strip() for p in args.collision_keep.split(",") if p.strip()
        )
        cfg = MJXTrainConfig(
            iterations=args.iterations,
            horizon=args.horizon,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            es_sigma=args.es_sigma,
            es_samples=args.es_samples,
            es_batch_size=args.es_batch_size,
            strip_visual_meshes=args.strip_visual_meshes,
            strip_noncolliding_geoms=args.strip_noncolliding_geoms,
            collision_keep_patterns=keep_patterns,
            seed=args.seed,
        )
        trainer = trainer_cls(sink=sink)
        report = trainer.train(model_path=args.model, out_dir=out_dir, cfg=cfg)
        print(
            f"requested={args.prefer} selected={strategy.name} "
            f"iterations={report.iterations} final_loss={report.final_loss:.6f} "
            f"checkpoint={report.checkpoint}"
        )
        if args.show_events:
            _print_events(collector.events)
        return 0
    finally:
        if tb_sink is not None:
            tb_sink.close()


def _print_events(events: list[Any]) -> None:
    for event in events:
        details = ", ".join(f"{k}={v}" for k, v in sorted(event.payload.items()))
        print(f"- {event.name}" + (f" ({details})" if details else ""))


if __name__ == "__main__":
    raise SystemExit(main())
