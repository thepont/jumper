"""CLI probe for MJX strategy selection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .factory import AcceleratorFactory
from .observer import ListEventSink
from .pipeline import MJXAccelerationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mjx-probe", description=__doc__)
    parser.add_argument("--prefer", choices=("auto", "mjx", "cpu"), default="auto")
    parser.add_argument("--payload-size", type=int, default=8)
    parser.add_argument("--show-events", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, factory: AcceleratorFactory | None = None) -> int:
    args = build_parser().parse_args(argv)
    sink = ListEventSink()
    pipeline = MJXAccelerationPipeline(factory=factory, sink=sink)
    payload = [1.0] * max(0, args.payload_size)
    report = pipeline.run(payload, preference=args.prefer)

    print(
        f"requested={report.requested} selected={report.selected} "
        f"count={report.result['count']} total={report.result['total']:.1f} "
        f"events={len(sink.events)}"
    )
    if args.show_events:
        for event in sink.events:
            details = ", ".join(f"{k}={v}" for k, v in sorted(event.payload.items()))
            print(f"- {event.name}" + (f" ({details})" if details else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
