"""Persistent CPU workers for unchanged pickup checks, NOT parallel PPO.

Each worker owns one simulator and controller at a time. Completed workers take
the next queued case. Imports are reused, environments are rebuilt per case,
and failures are retained without retries. No videos, training changes or
checkpoint writes. Keep this module's top level free of simulation imports:
spawned workers set single-thread limits before importing those libraries.
"""

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import io
import itertools
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

from mjrl.sweep import run_process_sweep, worker_capacity


def position_cases(args):
    """Enumerate all requested layouts before queuing; never filter or retry.

    Repeated grid flags specify XY points, with the full object x bin product.
    A grid replaces that entity's single offset, rather than adding to it.
    """
    from .pickup_check import fixture_positions

    objects = args.object_grid_mm or [args.object_offset_mm]
    bins = args.bin_grid_mm or [args.bin_offset_mm]
    sides = ("left", "right") if args.side == "both" else (args.side,)
    if len(objects) * len(bins) > 256:
        raise ValueError("position grid must contain at most 256 object/bin pairs")
    count = len(objects) * len(bins) * len(sides) * len(args.seeds) * args.repeat
    if not 1 <= count <= 10000:
        raise ValueError("position sweep must contain between 1 and 10000 cases")
    cases = []
    for side, obj, tray in itertools.product(sides, objects, bins):
        fixture_positions(side, object_offset_mm=obj, bin_offset_mm=tray)
        for seed in args.seeds * args.repeat:
            cases.append({"side": side, "seed": seed, "noise_mm": args.noise_mm,
                          "seconds": args.seconds, "object_offset_mm": tuple(obj),
                          "bin_offset_mm": tuple(tray),
                          "negative_control": args.negative_control,
                          "sensor_dropout": args.sensor_dropout})
    return tuple(cases)


def _pickup_case(case):
    started = time.monotonic()
    # Heavy imports happen only after the pool's thread-limit initializer.
    from mjlab.envs import ManagerBasedRlEnv
    from .pickup_check import run

    steps, substeps = 0, 0
    step_wall = 0.
    original = ManagerBasedRlEnv.step

    def counted(env, action):
        nonlocal steps, substeps, step_wall
        step_started = time.monotonic()
        result = original(env, action)
        step_wall += time.monotonic() - step_started
        steps += 1
        substeps += env.cfg.decimation
        return result

    args = SimpleNamespace(**case, video=None)
    with io.StringIO() as log, redirect_stdout(log), redirect_stderr(log):
        # Restore the original method even on failure before reusing the worker.
        with patch.object(ManagerBasedRlEnv, "step", counted):
            report = run(args)
        diagnostics = log.getvalue()
    return {
        "seed": case["seed"], "side": case.get("side", "left"),
        "object_offset_mm": case.get("object_offset_mm", (0, 0)),
        "bin_offset_mm": case.get("bin_offset_mm", (0, 0)),
        "worker_pid": os.getpid(), "report": report,
        "control_steps": steps, "physics_substeps": substeps,
        "case_wall_seconds": time.monotonic() - started,
        "env_step_wall_seconds": step_wall, "diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulated-estimates", action="store_true", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(12)))
    parser.add_argument("--side", choices=("left", "right", "both"), default="left")
    object_positions = parser.add_mutually_exclusive_group()
    bin_positions = parser.add_mutually_exclusive_group()
    object_positions.add_argument("--object-offset-mm", nargs=2, type=float, default=(0, 0),
                        metavar=("X", "Y"))
    bin_positions.add_argument("--bin-offset-mm", nargs=2, type=float, default=(0, 0),
                        metavar=("X", "Y"))
    object_positions.add_argument("--object-grid-mm", nargs=2, type=float, action="append",
                        metavar=("X", "Y"), help="repeat for each object XY point; replaces offset")
    bin_positions.add_argument("--bin-grid-mm", nargs=2, type=float, action="append",
                        metavar=("X", "Y"), help="repeat for each bin XY point; replaces offset")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(12, max(1, worker_capacity() // 2)))
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--noise-mm", type=float, default=2.)
    parser.add_argument("--wall-timeout", type=float, default=600.)
    parser.add_argument("--fresh-workers", action="store_true",
                        help="recreate workers per case, for a cold-start comparison")
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--sensor-dropout", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 100 or len(args.seeds) * args.repeat > 10000:
        parser.error("repeat must be in [1, 100], with at most 10000 cases")
    if any(not 0 <= seed <= 2**32 - 2 for seed in args.seeds):
        parser.error("seeds must be in [0, 2**32 - 2]")
    for value, low, high, name in ((args.seconds, 1, 60, "seconds"),
                                  (args.noise_mm, 0, 20, "noise-mm"),
                                  (args.wall_timeout, .01, 86400, "wall-timeout")):
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f"{name} must be finite and in [{low}, {high}]")
    if not 1 <= args.workers <= worker_capacity():
        parser.error("workers must be positive and no greater than available CPUs or 64")
    if args.report.exists():
        parser.error("report already exists; choose a new path to preserve previous results")
    try:
        cases = position_cases(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"[pickup-sweep] {len(cases)} cases, {min(args.workers, len(cases))} workers, "
          f"{'fresh' if args.fresh_workers else 'persistent'}, NOT PPO", flush=True)
    sweep = run_process_sweep(_pickup_case, cases, workers=args.workers,
                              timeout_s=args.wall_timeout,
                              max_cases_per_worker=1 if args.fresh_workers else None)
    controls = sum(row["control_steps"] for row in sweep.results)
    substeps = sum(row["physics_substeps"] for row in sweep.results)
    summary = {
        "side": args.side,
        "requested_cases": len(cases),
        "completed_cases": len(sweep.results),
        "successful_cases": sum(row["report"]["success"] for row in sweep.results),
        "failure_types": dict(Counter(
            "unreachable target" if row["report"]["reason"].startswith("unreachable target")
            else row["report"]["reason"] for row in sweep.results if not row["report"]["success"])),
        "workers": min(args.workers, len(cases)), "fresh_workers": args.fresh_workers,
        "wall_seconds": sweep.wall_seconds,
        "total_control_world_steps": controls, "total_physics_substeps": substeps,
        "control_world_steps_per_second": controls / sweep.wall_seconds,
        "physics_substeps_per_second": substeps / sweep.wall_seconds,
        "cases_per_second": len(sweep.results) / sweep.wall_seconds,
        "scope": "scripted CPU diagnostic with synthetic estimates, NOT PPO training",
        "cases": sweep.results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as output:
        output.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    # Completed failed pickup attempts are valid measurements, not runner errors.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
