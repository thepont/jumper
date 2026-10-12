"""Bounded independent-case sweeps; no shared simulator or controller state."""

from dataclasses import dataclass
import itertools
import math
import multiprocessing
import os
import pickle
import time

from mjrl.backend.resolve import _available_cores


@dataclass(frozen=True)
class SweepResult:
    results: tuple
    wall_seconds: float


def worker_capacity():
    """Use the native backend's affinity-aware resource authority."""
    return min(_available_cores(), 64)


def _single_thread_worker():
    # Spawn imports the caller before this initializer. Callers must therefore
    # keep heavyweight simulation imports inside their case function.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "MJRL_CPU_THREADS"):
        os.environ[name] = "1"


def _case(job):
    index, operation, item = job
    return index, operation(item)


def run_process_sweep(operation, cases, *, workers=1, timeout_s=600.,
                      max_cases_per_worker=None) -> SweepResult:
    """Reuse isolated spawned workers, retaining case order and explicit failure.

    The callable is the Strategy for a case. It must be importable, bounded and
    own its resources; returning a failed experiment is distinct from crashing.
    The wall deadline includes startup and all queued work. Timeout or worker
    errors terminate only this sweep's owned child processes. Training settings,
    parent thread limits and unrelated processes are never modified.
    """
    capacity = worker_capacity()
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= capacity:
        raise ValueError(f"workers must be an integer in [1, {capacity}]")
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 86400:
        raise ValueError("timeout must be finite and in (0, 86400]")
    if max_cases_per_worker is not None and (
        isinstance(max_cases_per_worker, bool) or not isinstance(max_cases_per_worker, int)
        or not 1 <= max_cases_per_worker <= 10000
    ):
        raise ValueError("worker case limit must be an integer in [1, 10000]")
    if not callable(operation):
        raise ValueError("case strategy must be callable")
    items = tuple(itertools.islice(cases, 10001))
    if len(items) > 10000:
        raise ValueError("case limit is 10000")
    try:
        pickle.dumps((operation, items))
    except (TypeError, AttributeError, pickle.PicklingError) as error:
        raise ValueError("case strategy and inputs must be serializable") from error
    started = time.monotonic()
    if not items:
        return SweepResult((), time.monotonic() - started)
    jobs = ((index, operation, item) for index, item in enumerate(items))
    results = [None] * len(items)
    with multiprocessing.get_context("spawn").Pool(
        processes=min(workers, len(items)), initializer=_single_thread_worker,
        maxtasksperchild=max_cases_per_worker,
    ) as pool:
        pending = pool.imap_unordered(_case, jobs, chunksize=1)
        for _ in items:
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("sweep wall deadline exceeded")
            try:
                index, result = pending.next(timeout=remaining)
            except multiprocessing.TimeoutError as error:
                raise TimeoutError("sweep wall deadline exceeded") from error
            results[index] = result
    return SweepResult(tuple(results), time.monotonic() - started)
