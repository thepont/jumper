"""Independent cases must reuse isolated workers without loss or oversubscription."""

import os
import time

import pytest

from mjrl.sweep import run_process_sweep


def _identify(value):
    return value, os.getpid(), os.environ.get("OMP_NUM_THREADS"), os.environ.get("MJRL_CPU_THREADS")


def _reject(value):
    raise ValueError(f"rejected case {value}")


def _slow(value):
    time.sleep(.5)
    return value


def test_given_many_cases_one_worker_reuses_startup_and_keeps_input_order():
    result = run_process_sweep(_identify, (17, 3, 17, 9), workers=1)
    assert [row[0] for row in result.results] == [17, 3, 17, 9]
    assert len({row[1] for row in result.results}) == 1
    assert result.results[0][1] != os.getpid()
    assert all(row[2:] == ("1", "1") for row in result.results)
    assert result.wall_seconds > 0


def test_given_worker_failure_the_sweep_does_not_claim_completion():
    with pytest.raises(ValueError, match="rejected case 29"):
        run_process_sweep(_reject, (29,), workers=1)


def test_cold_start_comparison_recreates_workers_without_changing_case_order():
    result = run_process_sweep(_identify, (5, 3, 9), workers=1, max_cases_per_worker=1)
    assert [row[0] for row in result.results] == [5, 3, 9]
    assert len({row[1] for row in result.results}) == 3


def test_child_thread_limits_do_not_mutate_the_parent_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    result = run_process_sweep(_identify, (1,), workers=1)
    assert result.results[0][2] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "3"


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux affinity boundary")
def test_worker_limit_respects_affinity_not_the_host_cpu_count(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0})
    with pytest.raises(ValueError, match="workers"):
        run_process_sweep(_reject, (1,), workers=2)


def test_given_stalled_workers_the_deadline_returns_control():
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        run_process_sweep(_slow, (1,), workers=1, timeout_s=.05)
    assert time.monotonic() - started < 5


def test_empty_cases_require_no_worker_execution():
    assert run_process_sweep(_reject, (), workers=1).results == ()


@pytest.mark.parametrize("options", [
    {"workers": 0}, {"workers": -1}, {"workers": True}, {"workers": 1.5},
    {"workers": 10000}, {"timeout_s": 0}, {"timeout_s": float("nan")},
    {"max_cases_per_worker": 0}, {"max_cases_per_worker": True},
])
def test_invalid_resource_limits_fail_before_worker_execution(options):
    with pytest.raises(ValueError):
        run_process_sweep(_reject, (1,), **options)


def test_unbounded_case_generator_is_rejected_without_hanging():
    import itertools

    with pytest.raises(ValueError, match="case limit"):
        run_process_sweep(_identify, itertools.repeat(1), workers=1)
