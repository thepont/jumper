"""Bounded serial size experiments; preserve rejected policies and evidence."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .bootstrap import bounded_int
from .size_curriculum import queue_decision

REFERENCE = Path('logs/jumper/jumper.pickup/placement-fit-snapshot-sweep-2026-10-13/fit_epoch_100.pt')


def _write_manifest(output, manifest):
    pending = output / 'queue.pending.json'
    pending.write_text(json.dumps(manifest, indent=2))
    pending.replace(output / 'queue.json')


def evaluate_pair(reference, source_variation, candidate, variation, seed, output, deadline):
    """Use the existing physical evaluator, with matched fresh seeds per size."""
    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend
    use_backend(resolve(backend='native', device='cpu', num_envs=1, cpu_threads=1))
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .bootstrap import evaluate
    from .env_cfg import env_cfg
    from .rl_cfg import agent_cfg
    from .runner import PickupRunner
    torch.set_num_threads(1)
    reports = {}
    for label, path, width in (('reference', reference, source_variation),
                               ('candidate', candidate, variation)):
        cfg = env_cfg(play=True, training_stage='place-near', position_jitter_mm=0.,
                      bounded_action_history=True, size_variation=width)
        env = ManagerBasedRlEnv(cfg, device='cpu')
        try:
            runner = PickupRunner(RslRlVecEnvWrapper(env), asdict(agent_cfg()), device='cpu')
            runner.load(str(path))
            policy = runner.get_inference_policy(device='cpu')
        finally:
            env.close()
        reports[label] = {}
        for index, size in enumerate(('small', 'standard', 'large')):
            cfg = env_cfg(play=True, training_stage='place-near', object_size=size,
                size_variation=variation, position_jitter_mm=0., bounded_action_history=True)
            cfg.auto_reset = False
            env = ManagerBasedRlEnv(cfg, device='cpu')
            rows = []
            try:
                for trial_seed in range(seed + index * 100, seed + index * 100 + 12):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('evaluation reached its compute deadline')
                    env.common_step_counter = 0
                    row = evaluate(env, policy, [trial_seed])[0]
                    term = env.action_manager.get_term('claw')
                    row['terminal_evidence'] = {name: bool(getattr(term.evidence, name)[0])
                        for name in ('picked', 'pinched', 'inside', 'released')}
                    rows.append(row)
                    print('QUEUE_TRIAL', json.dumps({'actor': label, 'size': size, **row}), flush=True)
            finally:
                env.close()
            reports[label][size] = rows
            with (output / f'{label}_{size}.json').open('x') as target:
                json.dump({'checkpoint': str(path), 'variation': variation,
                           'fresh_clock': True, 'results': rows}, target, indent=2)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--minutes', type=int, default=45)
    parser.add_argument('--job-minutes', type=int, default=15)
    parser.add_argument('--seed', type=int, default=133)
    args = parser.parse_args()
    bounded_int(args.runs, 'runs', 1, 5)
    bounded_int(args.minutes, 'minutes', 1, 120)
    bounded_int(args.job_minutes, 'job-minutes', 1, 30)
    bounded_int(args.seed, 'seed', 0, 2**31 - 110000)
    if not REFERENCE.is_file():
        raise ValueError('explicit initial reference checkpoint is missing')
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    deadline = began + args.minutes * 60
    incumbent, incumbent_width, variation = REFERENCE, None, .05
    manifest = {'schema': 'jumper.pickup.size-queue/1', 'initial_incumbent': str(REFERENCE),
        'incumbent': str(REFERENCE), 'incumbent_variation': None, 'next_variation': variation,
        'runs_limit': args.runs, 'minutes_limit': args.minutes, 'job_minutes_limit': args.job_minutes,
        'completed_jobs': 0, 'status': 'running', 'jobs': [], 'hardware': 'AMD Ryzen 9 3900X',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
        'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest(),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'mujoco', 'numpy')},
        'protocol': {'backend': 'native', 'device': 'cpu', 'workers': 1, 'worlds': 1,
            'native_threads': 1, 'physics_hz': 200, 'control_hz': 50,
            'position_jitter_mm': 0., 'detection_noise_m': .002, 'dropout': .05,
            'attempts': 36, 'holdout_episodes': 6, 'fit_epochs': 200,
            'paired_trials_per_size': 12, 'hardware_promoted': False,
            'training': 'sequential actor-only imitation, not independent PPO seeds'}}
    _write_manifest(args.output, manifest)
    child_env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for index in range(args.runs):
        if time.monotonic() >= deadline:
            manifest['status'] = 'budget-expired'
            break
        job_deadline = min(deadline, time.monotonic() + args.job_minutes * 60)
        output = args.output / f'run_{index + 1}'
        output.mkdir()
        fit = output / 'fit'
        command = [sys.executable, '-m', 'tasks.jumper.pickup.place_train',
            '--placement-checkpoint', str(incumbent), '--output', str(fit),
            '--episodes', '36', '--holdout-episodes', '6', '--eval-episodes', '1',
            '--epochs', '200', '--seed', str(args.seed + index * 1000),
            '--size-variation', str(variation), '--position-jitter-mm', '0',
            '--bounded-action-history', '--save-fit-every', '50']
        if incumbent_width is not None:
            command.extend(['--source-size-variation', str(incumbent_width)])
        job = {'run': index + 1, 'variation': variation, 'source': str(incumbent),
               'command': command, 'status': 'training', 'evaluation_seed': args.seed + 100000 + index * 1000}
        manifest['jobs'].append(job)
        _write_manifest(args.output, manifest)
        print('QUEUE_JOB', json.dumps(job), flush=True)
        started = time.monotonic()
        try:
            with (output / 'training.log').open('x') as log:
                child = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=max(.001, job_deadline - time.monotonic()))
            if child.returncode:
                job['status'] = 'training-failed-preserved'
                job['returncode'] = child.returncode
            else:
                candidate = fit / 'placement_warm_start.pt'
                job['status'] = 'evaluating'
                _write_manifest(args.output, manifest)
                reports = evaluate_pair(incumbent, incumbent_width, candidate, variation,
                                        job['evaluation_seed'], output, job_deadline)
                decision = queue_decision(variation, reports['reference'], reports['candidate'])
                job.update(decision, candidate=str(candidate))
                job['summary'] = {label: {size: {
                    'placed': sum(r['placed'] for r in rows), 'picked_up': sum(r['picked_up'] for r in rows),
                    'intrusions': sum('chassis_intrusion' in r['termination'] for r in rows)}
                    for size, rows in sizes.items()} for label, sizes in reports.items()}
                if decision['retain_candidate']:
                    incumbent, incumbent_width = candidate, variation
                variation = decision['next_variation']
        except (TimeoutError, subprocess.TimeoutExpired):
            job['status'] = 'timed-out-preserved'
        job['wall_seconds'] = time.monotonic() - started
        manifest.update(completed_jobs=index + 1, incumbent=str(incumbent),
                        incumbent_variation=incumbent_width, next_variation=variation)
        _write_manifest(args.output, manifest)
        print('QUEUE_RESULT', json.dumps(job), flush=True)
    if manifest['status'] == 'running':
        manifest['status'] = 'review-required'
    manifest['wall_seconds'] = time.monotonic() - began
    _write_manifest(args.output, manifest)
    print('QUEUE_FINISHED', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
