"""Verified demonstrations -> existing sensor-only actor -> unassisted replay.

Run with ``python -m tasks.jumper.pickup.bootstrap --help``. The teacher is a
collection-only Strategy; evaluation accepts only an observation-to-action
callable. It never constructs the teacher or substitutes its actions.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from tensordict import TensorDict


def bounded_int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def bounded_action(action):
    if (not isinstance(action, torch.Tensor) or action.shape != (1, 5) or
            not action.is_floating_point() or not bool(torch.isfinite(action).all())):
        raise ValueError('teacher must produce finite [1,5] actions before saturation')
    return action.clamp(-1, 1)


def bootstrap_agent_cfg(initial_std):
    from .rl_cfg import agent_cfg
    if (isinstance(initial_std, bool) or not isinstance(initial_std, (float, int)) or
            not np.isfinite(initial_std) or not 0 < initial_std <= 1.):
        raise ValueError('initial-std must be finite in (0, 1]')
    cfg = asdict(agent_cfg())
    cfg['actor']['distribution_cfg']['init_std'] = initial_std
    return cfg


@dataclass(frozen=True)
class DemoEpisode:
    seed: int
    observations: torch.Tensor
    actions: torch.Tensor

    def __post_init__(self):
        bounded_int(self.seed, 'seed', 0, 2**31 - 1)
        obs, actions = self.observations, self.actions
        if (not isinstance(obs, torch.Tensor) or not isinstance(actions, torch.Tensor) or
                obs.ndim != 2 or actions.ndim != 2 or obs.shape[1] != 117 or
                actions.shape != (len(obs), 5) or not 1 <= len(obs) <= 1500 or
                not obs.is_floating_point() or not actions.is_floating_point() or
                not bool(torch.isfinite(obs).all() & torch.isfinite(actions).all()) or
                bool((actions.abs() > 1).any())):
            raise ValueError('demonstration needs finite [1..1500,117] sensor inputs and bounded [N,5] actions')


def split_episodes(episodes, holdout_count):
    bounded_int(holdout_count, 'holdout-count', 1, 128)
    if len(episodes) <= holdout_count or len({e.seed for e in episodes}) != len(episodes):
        raise ValueError('need distinct successful episodes in both train and holdout sets')
    return episodes[:-holdout_count], episodes[-holdout_count:]


def collect(env, teacher_factory, *, seeds, max_steps=500, settle_s=1., goal='picked_up'):
    """Accept whole qualifying episodes, never failed/colliding partial traces."""
    bounded_int(max_steps, 'max-steps', 1, 1500)
    if goal not in ('picked_up', 'placed'):
        raise ValueError('unsupported demonstration goal')
    if env.num_envs != 1 or env.cfg.auto_reset:
        raise ValueError('demonstration collection needs one world without automatic resets')
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or len(seeds) > 128:
        raise ValueError('need 1..128 distinct demonstration seeds')
    for seed in seeds:
        bounded_int(seed, 'seed', 0, 2**31 - 1)
    if not isinstance(settle_s, (int, float)) or not np.isfinite(settle_s) or settle_s < 0:
        raise ValueError('settle-s must be finite and nonnegative')
    episodes, reports = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        teacher = teacher_factory(env)
        inputs, labels = [], []
        picked = False
        # Contact sensors lazily allocate buffers while stepping. no_grad keeps
        # those buffers writable by the next reset outside this context.
        with torch.no_grad():
            for step in range(max_steps):
                elapsed = step * env.step_dt
                action = bounded_action(teacher(elapsed))
                # Snapshot BEFORE stepping. Clamp BEFORE env.step as previous raw
                # actions are themselves observations, not merely servo commands.
                if elapsed >= settle_s:
                    inputs.append(obs['actor'].clone().cpu())
                    labels.append(action.clone().cpu())
                obs, _, terminated, truncated, _ = env.step(action)
                if bool((terminated | truncated).any()):
                    failed = any(bool(env.termination_manager.get_term(name)[0]) for name in
                                 ('fell_over', 'too_low', 'invalid_fixture', 'chassis_intrusion'))
                    picked = bool(env.termination_manager.get_term(goal)[0]) and not failed
                    break
        report = {'seed': seed, goal: picked, 'seconds': (step + 1) * env.step_dt,
                  'samples': len(inputs)}
        reports.append(report)
        if picked and inputs:
            episodes.append(DemoEpisode(seed, torch.cat(inputs), torch.cat(labels)))
        print('DEMONSTRATION', json.dumps(report), flush=True)
    return episodes, reports


class LeftPickupTeacher:
    """Previously checked near-reset sequence; privileged labels, not inputs.

    MuJoCo 3.11.0/native CPU, 2026-10-13: LF standard qualified at 4.92 s,
    32.98 mm whole-hull clearance, no chassis penetration. Every new rollout
    must qualify again; that measurement does not authorise accepting a trace.
    """

    def __init__(self, env):
        import mujoco
        from mjrl.manipulation import PointIK
        from ..five_foot.tools.pickup_check import GRASP_ARM
        self.env = env
        self.term = env.action_manager.get_term('claw')
        if (self.term.cfg.side != 'left' or self.term.cfg.training_stage not in ('grasp', 'place-near') or
                self.term.cfg.grasp_start != 'near'):
            raise ValueError('teacher is verified only for left/grasp/near')
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.ik = PointIK(self.model, 'robot/LF_palm_link',
                          tuple(f'robot/LF_J{i}_joint' for i in range(4)),
                          np.asarray(self.term.cfg.local_point))
        self.start = GRASP_ARM.copy()
        self.lift = None

    def __call__(self, elapsed):
        from ..five_foot.tools.pickup_check import GRASP_OFFSET
        self.data.qpos[:] = self.env.sim.data.qpos[0].cpu().numpy()
        target = self.start
        if 1. < elapsed < 2.:
            goal = (self.term.object_data('root_link_pos_w')[0].cpu().numpy() +
                    GRASP_OFFSET * float(self.term.scales[self.term.object_index][0]))
            target = self.ik.solve(self.data.qpos, goal).joints
        if elapsed >= 3.:
            if self.lift is None:
                goal = self.term.evidence.grip_point()[0].cpu().numpy() + np.array([0, 0, .05])
                self.lift = self.ik.solve(self.data.qpos, goal).joints
            target = self.lift
        action = torch.zeros(1, 5, device=self.env.device)
        action[:, :4] = torch.as_tensor(
            (target - self.term.targets[0, :4].cpu().numpy()) /
            (self.term.cfg.arm_speed * self.env.step_dt), device=self.env.device)
        action[:, 4] = -1. if elapsed < 2. else 1.
        return bounded_action(action)


def fit_actor(runner, train, holdout, *, epochs, batch_size, learning_rate, seed,
              update_normalization=True, epoch_callback=None):
    """Behaviour clone only the existing actor MLP; keep the critic/PPO fresh.

    An optional read-only observer receives each completed 1-based epoch before
    the best validation state is restored. Saving a snapshot does not qualify it
    for deployment or replace independent closed-loop policy evaluation.
    """
    bounded_int(epochs, 'epochs', 1, 5000)
    bounded_int(batch_size, 'batch-size', 1, 8192)
    bounded_int(seed, 'seed', 0, 2**31 - 1)
    if epoch_callback is not None and not callable(epoch_callback):
        raise ValueError('epoch-callback must be callable or absent')
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or
            not np.isfinite(learning_rate) or not 0 < learning_rate <= .1):
        raise ValueError('learning-rate must be finite in (0, .1]')
    if not train or not holdout or {e.seed for e in train} & {e.seed for e in holdout}:
        raise ValueError('training and validation must use distinct whole episodes')
    torch.manual_seed(seed)
    actor = runner.alg.actor
    device = next(actor.parameters()).device
    inputs = torch.cat([e.observations for e in train]).to(device)
    targets = torch.cat([e.actions for e in train]).to(device)
    validation = torch.cat([e.observations for e in holdout]).to(device)
    validation_targets = torch.cat([e.actions for e in holdout]).to(device)
    train_obs = TensorDict({'actor': inputs}, batch_size=[len(inputs)])
    valid_obs = TensorDict({'actor': validation}, batch_size=[len(validation)])
    if type(update_normalization) is not bool:
        raise ValueError('update-normalization must be boolean')
    actor.train()
    if update_normalization:
        actor.update_normalization(train_obs)
    elif int(actor.obs_normalizer.count) <= 0:
        raise ValueError('frozen imitation requires fitted source statistics')
    # Freeze statistics after the TRAIN split only. Validation cannot change them.
    actor.eval()
    optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial = float((actor(train_obs) - targets).square().mean())
    if not np.isfinite(initial):
        raise ValueError('initial imitation loss is nonfinite')
    best_mse, best_state, selected_epoch = float('inf'), None, 0
    for epoch in range(epochs):
        for ids in torch.randperm(len(inputs), device=device).split(batch_size):
            loss = (actor(train_obs[ids]) - targets[ids]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('imitation loss is nonfinite')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.)
            optimizer.step()
        with torch.no_grad():
            mse = float((actor(valid_obs) - validation_targets).square().mean())
        if not np.isfinite(mse):
            raise ValueError('validation loss is nonfinite')
        if mse < best_mse:
            best_mse = mse
            best_state = {name: value.clone() for name, value in actor.state_dict().items()}
            selected_epoch = epoch + 1
        if epoch_callback is not None:
            epoch_callback(epoch + 1)
        if epoch % 50 == 0 or epoch == epochs - 1:
            print('IMITATION', json.dumps({'epoch': epoch + 1, 'holdout_mse': mse}), flush=True)
    actor.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        return {'initial_train_mse': initial,
                'train_mse': float((actor(train_obs) - targets).square().mean()),
                'holdout_mse': float((actor(valid_obs) - validation_targets).square().mean()),
                'last_holdout_mse': mse, 'selected_epoch': selected_epoch,
                'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate}


def evaluate(env, policy, seeds, *, video=None):
    """Unassisted, observation-only first episodes; teacher never constructed."""
    from .mdp.safety import chassis_penetration
    from .mdp.terms import grasp_completed
    if env.num_envs != 1 or env.cfg.auto_reset:
        raise ValueError('evaluation needs one world without automatic resets')
    seeds = tuple(seeds)
    if not seeds or len(seeds) > 128 or len(set(seeds)) != len(seeds):
        raise ValueError('evaluation needs 1..128 distinct seeds')
    for seed in seeds:
        bounded_int(seed, 'seed', 0, 2**31 - 1)
    results = []
    with torch.no_grad():
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            term = env.action_manager.get_term('claw')
            start = env.scene['robot'].data.joint_pos[:, term.joints[:4]].clone()
            depth, clearance, excursion, raw_peak = 0., 0., 0., 0.
            overshoot, components = 0, 0
            for step in range(env.max_episode_length):
                action = policy(TensorDict({'actor': obs['actor']}, batch_size=[1]))
                # Preserve the raw-action convention of play.py and PPO. Only
                # the existing action term clamps servo commands; report drift
                # from the bounded demonstration histories rather than hiding it.
                raw_peak = max(raw_peak, float(action.abs().max()))
                overshoot += int((action.abs() > 1).sum())
                components += action.numel()
                obs, _, terminated, truncated, _ = env.step(action)
                if video is not None:
                    video.update()
                depth = max(depth, float(chassis_penetration(env)[0]))
                clearance = max(clearance, float(term.evidence.clearance[0]))
                excursion = max(excursion, float(
                    (env.scene['robot'].data.joint_pos[:, term.joints[:4]] - start).abs().max()))
                if bool((terminated | truncated).any()):
                    break
            result = {'seed': seed, 'seconds': (step + 1) * env.step_dt,
                      'picked_up': bool(env.termination_manager.get_term('picked_up')[0])
                                   if term.cfg.training_stage == 'grasp' else
                                   bool(grasp_completed(env)[0]),
                      'placed': bool(env.termination_manager.get_term('placed')[0])
                                if term.cfg.training_stage != 'grasp' else False,
                      'termination': [name for name in env.termination_manager.active_terms
                                      if bool(env.termination_manager.get_term(name)[0])],
                      'max_chassis_penetration_mm': depth * 1000,
                      'max_floor_clearance_mm': clearance * 1000,
                      'arm_excursion_rad': excursion, 'max_raw_action': raw_peak,
                      'raw_overshoot_fraction': overshoot / components}
            results.append(result)
            print('NN_EVALUATION', json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--holdout-episodes', type=int, default=4)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=.001)
    # Native CPU/3900X, 2026-10-13: same fitted mean, std .6 gave 0/12
    # stochastic pickups (six intrusions); .1 gave 11/12 (no intrusions).
    # This opt-in bootstrap choice does not retune the ordinary task PPO config.
    parser.add_argument('--initial-std', type=float, default=.1,
                        help='Gaussian exploration std saved for PPO, not trained by imitation')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--position-jitter-mm', type=float, default=0.)
    args = parser.parse_args()
    bounded_int(args.episodes, 'episodes', 2, 128)
    bounded_int(args.holdout_episodes, 'holdout-episodes', 1, args.episodes - 1)
    bounded_int(args.eval_episodes, 'eval-episodes', 1, 128)
    bounded_int(args.seed, 'seed', 0, 2**31 - 1128)
    bounded_int(args.epochs, 'epochs', 1, 5000)
    bounded_int(args.batch_size, 'batch-size', 1, 8192)
    if not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .1:
        parser.error('learning-rate must be finite in (0, .1]')
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend
    use_backend(resolve(backend='native', device='cpu', num_envs=1, cpu_threads=1))
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .env_cfg import env_cfg
    from .runner import PickupRunner
    cfg = env_cfg(play=True, training_stage='grasp', grasp_start='near',
                  object_size='standard', position_jitter_mm=args.position_jitter_mm)
    cfg.auto_reset = False
    # Capture source/config identity BEFORE collection. HEAD alone does not
    # identify a dirty checkout, particularly an untracked new trainer module.
    source = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
              'bootstrap_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest()}
    resolved = {'position_jitter_mm': args.position_jitter_mm,
                'detection_dropout': cfg.actions['claw'].dropout,
                'detection_noise_m': cfg.actions['claw'].noise_m,
                'decimation': cfg.decimation, 'physics_dt': cfg.sim.mujoco.timestep,
                'initial_std': args.initial_std, 'seed': args.seed,
                'object_size': 'standard', 'stage': 'grasp', 'start': 'near', 'side': 'left'}
    bootstrap_agent_cfg(args.initial_std)
    output = args.output or Path('logs/jumper/jumper.pickup') / datetime.now().strftime('bootstrap-%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = ManagerBasedRlEnv(cfg, device='cpu')
    try:
        runner = PickupRunner(RslRlVecEnvWrapper(env), bootstrap_agent_cfg(args.initial_std), device='cpu')
        episodes, reports = collect(env, LeftPickupTeacher,
                                     seeds=range(args.seed, args.seed + args.episodes))
        train, holdout = split_episodes(episodes, args.holdout_episodes)
        data = {'schema': 'jumper.pickup.demonstrations/1', 'contract': runner.contract(),
                'episodes': [{'seed': e.seed, 'actor': e.observations, 'actions': e.actions} for e in episodes],
                'reports': reports, 'train_seeds': [e.seed for e in train],
                'holdout_seeds': [e.seed for e in holdout]}
        dataset_path = output / 'demonstrations.pt'
        torch.save(data, dataset_path)
        fit = fit_actor(runner, train, holdout, epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=args.learning_rate, seed=args.seed)
        provenance = {'schema': 'jumper.pickup.imitation/1', 'teacher': 'checked-left-near-sequence/1',
                      'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                      **source, 'resolved_config': resolved,
                      'train_seeds': data['train_seeds'], 'holdout_seeds': data['holdout_seeds'],
                      'fit': fit, 'actor_inputs': 'existing-117-sensor-history-only',
                      'critic': 'fresh-untrained', 'ppo_optimizer': 'fresh', 'step_dt': env.step_dt}
        checkpoint = output / 'warm_start.pt'
        runner.save(str(checkpoint), {'imitation': provenance})
        probe = TensorDict({'actor': holdout[0].observations},
                           batch_size=[len(holdout[0].observations)])
        with torch.no_grad():
            expected = runner.alg.actor(probe).clone()
        # A FRESH runner exercises the production loader and normalization buffers.
        # Replay the saved policy, not the in-memory fitted actor or teacher state.
        runner = PickupRunner(RslRlVecEnvWrapper(env), bootstrap_agent_cfg(args.initial_std), device='cpu')
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device='cpu')
        with torch.no_grad():
            if not torch.equal(expected, policy(probe)):
                raise RuntimeError('saved actor does not reproduce fitted deterministic outputs')
        results = evaluate(env, policy,
                           range(args.seed + 1000, args.seed + 1000 + args.eval_episodes))
        report = {'checkpoint': str(checkpoint), 'demonstrations': reports,
                  'provenance': provenance, 'evaluation': results,
                  'pickups': sum(r['picked_up'] for r in results),
                  'eval_episodes': len(results)}
        with (output / 'report.json').open('x') as destination:
            json.dump(report, destination, indent=2)
        print('BOOTSTRAP_RESULT', json.dumps({'output': str(output), 'fit': fit,
                                             'pickups': report['pickups'], 'episodes': len(results)}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
