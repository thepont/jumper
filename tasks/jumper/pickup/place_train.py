"""Near-start carry/place curriculum: verified labels, sensor-only NN replay."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from tensordict import TensorDict

from .bootstrap import LeftPickupTeacher, bounded_action, bounded_int, collect, evaluate, fit_actor, split_episodes
from .bootstrap import DemoEpisode


def load_grasp_rehearsal(path, placement_train, placement_holdout, source_contract):
    """Load qualified whole episodes; keep validation seeds out of BOTH goals."""
    with path.open('rb') as source:
        contents = source.read(32 * 1024 * 1024 + 1)
    if len(contents) > 32 * 1024 * 1024:
        raise ValueError('grasp demonstrations exceed 32 MiB')
    data = torch.load(io.BytesIO(contents), weights_only=True, map_location='cpu')
    if (not isinstance(data, dict) or data.get('schema') != 'jumper.pickup.demonstrations/1' or
            data.get('contract') != source_contract):
        raise ValueError('grasp demonstrations need the compatible qualified grasp contract')
    episodes, reports = data.get('episodes'), data.get('reports')
    if (not isinstance(episodes, list) or not 2 <= len(episodes) <= 128 or
            not isinstance(reports, list) or not 2 <= len(reports) <= 128):
        raise ValueError('grasp demonstrations need bounded episodes and qualification reports')
    qualified = {}
    for report in reports:
        if not isinstance(report, dict) or 'seed' not in report:
            raise ValueError('invalid grasp qualification report')
        bounded_int(report['seed'], 'seed', 0, 2**31 - 1)
        if report['seed'] in qualified:
            raise ValueError('duplicate grasp qualification seed')
        qualified[report['seed']] = report.get('picked_up') is True
    parsed = {}
    for item in episodes:
        if not isinstance(item, dict) or not {'seed', 'actor', 'actions'} <= item.keys():
            raise ValueError('invalid grasp episode')
        episode = DemoEpisode(item['seed'], item['actor'], item['actions'])
        if episode.seed in parsed or not qualified.get(episode.seed, False):
            raise ValueError('duplicate or unqualified grasp episode')
        parsed[episode.seed] = episode
    groups = []
    for name in ('train_seeds', 'holdout_seeds'):
        ids = data.get(name)
        if not isinstance(ids, list) or not ids or len(ids) > 128:
            raise ValueError('invalid grasp episode split')
        for seed in ids:
            bounded_int(seed, 'seed', 0, 2**31 - 1)
        if len(set(ids)) != len(ids) or not set(ids) <= parsed.keys():
            raise ValueError('invalid grasp split membership')
        groups.append(set(ids))
    train_ids, held_ids = groups
    if train_ids & held_ids or train_ids | held_ids != parsed.keys():
        raise ValueError('grasp split must partition whole episodes')
    train_ids -= {e.seed for e in placement_holdout}
    held_ids -= {e.seed for e in placement_train}
    if not train_ids or not held_ids:
        raise ValueError('cross-goal exclusion left no grasp train or holdout episodes')
    return ([parsed[i] for i in sorted(train_ids)], [parsed[i] for i in sorted(held_ids)])


class LeftPlacementTeacher:
    """Collection-only Strategy; only bounded servo targets enter simulation.

    3900X/native MuJoCo 3.11.0, 2026-10-13: seed 17 qualified at 11.88 s,
    with whole-hull containment, release, 0.5 s settling and no chassis
    penetration. Forcing the closing axis horizontal lost the object; continue
    the same position-only IK branch instead. Each new episode must requalify.
    """

    def __init__(self, env):
        term = env.action_manager.get_term('claw')
        if term.cfg.training_stage != 'place-near':
            raise ValueError('placement teacher needs the explicit place-near objective')
        self.pickup = LeftPickupTeacher(env)
        self.env, self.term = env, term
        self.stage, self.entered = 'pickup', 0.
        self.reference = self.last_joints = self.withdraw = None

    def __call__(self, elapsed):
        evidence = self.term.evidence
        evidence.update()
        if self.stage == 'pickup' and bool(evidence.picked[0]):
            self.stage, self.entered = 'carry', elapsed
            self.reference = evidence.grip_point()[0].cpu().numpy().copy()
            self.last_joints = self.term.targets[0, :4].cpu().numpy().copy()
        if self.stage == 'pickup':
            return self.pickup(elapsed)
        obj = self.term.object_data('root_link_pos_w')[0].cpu().numpy()
        tray = self.env.scene['dropbin'].data.root_link_pos_w[0].cpu().numpy()
        grip = evidence.grip_point()[0].cpu().numpy()
        object_goal = tray + np.array([.02, -.005, .095 if self.stage == 'carry' else .070])
        goal = self.withdraw if self.stage in ('release', 'verify') else grip + object_goal - obj
        delta = goal - self.reference
        length = np.linalg.norm(delta)
        travel = (.03 if self.stage == 'carry' else .02) * self.env.step_dt
        self.reference = goal.copy() if length <= travel else self.reference + delta * travel / length
        seed = self.env.sim.data.qpos[0].cpu().numpy().copy()
        seed[self.pickup.ik.qadr] = self.last_joints
        self.last_joints = self.pickup.ik.solve(seed, self.reference).joints.copy()
        action = torch.zeros(1, 5, device=self.env.device)
        action[:, :4] = torch.as_tensor(
            (self.last_joints - self.term.targets[0, :4].cpu().numpy()) /
            (self.term.cfg.arm_speed * self.env.step_dt), device=self.env.device)
        action[:, 4] = -1. if self.stage in ('release', 'verify') else 1.
        if self.stage == 'carry' and np.linalg.norm(object_goal - obj) < .012:
            self.stage, self.entered = 'lower', elapsed
        elif (self.stage == 'lower' and np.linalg.norm((object_goal - obj)[:2]) < .012 and
              obj[2] - tray[2] < .075 and elapsed - self.entered >= .5):
            self.stage, self.entered = 'release', elapsed
            self.withdraw = grip.copy() + np.array([0, 0, .035])
        elif self.stage == 'release' and elapsed - self.entered >= 1.:
            self.stage, self.entered = 'verify', elapsed
        return bounded_action(action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grasp-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--holdout-episodes', type=int, default=4)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--position-jitter-mm', type=float, default=0.)
    parser.add_argument('--bounded-action-history', action='store_true')
    parser.add_argument('--grasp-demonstrations', type=Path,
                        help='optional compatible whole-episode pickup rehearsal dataset')
    args = parser.parse_args()
    bounded_int(args.episodes, 'episodes', 2, 128)
    bounded_int(args.holdout_episodes, 'holdout-episodes', 1, args.episodes - 1)
    bounded_int(args.eval_episodes, 'eval-episodes', 1, 128)
    bounded_int(args.epochs, 'epochs', 1, 5000)
    bounded_int(args.seed, 'seed', 0, 2**31 - 2128)
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend
    use_backend(resolve(backend='native', device='cpu', num_envs=1, cpu_threads=1))
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .env_cfg import env_cfg
    from .rl_cfg import agent_cfg
    from .runner import PickupRunner
    cfg = env_cfg(play=True, training_stage='place-near', object_size='standard',
                  position_jitter_mm=args.position_jitter_mm, freeze_actor_normalization=True,
                  bounded_action_history=args.bounded_action_history)
    cfg.auto_reset = False
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source_identity = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'place_trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
        'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'])).hexdigest()}
    env = ManagerBasedRlEnv(cfg, device='cpu')
    try:
        runner = PickupRunner(RslRlVecEnvWrapper(env), asdict(agent_cfg()), device='cpu')
        runner.transfer_grasp_actor(args.grasp_checkpoint)
        runner.save(str(args.output / 'transferred_actor.pt'))
        norm_before = {k: v.clone() for k, v in runner.alg.actor.obs_normalizer.state_dict().items()}
        episodes, reports = collect(env, LeftPlacementTeacher, seeds=range(args.seed, args.seed + args.episodes),
                                     max_steps=1500, goal='placed')
        train, holdout = split_episodes(episodes, args.holdout_episodes)
        grasp_train, grasp_holdout = [], []
        if args.grasp_demonstrations is not None:
            grasp_train, grasp_holdout = load_grasp_rehearsal(
                args.grasp_demonstrations, train, holdout, runner.contract(training_stage='grasp'))
        data = {'schema': 'jumper.pickup.placement-demonstrations/1', 'contract': runner.contract(),
                'episodes': [{'seed': e.seed, 'actor': e.observations, 'actions': e.actions} for e in episodes],
                'reports': reports, 'train_seeds': [e.seed for e in train], 'holdout_seeds': [e.seed for e in holdout]}
        data['grasp_rehearsal'] = {
            'source_contract': runner.contract(training_stage='grasp'),
            'train': [{'seed': e.seed, 'actor': e.observations, 'actions': e.actions} for e in grasp_train],
            'holdout': [{'seed': e.seed, 'actor': e.observations, 'actions': e.actions} for e in grasp_holdout]}
        dataset = args.output / 'demonstrations.pt'
        torch.save(data, dataset)
        fit = fit_actor(runner, train + grasp_train, holdout + grasp_holdout,
                        epochs=args.epochs, batch_size=512,
                        learning_rate=.001, seed=args.seed, update_normalization=False)
        if not all(torch.equal(v, runner.alg.actor.obs_normalizer.state_dict()[k]) for k, v in norm_before.items()):
            raise RuntimeError('placement fitting changed the retained input transform')
        provenance = {**source_identity, 'schema': 'jumper.pickup.placement-imitation/1',
                      'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                      'train_seeds': data['train_seeds'], 'holdout_seeds': data['holdout_seeds'],
                      'position_jitter_mm': args.position_jitter_mm,
                      'bounded_action_history': args.bounded_action_history,
                      'grasp_rehearsal_train_seeds': [e.seed for e in grasp_train],
                      'grasp_rehearsal_holdout_seeds': [e.seed for e in grasp_holdout],
                      'actor_normalization': 'frozen-source', 'fit': fit}
        checkpoint = args.output / 'placement_warm_start.pt'
        runner.save(str(checkpoint), {'placement_imitation': provenance})
        probe = TensorDict({'actor': holdout[0].observations},
                           batch_size=[len(holdout[0].observations)])
        with torch.no_grad():
            expected = runner.alg.actor(probe).clone()
        runner = PickupRunner(RslRlVecEnvWrapper(env), asdict(agent_cfg()), device='cpu')
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device='cpu')
        with torch.no_grad():
            if not torch.equal(expected, policy(probe)):
                raise RuntimeError('saved actor does not reproduce fitted placement outputs')
        placements = evaluate(env, policy, range(args.seed + 1000, args.seed + 1000 + args.eval_episodes))
        grasp_cfg = env_cfg(play=True, training_stage='grasp', object_size='standard', position_jitter_mm=0.)
        grasp_cfg.auto_reset = False
        grasp_env = ManagerBasedRlEnv(grasp_cfg, device='cpu')
        try:
            # Read-only performance check of the SAME saved actor in its original
            # grasp scope. This is not a cross-objective optimizer/clock resume.
            grasps = evaluate(grasp_env, policy, range(args.seed + 2000, args.seed + 2000 + args.eval_episodes))
        finally:
            grasp_env.close()
        report = {'checkpoint': str(checkpoint), 'demonstrations': reports, 'provenance': provenance,
                  'placements': placements, 'grasp_retention': grasps,
                  'placement_successes': sum(r['placed'] for r in placements),
                  'grasp_successes': sum(r['picked_up'] for r in grasps)}
        with (args.output / 'report.json').open('x') as target:
            json.dump(report, target, indent=2)
        print('PLACEMENT_RESULT', json.dumps({k: v for k, v in report.items()
                                              if k not in ('demonstrations', 'placements', 'grasp_retention', 'provenance')}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
