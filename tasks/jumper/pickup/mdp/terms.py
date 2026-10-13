"""Actor inputs, privileged rewards, reset layout and honest outcome metrics."""
import torch


def claw(env):
    return env.action_manager.get_term('claw')


def proprioception(env):
    term = claw(env)
    data = env.scene['robot'].data
    history = (term.raw_action.clamp(-1, 1) if
               getattr(getattr(term, 'cfg', None), 'bounded_action_history', False) else term.raw_action)
    return torch.cat((data.joint_pos[:, term.joints], data.joint_vel[:, term.joints],
                      term.targets, history, data.projected_gravity_b,
                      data.root_link_ang_vel_b), dim=1)


def detections(env):
    term = claw(env)
    # Size comes through the same delayed/noisy producer and validity mask.
    return term.detections.observation()


def privileged_state(env):
    from mjlab.utils.lab_api.math import quat_apply_inverse
    term = claw(env)
    term.evidence.update()
    robot = env.scene['robot'].data
    object_pos = term.object_data('root_link_pos_w')
    tray_pos = env.scene['dropbin'].data.root_link_pos_w
    return torch.cat((quat_apply_inverse(robot.root_link_quat_w, object_pos - robot.root_link_pos_w),
                      quat_apply_inverse(robot.root_link_quat_w, tray_pos - robot.root_link_pos_w),
                      term.object_data('root_link_lin_vel_w'),
                      term.object_data('root_link_ang_vel_w'),
                      term.object_data('root_link_quat_w'),
                      torch.stack((term.evidence.pinched.float(), term.evidence.picked.float(),
                                   term.evidence.inside.float(), term.evidence.released.float()), 1)), 1)


def reward(env, *, kind, grasp_offset, reach_std_m, carry_std_m, lift_target_m, carry_height_m):
    term = claw(env)
    evidence = term.evidence
    evidence.update()
    obj = term.object_data('root_link_pos_w')
    if kind == 'reach':
        offset = torch.tensor(grasp_offset, device=env.device, dtype=obj.dtype) * term.scales[term.object_index, None]
        return torch.exp(-(evidence.grip_point() - obj - offset).norm(dim=1) / reach_std_m) * ~evidence.picked
    if kind == 'grip':
        return evidence.pinched.float()
    if kind == 'lift':
        return (evidence.clearance / lift_target_m).clamp(0, 1) * evidence.pinched
    if kind == 'carry':
        goal = env.scene['dropbin'].data.root_link_pos_w.clone()
        goal[:, 2] += carry_height_m
        return torch.exp(-(obj - goal).norm(dim=1) / carry_std_m) * evidence.picked * evidence.pinched
    if kind == 'release':
        return (evidence.picked & evidence.inside & evidence.released).float()
    if kind == 'placed':
        if getattr(getattr(term, 'cfg', None), 'training_stage', 'place') == 'place-near':
            return placement_completed(env).float()
        return evidence.success.float()
    if kind == 'picked':
        return grasp_completed(env).float()
    raise ValueError(f'unknown pickup reward {kind!r}')


class PotentialReward:
    """Discount-correct progress shaping: waiting cannot farm grip income.

    Managers scale rewards by dt, so divide by dt here. True terminal states
    have zero potential; timeout states retain potential for PPO bootstrapping.
    The discount is the SAME task-owned constant as PPO's gamma.
    """

    def __init__(self, cfg, env):
        self.previous = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.previous[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, *, kind, grasp_offset, discount,
                 reach_std_m, carry_std_m, lift_target_m, carry_height_m):
        evidence = claw(env).evidence
        value = reward(env, kind=kind, grasp_offset=grasp_offset,
                       reach_std_m=reach_std_m, carry_std_m=carry_std_m,
                       lift_target_m=lift_target_m, carry_height_m=carry_height_m)
        eligible = (evidence.initial_clear & evidence.initial_outside &
                    ((env.common_step_counter - evidence.reset_step) * env.step_dt > 1.))
        value = torch.where(eligible & ~env.reset_terminated, value, 0.)
        result = (discount * value - self.previous) / env.step_dt
        self.previous[:] = value
        return result


def grasp_completed(env):
    """Qualifying pickup pays nothing on a simultaneous physical failure.

    Grasp completion is evaluated after the existing failure terms. It is also
    reused after the termination pass for metrics and the one-shot bonus.
    """
    evidence = claw(env).evidence
    evidence.update()
    failed = torch.stack([env.termination_manager.get_term(name)
                           for name in ('fell_over', 'too_low', 'invalid_fixture', 'chassis_intrusion')], 1).any(dim=1)
    return evidence.picked & evidence.valid_trajectory & ~evidence.invalid_fixture() & ~failed


def placement_completed(env):
    return grasp_completed(env) & claw(env).evidence.success


def outcome(env, *, kind):
    term = claw(env)
    evidence = term.evidence
    evidence.update()
    if kind == 'invalid':
        return evidence.invalid_fixture()
    if kind == 'chassis_intrusion':
        return ~evidence.valid_trajectory
    if kind == 'picked' and getattr(getattr(term, 'cfg', None), 'training_stage', 'place') in ('grasp', 'place-near'):
        return grasp_completed(env)
    if kind == 'success' and getattr(getattr(term, 'cfg', None), 'training_stage', 'place') == 'place-near':
        return placement_completed(env)
    return getattr(evidence, kind)


def metric(env, *, kind):
    if kind == 'chassis_penetration':
        from .safety import chassis_penetration
        return chassis_penetration(env)
    if kind == 'chassis_contacts':
        from .safety import chassis_contacts
        return chassis_contacts(env)
    if kind == 'reach_distance':
        term = claw(env)
        # Distance to the object centre, not a hidden IK target.
        return (term.evidence.grip_point() - term.object_data('root_link_pos_w')).norm(dim=1)
    if kind == 'size_scale':
        term = claw(env)
        return term.scales[term.object_index]
    if kind == 'clearance':
        term = claw(env)
        term.evidence.update()
        return term.evidence.clearance
    return outcome(env, kind=kind).float()


def reset_layout(env, env_ids, *, jitter_m, side, object_names, scales,
                 object_size, object_xy, bin_xy, tray_half, wall_thickness, object_size_mode):
    from mjlab.envs.mdp.events import resolve_env_ids
    ids = resolve_env_ids(env, env_ids)
    count = len(ids)
    selected = (torch.randint(len(scales), (count,), device=env.device)
                if object_size_mode == 'mixed' else
                torch.full((count,), ('small', 'standard', 'large').index(object_size_mode),
                           device=env.device, dtype=torch.long))
    term = claw(env)
    term.object_index[ids] = selected
    origins = env.scene.env_origins[ids]
    jitter = (torch.rand(count, 2, device=env.device) * 2 - 1) * jitter_m
    object_pos = torch.tensor(object_xy, device=env.device, dtype=origins.dtype) + jitter
    bin_pos = torch.tensor(bin_xy, device=env.device, dtype=origins.dtype) + (torch.rand_like(jitter) * 2 - 1) * jitter_m / 4
    selected_scale = term.scales[selected]
    # Negative X jitter and a larger near face put the bar into the stowed
    # palm pad while the robot settles. Keep a 10 mm forward margin and vary
    # X outwards, compensating for physical width; do not silently accept those
    # invalid starts (native four-world probe, 2026-10-13).
    object_pos[:, 0] = object_xy[0] + .010 + jitter[:, 0].abs() + object_size[0] * (selected_scale - 1)
    # Keep the whole upright object clear of the tray's outer footprint. Sampling
    # an object already inside the goal would manufacture success evidence.
    bin_pos[:, 1] = torch.maximum(bin_pos[:, 1], object_pos[:, 1] + tray_half +
                                 wall_thickness + object_size[1] * selected_scale + .005)
    sign = 1. if side == 'left' else -1.
    object_pos[:, 1] *= sign
    bin_pos[:, 1] *= sign
    for i, name in enumerate(object_names):
        prop = env.scene[name]
        state = prop.data.default_root_state[ids].clone()
        state[:, :3] = origins
        state[:, 0] += 2. + i * .3
        state[:, 2] += object_size[2] * scales[i] + .002
        active = selected == i
        state[active, :2] = origins[active, :2] + object_pos[active]
        state[:, 7:] = 0
        prop.write_root_state_to_sim(state, env_ids=ids)
    tray = env.scene['dropbin']
    state = tray.data.default_root_state[ids].clone()
    state[:, :3] = origins
    state[:, :2] += bin_pos
    state[:, 7:] = 0
    # Entity wraps static props as mocap bodies: reset their pose, never give
    # a tray a free joint or move it during a rollout.
    tray.write_mocap_pose_to_sim(state[:, :7], env_ids=ids)
