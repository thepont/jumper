"""Native CPU experimental central-load shaping; not a pickup success predicate.

Each loaded contact is scored before averaging, so opposite edge contacts cannot
cancel into a fabricated central bonus. Forces are simulator post-forward loads,
not hardware pressure readings. Existing observations and physics are unchanged.
"""
import math
import numpy as np
import torch


def central_load_score(records, half_extents, *, width_fraction=.35):
    half = np.asarray(half_extents, dtype=float)
    if (half.shape != (3,) or not np.isfinite(half).all() or (half <= 0).any() or
            type(width_fraction) not in (int, float) or not math.isfinite(width_fraction) or
            not 0 < width_fraction <= 1 or len(records) > 4096):
        raise ValueError('central score needs positive finite extents and bounded width/contacts')
    totals, weighted = np.zeros(2), np.zeros(2)
    for pad, point, force, distance, active in records:
        point = np.asarray(point, dtype=float)
        if (type(pad) is not int or pad not in (0, 1) or point.shape != (3,) or
                not np.isfinite(point).all() or not np.isfinite([force, distance]).all() or
                type(active) is not bool):
            raise ValueError('contact must have a pad, finite object-frame point/load/distance and activity')
        if not active or distance > 0 or force <= 1e-6:
            continue
        fraction = float(np.max(np.abs(point[[0, 2]]) / half[[0, 2]]))
        totals[pad] += force
        weighted[pad] += force * math.exp(-.5 * (fraction / width_fraction)**2)
    if not np.isfinite(totals).all() or not np.isfinite(weighted).all():
        raise ValueError('central contact accumulation overflowed')
    return float(np.min(weighted / totals)) if (totals > 0).all() else 0.


def native_central_scores(env):
    import mujoco
    if torch.device(env.device).type != 'cpu' or not hasattr(env.sim, 'env_mjdata'):
        raise ValueError('central-load experiment requires native CPU simulation')
    term = env.action_manager.get_term('claw')
    if term.cfg.side != 'left':
        raise ValueError('central-load experiment is left-claw only')
    values = []
    for world in range(env.num_envs):
        model, data = env.sim.env_mjdata(world)
        name = term.object_names[int(term.object_index[world])]
        body = model.body(name + '/' + ('soap' if name == 'soap' else name)).id
        pads = [model.geom(f'robot/LF_{p}_grip_insert_link_meshcol').id
                for p in ('palm', 'finger')]
        rotation = data.xmat[body].reshape(3, 3)
        records = []
        for cid, contact in enumerate(data.contact[:data.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            for pad, geom in enumerate(pads):
                if ((a == geom and model.geom_bodyid[b] == body) or
                        (b == geom and model.geom_bodyid[a] == body)):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, cid, force)
                    point = rotation.T @ (contact.pos - data.xpos[body])
                    records.append((pad, point, float(force[0]), float(contact.dist),
                                    bool(contact.efc_address >= 0)))
        half = np.asarray(term.cfg.object_size) * float(term.scales[term.object_index[world]])
        values.append(central_load_score(records, half))
    return torch.tensor(values, dtype=torch.float32, device=env.device)


class NativeCentralPotential:
    """Discount-correct auxiliary potential; no passive grip-income stream."""

    def __init__(self, cfg, env):
        self.previous = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.previous[slice(None) if env_ids is None else env_ids] = 0.

    def __call__(self, env, *, discount):
        if type(discount) not in (int, float) or not 0 < discount < 1:
            raise ValueError('central potential discount must be in (0, 1)')
        term = env.action_manager.get_term('claw')
        evidence = term.evidence
        evidence.update()
        value = native_central_scores(env)
        eligible = (evidence.initial_clear & evidence.initial_outside &
                    evidence.valid_trajectory & ~evidence.self_contacted &
                    ((env.common_step_counter - evidence.reset_step) * env.step_dt > 1.))
        value = torch.where(eligible & ~env.reset_terminated, value, 0.)
        result = (discount * value - self.previous) / env.step_dt
        self.previous[:] = value
        return result


def central_pickup_bonus(env):
    """Additional objective preference only on the existing physical pickup witness."""
    from .mdp.terms import grasp_completed
    return native_central_scores(env) * grasp_completed(env)


class NativeSelfClearanceFailure:
    """Experiment-only mask-independent rejection, including masked arm pairs."""

    def __init__(self, cfg, env):
        from mjrl.clearance import GeometryClearance
        from .collision import nonadjacent_arm_pairs
        if torch.device(env.device).type != 'cpu' or not hasattr(env.sim, 'env_mjdata'):
            raise ValueError('explicit self-clearance experiment requires native CPU')
        self.checkers = []
        for world in range(env.num_envs):
            model, _ = env.sim.env_mjdata(world)
            self.checkers.append(GeometryClearance(model, nonadjacent_arm_pairs(model), margin_m=.001))

    def __call__(self, env):
        failed = [not checker.check(env.sim.data.qpos[i].cpu().numpy()).passed
                  for i, checker in enumerate(self.checkers)]
        failed = torch.tensor(failed, dtype=torch.bool, device=env.device)
        # Reuse the task's reset-owned latch, so same-frame success/reward fails.
        env.action_manager.get_term('claw').evidence.self_contacted |= failed
        return failed
