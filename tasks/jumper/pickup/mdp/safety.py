"""Privileged chassis witnesses, shared by evaluation and reporting only."""
import torch


def robot_self_contact(env):
    """Reject detected robot/robot touch, including resolved substep penetration.

    This does not certify positive clearance or detect mask-excluded pairs.
    Unmatched history slots are zero, so they must not be treated as contacts.
    """
    data = env.scene['pickup_self_contact'].data
    return data.found.gt(0).any(dim=1) | data.dist_history.lt(0).any(dim=(1, 2))


def chassis_penetration(env):
    """Deepest contact across both arms and every substep of this control frame.

    Sensors use mindist reduction. Non-contact distance slots are zero, so only
    negative signed distances contribute. Do not use large-cutoff mesh distance
    queries: these can return zero for separated geoms on MuJoCo 3.11.0.
    """
    depths = []
    for side in ('LF', 'RF'):
        data = env.scene[f'pickup_chassis_{side}'].data
        depths.append((-data.dist).clamp_min(0).amax(dim=1))
        depths.append((-data.dist_history).clamp_min(0).amax(dim=(1, 2)))
    return torch.stack(depths, dim=1).amax(dim=1)


def chassis_contacts(env):
    """Current matched contact count, not an actor observation."""
    return sum(env.scene[f'pickup_chassis_{side}'].data.found.sum(dim=1)
               for side in ('LF', 'RF'))
