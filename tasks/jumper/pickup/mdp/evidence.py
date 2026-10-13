"""Privileged physical evaluator; never passed to the deployed actor."""
import math

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from .safety import chassis_penetration


class PickupEvidence:
    def __init__(self, env, *, side, local_point, finger_ids, finger_sign,
                 open_angle, tray_half, floor_thickness, hull, objects,
                 max_chassis_penetration_m, grip_min_samples=None):
        self.env, self.side = env, side
        self.local_point = torch.tensor(local_point, device=env.device, dtype=torch.float32)
        prefix = 'LF' if side == 'left' else 'RF'
        self.palm = env.scene['robot'].find_bodies([f'{prefix}_palm_link'])[0][0]
        self.finger_ids, self.finger_sign = finger_ids, finger_sign
        self.open_angle, self.tray_half, self.floor_thickness = open_angle, tray_half, floor_thickness
        self.hull = torch.tensor(hull, device=env.device, dtype=torch.float32)
        self.objects = objects
        self.max_chassis_penetration_m = max_chassis_penetration_m
        self.grip_min_samples = (math.ceil(.2 / env.step_dt) if grip_min_samples is None
                                 else grip_min_samples)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.reset_step = torch.zeros_like(self.last_step)
        self.grip_steps = torch.zeros_like(self.last_step)
        self.settle_steps = torch.zeros_like(self.last_step)
        self.picked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.success = torch.zeros_like(self.picked)
        self.initial_clear = torch.ones_like(self.picked)
        self.valid_trajectory = torch.ones_like(self.picked)
        self.initial_outside = torch.zeros_like(self.picked)
        self.initialized = torch.zeros_like(self.picked)
        self.anchor = torch.zeros(env.num_envs, len(hull), 3, device=env.device)
        self.inside = torch.zeros_like(self.picked)
        self.pinched = torch.zeros_like(self.picked)
        self.clearance = torch.zeros(env.num_envs, device=env.device)
        self.released = torch.zeros_like(self.picked)

    def reset(self, ids):
        self.last_step[ids] = -1
        self.reset_step[ids] = self.env.common_step_counter
        for value in (self.grip_steps, self.settle_steps):
            value[ids] = 0
        for value in (self.picked, self.success, self.initial_outside, self.initialized,
                      self.inside, self.pinched, self.released):
            value[ids] = False
        self.initial_clear[ids] = True
        self.valid_trajectory[ids] = True

    def grip_point(self):
        data = self.env.scene['robot'].data
        return data.body_link_pos_w[:, self.palm] + quat_apply(
            data.body_link_quat_w[:, self.palm], self.local_point.expand(self.env.num_envs, -1))

    def update(self):
        env = self.env
        fresh = self.last_step != env.common_step_counter
        if not bool(fresh.any()):
            return
        tray = env.scene['dropbin'].data
        scales = self.objects.scales[self.objects.object_index, None, None]
        points = quat_apply(self.objects.object_data('root_link_quat_w')[:, None].expand(-1, len(self.hull), -1),
                            self.hull[None] * scales) + self.objects.object_data('root_link_pos_w')[:, None]
        in_bin = quat_apply_inverse(tray.root_link_quat_w[:, None].expand(-1, len(self.hull), -1),
                                    points - tray.root_link_pos_w[:, None])
        robot_contacts = torch.stack([env.scene[f'pickup_robot_contact_{i}'].data.found.any(dim=1)
                                      for i in range(len(self.objects.object_names))], 1)
        rows = torch.arange(env.num_envs, device=env.device)
        contact = robot_contacts[rows, self.objects.object_index]
        initialize = fresh & ~self.initialized
        outer = self.tray_half + self.floor_thickness
        outside = ((in_bin[:, :, :2].amin(dim=1) > outer) |
                   (in_bin[:, :, :2].amax(dim=1) < -outer)).any(dim=1)
        self.initial_outside[initialize] = outside[initialize]
        self.initialized[initialize] = True
        elapsed = (env.common_step_counter - self.reset_step) * env.step_dt
        settling = fresh & (elapsed <= 1.)
        self.initial_clear[settling] &= ~contact[settling]
        self.valid_trajectory[fresh] &= (chassis_penetration(env)[fresh] <=
                                         self.max_chassis_penetration_m)
        eligible = (self.initial_outside & self.initial_clear & self.valid_trajectory &
                    (elapsed > 1.))
        grips = torch.stack([env.scene[f'pickup_grip_{i}'].data.found.gt(0).all(dim=1)
                              for i in range(len(self.objects.object_names))], 1)
        self.pinched[:] = grips[rows, self.objects.object_index]
        self.clearance[:] = points[:, :, 2].amin(dim=1) - env.scene.env_origins[:, 2]
        total_contacts = torch.stack([env.scene[f'pickup_object_contact_{i}'].data.found.sum(dim=1)
                                       for i in range(len(self.objects.object_names))], 1)
        claw_contacts = torch.stack([env.scene[f'pickup_claw_contact_{i}'].data.found.sum(dim=1)
                                      for i in range(len(self.objects.object_names))], 1)
        unsupported = total_contacts[rows, self.objects.object_index] <= claw_contacts[rows, self.objects.object_index]
        # The tray rim is 24 mm tall: floor clearance alone can certify an
        # object still supported by that rim. Require ALL contacts to belong
        # to the carried palm subtree, excluding tray/terrain/other legs.
        grasp = eligible & self.pinched & unsupported & (self.clearance > .015)
        self.grip_steps[fresh] = torch.where(grasp, self.grip_steps + 1, 0)[fresh]
        self.picked[fresh] |= self.grip_steps[fresh] >= self.grip_min_samples
        self.inside[:] = ((in_bin[:, :, :2].abs() < self.tray_half).all(dim=(1, 2)) &
                          (in_bin[:, :, 2].amin(dim=1) > -.004) &
                          (in_bin[:, :, 2].amin(dim=1) < self.floor_thickness + .008))
        angle = env.scene['robot'].data.joint_pos[:, self.finger_ids[0]] * self.finger_sign
        self.released[:] = (angle < self.open_angle + .15) & ~contact
        valid = eligible & self.picked & self.inside & self.released
        drift = (points - self.anchor).norm(dim=-1).amax(dim=1)
        restart = fresh & valid & ((self.settle_steps == 0) | (drift >= .003))
        self.anchor[restart] = points[restart]
        self.settle_steps[restart] = 0
        self.settle_steps[fresh] = torch.where(valid, self.settle_steps + 1, 0)[fresh]
        self.success[fresh] = self.settle_steps[fresh] >= math.ceil(.5 / env.step_dt) + 1
        self.last_step[fresh] = env.common_step_counter

    def invalid_fixture(self):
        self.update()
        elapsed = (self.env.common_step_counter - self.reset_step) * self.env.step_dt
        return (elapsed >= 1.) & ~(self.initial_clear & self.initial_outside)
