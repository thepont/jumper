"""Explicit synthetic detections, not a camera or an image detector."""
import math

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


class SimulatedDetections:
    """Sample at 10 Hz, deliver after 60 ms, expire after 250 ms.

    World estimates retain the capture pose, then are reprojected into the current
    base frame using ideal proprioception. Neither fresh simulator positions nor
    contact/success labels are returned as actor observations.
    """

    def __init__(self, env, *, noise_m, dropout):
        self.env, self.noise_m, self.dropout = env, noise_m, dropout
        shape = (env.num_envs, 2)
        self.pending = torch.zeros(*shape, 3, device=env.device)
        self.world = torch.zeros_like(self.pending)
        self.pending_size = torch.zeros(env.num_envs, 3, device=env.device)
        self.size = torch.zeros_like(self.pending_size)
        self.pending_valid = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.valid = torch.zeros_like(self.pending_valid)
        self.captured = torch.full(shape, -1e6, device=env.device)
        self.pending_at = torch.full((env.num_envs,), -1e6, device=env.device)
        self.due = torch.full((env.num_envs,), float('inf'), device=env.device)
        self.next_capture = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, ids):
        self.valid[ids] = False
        self.size[ids] = 0
        self.pending_valid[ids] = False
        self.captured[ids] = -1e6
        self.due[ids] = float('inf')
        self.next_capture[ids] = self.env.common_step_counter * self.env.step_dt
        self.last_step[ids] = -1

    def update(self):
        env = self.env
        step = env.common_step_counter
        fresh = self.last_step != step
        if not bool(fresh.any()):
            return
        now = step * env.step_dt
        ready = fresh & (self.due <= now + 1e-6)
        self.world[ready] = self.pending[ready]
        self.valid[ready] = self.pending_valid[ready]
        self.size[ready] = self.pending_size[ready]
        self.captured[ready] = self.pending_at[ready, None]
        self.due[ready] = float('inf')
        expose = fresh & (self.next_capture <= now + 1e-6)
        robot = env.scene['robot'].data
        task = env.action_manager.get_term('claw')
        truth = torch.stack([task.object_data('root_link_pos_w'), env.scene['dropbin'].data.root_link_pos_w], 1)
        quat = robot.root_link_quat_w[:, None].expand(-1, 2, -1)
        relative = quat_apply_inverse(quat, truth - robot.root_link_pos_w[:, None])
        radii = torch.stack([task.object_dimensions()[:, :2].norm(dim=1) / 2,
                             torch.full((env.num_envs,), task.cfg.tray_half, device=env.device)], 1)
        angular_radius = torch.asin((radii / relative[:, :, :2].norm(dim=-1).clamp(min=.001)).clamp(max=1))
        visible = ((relative[:, :, 0] > .05) &
                   (torch.atan2(relative[:, :, 1], relative[:, :, 0]).abs() <
                    math.radians(55) + angular_radius))
        detected = visible & (torch.rand_like(self.captured) >= self.dropout)
        noisy = relative + torch.randn_like(relative) * self.noise_m
        self.pending[expose] = (robot.root_link_pos_w[:, None] + quat_apply(quat, noisy))[expose]
        self.pending_valid[expose] = detected[expose]
        dimensions = task.object_dimensions()
        self.pending_size[expose] = (dimensions + torch.randn_like(dimensions) * self.noise_m).clamp(min=.001)[expose]
        self.pending_at[expose] = now
        self.due[expose] = now + .06
        self.next_capture[expose] = now + .1
        self.last_step[fresh] = step

    def usable(self):
        self.update()
        age = self.env.common_step_counter * self.env.step_dt - self.captured
        return self.valid & (age <= .25)

    def observation(self):
        valid = self.usable()
        robot = self.env.scene['robot'].data
        quat = robot.root_link_quat_w[:, None].expand(-1, 2, -1)
        positions = quat_apply_inverse(quat, self.world - robot.root_link_pos_w[:, None])
        positions = torch.where(valid[:, :, None], positions, 0.)
        age = (self.env.common_step_counter * self.env.step_dt - self.captured).clamp(0, .25)
        size = torch.where(valid[:, :1], self.size, 0.)
        return torch.cat((positions.flatten(1), valid.float(), age, size), dim=1)
