"""Opt-in synthetic pose adapter for grasp advice, not an implemented camera.

Reuse the existing 10 Hz/60 ms/dropout/expiry position producer unchanged.
Orientation is captured at the same instant and delivered with that packet.
Its independent RNG avoids changing existing position observations or NN inputs.
This assumes an orientation estimator; real sensing/calibration is still absent.
"""
import math

import torch
from mjlab.utils.lab_api.math import quat_conjugate, quat_mul
from mjrl.manipulation import TargetEstimate

from .grasp_advisor import ObjectPose
from .mdp.perception import SimulatedDetections


class PoseDetections(SimulatedDetections):
    def __init__(self, env, *, noise_m, dropout, orientation_noise_rad, seed):
        for value, name, high in ((noise_m, 'position noise', .1),
                                   (dropout, 'dropout', 1.),
                                   (orientation_noise_rad, 'orientation noise', .5)):
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= high):
                raise ValueError(f'{name} must be finite in [0, {high}]')
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError('pose sensor seed must be an integer in [0, 2**31)')
        super().__init__(env, noise_m=noise_m, dropout=dropout)
        self.orientation_noise_rad = orientation_noise_rad
        self.rng = torch.Generator(device=env.device).manual_seed(seed)
        self.pending_orientation = torch.zeros(env.num_envs, 4, device=env.device)
        self.pending_orientation[:, 0] = 1.
        self.orientation = self.pending_orientation.clone()

    def reset(self, ids):
        super().reset(ids)
        self.orientation[ids] = 0.
        self.orientation[ids, 0] = 1.
        self.pending_orientation[ids] = self.orientation[ids]

    def update(self):
        now = self.env.common_step_counter * self.env.step_dt
        fresh = self.last_step != self.env.common_step_counter
        ready = fresh & (self.due <= now + 1e-6)
        expose = fresh & (self.next_capture <= now + 1e-6)
        self.orientation[ready] = self.pending_orientation[ready]
        super().update()
        if bool(expose.any()):
            task = self.env.action_manager.get_term('claw')
            # The only privileged orientation read is at sensor capture, never
            # in the planner and never at delayed packet delivery.
            truth = task.object_data('root_link_quat_w')
            vector = torch.randn(self.env.num_envs, 3, device=self.env.device,
                                 generator=self.rng) * self.orientation_noise_rad
            angle = vector.norm(dim=1, keepdim=True)
            xyz = vector * (torch.sin(angle/2) / angle.clamp_min(1e-12))
            noise = torch.cat((torch.cos(angle/2), xyz), dim=1)
            self.pending_orientation[expose] = quat_mul(truth, noise)[expose]

    def estimate(self, world=0):
        if type(world) is not int or not 0 <= world < self.env.num_envs:
            raise ValueError('pose world index is outside the sensor batch')
        if not bool(self.usable()[world, 0]):
            return None
        robot = self.env.scene['robot'].data
        relative = quat_mul(quat_conjugate(robot.root_link_quat_w), self.orientation)
        position = tuple(float(v) for v in self.observation()[world, :3])
        # Availability confidence 1 is synthetic validity, not a calibrated
        # probability that a real camera pose is correct.
        target = TargetEstimate(position, float(self.captured[world, 0]), 1.)
        return ObjectPose(target, tuple(float(v) for v in relative[world]),
                          tuple(float(v) for v in self.size[world]))
