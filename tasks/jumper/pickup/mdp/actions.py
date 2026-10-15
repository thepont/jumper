"""Five learned controls, with bounded servo targets and a perception gate."""
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from tasks.jumper.five_foot.mdp.gripper import squeeze_limited

from .evidence import PickupEvidence
from .perception import SimulatedDetections


@dataclass(kw_only=True)
class ClawActionCfg(ActionTermCfg):
    joint_names: tuple[str, ...]
    side: str
    arm_speed: float
    open_angle: float
    closed_angle: float
    noise_m: float
    dropout: float
    local_point: tuple[float, ...]
    tray_half: float
    floor_thickness: float
    hull: tuple[tuple[float, ...], ...]
    object_names: tuple[str, ...]
    object_scales: tuple[float, ...]
    object_size: tuple[float, ...]
    max_chassis_penetration_m: float
    training_stage: str = 'place'
    grasp_start: str = 'near'
    grip_min_samples: int | None = None
    freeze_actor_normalization: bool = False
    bounded_action_history: bool = False
    finger_close_speed: float | None = None
    # Opt-in simulation composition seam; default training observations stay unchanged.
    detections_factory: Callable[..., SimulatedDetections] | None = None

    def __post_init__(self):
        rate = self.finger_close_speed
        if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float)) or
                                 not math.isfinite(rate) or rate <= 0):
            raise ValueError('finger-close-speed must be finite and positive, or absent')

    def build(self, env):
        return ClawActions(self, env)


class ClawActions(ActionTerm):
    def __init__(self, cfg, env):
        factory = SimulatedDetections if cfg.detections_factory is None else cfg.detections_factory
        if not callable(factory):
            raise ValueError('detections factory must be callable or absent')
        super().__init__(cfg, env)
        self.joints = self._entity.find_joints(list(cfg.joint_names), preserve_order=True)[0]
        self.finger_sign = 1. if cfg.side == 'left' else -1.
        self.targets = self._entity.data.joint_pos[:, self.joints].clone()
        self._finger_applied = self.targets[:, 4:5].clone()
        self._raw = torch.zeros_like(self.targets)
        self.limits = self._entity.data.joint_pos_limits[:, self.joints]
        self.object_index = torch.full((env.num_envs,), 1, device=env.device, dtype=torch.long)
        self.object_names = cfg.object_names
        self.scales = torch.tensor(cfg.object_scales, device=env.device)
        self.object_size = torch.tensor(cfg.object_size, device=env.device)
        self.detections = factory(env, noise_m=cfg.noise_m, dropout=cfg.dropout)
        if not isinstance(self.detections, SimulatedDetections) or self.detections.env is not env:
            raise ValueError('detections factory must return a compatible sensor for this environment')
        self.evidence = PickupEvidence(
            env, side=cfg.side, local_point=cfg.local_point, finger_ids=self.joints[-1:],
            finger_sign=self.finger_sign, open_angle=cfg.open_angle,
            tray_half=cfg.tray_half, floor_thickness=cfg.floor_thickness, hull=cfg.hull,
            objects=self, max_chassis_penetration_m=cfg.max_chassis_penetration_m,
            grip_min_samples=cfg.grip_min_samples)

    @property
    def action_dim(self):
        return 5

    @property
    def raw_action(self):
        return self._raw

    def object_data(self, field):
        values = torch.stack([getattr(self._env.scene[n].data, field) for n in self.object_names], 1)
        return values[torch.arange(self.num_envs, device=self.device), self.object_index]

    def object_dimensions(self):
        return self.object_size[None] * self.scales[self.object_index, None] * 2

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.targets[ids] = self._entity.data.joint_pos[ids][:, self.joints]
        self._finger_applied[ids] = self.targets[ids, 4:5]
        self.detections.reset(ids)
        self.evidence.reset(ids)

    def process_actions(self, actions):
        if actions.shape != self._raw.shape or not bool(torch.isfinite(actions).all()):
            raise ValueError('claw actions must be finite with shape [num_envs, 5]')
        self._raw[:] = actions
        bounded = actions.clamp(-1, 1)
        usable = self.detections.usable()
        detected = usable[:, 0] if self.cfg.training_stage == 'grasp' else usable.all(dim=1)
        ready = (detected &
                 ((self._env.common_step_counter - self.evidence.reset_step) * self._env.step_dt > 1.))
        arm = self.targets[:, :4] + bounded[:, :4] * self.cfg.arm_speed * self._env.step_dt
        arm = arm.clamp(self.limits[:, :4, 0], self.limits[:, :4, 1])
        self.targets[:, :4] = torch.where(ready[:, None], arm, self.targets[:, :4])
        finger = self.finger_sign * (self.cfg.open_angle + (bounded[:, 4] + 1) / 2 *
                                     (self.cfg.closed_angle - self.cfg.open_angle))
        self.targets[:, 4] = torch.where(ready, finger, self.targets[:, 4])

    def apply_actions(self):
        self._entity.set_joint_position_target(self.targets[:, :4], joint_ids=self.joints[:4])
        angle = self._entity.data.joint_pos[:, self.joints[-1:]]
        speed = self._entity.data.joint_vel[:, self.joints[-1:]]
        limited = self.finger_sign * squeeze_limited(
            self.finger_sign * self.targets[:, 4:5], self.finger_sign * angle,
            speed=self.finger_sign * speed)
        if self.cfg.finger_close_speed is not None:
            # Positive canonical motion closes either mirrored claw. Limit the
            # emitted target at physics cadence, not once per policy update.
            # Store the squeezed output, so stalled contact cannot wind up lead.
            ceiling = (self.finger_sign * self._finger_applied +
                       self.cfg.finger_close_speed * self._env.physics_dt)
            limited = self.finger_sign * torch.minimum(self.finger_sign * limited, ceiling)
        self._finger_applied[:] = limited
        self._entity.set_joint_position_target(limited, joint_ids=self.joints[-1:])
