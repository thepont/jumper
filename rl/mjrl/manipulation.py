"""Small manipulation building blocks, independent of task rewards and sensors.

IK uses a private kinematic model state: it never teleports the simulated robot.
Its residual is part of the result, not an implied claim that a pose is reachable.
It checks joint limits, not collision-free paths or physical grasp success.
"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from scipy.optimize import least_squares


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


@dataclass(frozen=True)
class TargetEstimate:
    """A measured target in the robot base frame, metres and monotonic seconds.

    None at the caller represents no detection. A last-seen estimate is not a
    live detection: callers must check its age and confidence before moving.
    The producer owns calibration and coordinate transforms; this type does
    not turn simulator truth into a real measurement.
    """

    position: tuple[float, float, float]
    observed_at_s: float
    confidence: float

    def __post_init__(self):
        object.__setattr__(self, "position", tuple(_vector(self.position, 3, "position")))
        if not math.isfinite(self.observed_at_s) or self.observed_at_s < 0:
            raise ValueError("observation timestamp must be finite and nonnegative")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")

    def usable(self, now_s, *, max_age_s, min_confidence):
        if not math.isfinite(now_s) or now_s < 0:
            raise ValueError("current time must be finite and nonnegative")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("maximum observation age must be finite and positive")
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("minimum confidence must be in [0, 1]")
        return (0 <= now_s - self.observed_at_s <= max_age_s
                and self.confidence >= min_confidence)


@dataclass(frozen=True)
class IKResult:
    joints: np.ndarray
    error_m: float


class PointIK:
    """Bounded inverse kinematics for a body-fixed point and selected joints.

    All other joints, including the base, remain at the supplied measured
    configuration. Targets are world-frame metres. A controller must convert
    base-frame sensor estimates explicitly, gate the residual, and rate-limit
    the result before sending it to servos. This is not a motion planner.
    """

    def __init__(self, model, body_name, joint_names, local_point, *, margin=.01):
        self.model = model
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body < 0:
            raise ValueError(f"unknown body {body_name!r}")
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("joint names must be nonempty and unique")
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if any(j < 0 for j in ids):
            raise ValueError("unknown IK joint")
        for j in ids:
            if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                         mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError("IK joints must be scalar hinge or slide joints")
            if not model.jnt_limited[j]:
                raise ValueError("IK requires explicit joint limits")
        if not math.isfinite(margin) or not 0 < margin < 1:
            raise ValueError("joint-limit margin fraction must be in (0, 1)")
        self.qadr = model.jnt_qposadr[ids].copy()
        self.dadr = model.jnt_dofadr[ids].copy()
        ranges = model.jnt_range[ids]
        inset = (ranges[:, 1] - ranges[:, 0]) * margin / 2
        self.lower, self.upper = ranges[:, 0] + inset, ranges[:, 1] - inset
        self.local_point = _vector(local_point, 3, "local point")
        self.data = mujoco.MjData(model)

    def solve(self, qpos, target) -> IKResult:
        state = _vector(qpos, self.model.nq, "qpos")
        goal = _vector(target, 3, "target")
        self.data.qpos[:] = state

        def residual(joints):
            self.data.qpos[self.qadr] = joints
            mujoco.mj_kinematics(self.model, self.data)
            return (self.data.xpos[self.body]
                    + self.data.xmat[self.body].reshape(3, 3) @ self.local_point - goal)

        fit = least_squares(residual, np.clip(state[self.qadr], self.lower, self.upper),
                            bounds=(self.lower, self.upper), max_nfev=60,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        return IKResult(fit.x.copy(), float(np.linalg.norm(residual(fit.x))))
