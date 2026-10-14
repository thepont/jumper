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
    axis_error_rad: float | None = None


@dataclass(frozen=True)
class JointPathResult:
    """Discrete collision-query evidence, not a continuous/hardware guarantee."""

    found: bool
    waypoints: tuple[np.ndarray, ...]
    checks: int
    reason: str


def shortcut_joint_path(waypoints, is_valid, *, max_checks=10000,
                        resolution_rad=.025) -> JointPathResult:
    """Remove intermediate stops only across checked collision-free segments.

    Uses the caller's existing collision Strategy. Greedy farthest-visible
    shortcuts preserve endpoints, sample each edge at the specified resolution,
    and fail closed on exhausted budgets. This is discrete geometric evidence,
    not a claim about actuator tracking or continuous collision clearance.
    """
    path = np.asarray(waypoints, dtype=float)
    if (path.ndim != 2 or not 2 <= len(path) <= 10000 or
            not 1 <= path.shape[1] <= 32 or not np.isfinite(path).all()):
        raise ValueError("path requires 2..10000 finite vectors of 1..32 joints")
    path = path.copy()
    if (isinstance(max_checks, bool) or not isinstance(max_checks, (int, np.integer)) or
            not 2 <= max_checks <= 1000000 or isinstance(resolution_rad, (bool, np.bool_)) or
            not isinstance(resolution_rad, (int, float, np.integer, np.floating)) or
            not math.isfinite(resolution_rad) or
            not 0 < resolution_rad <= math.pi or not callable(is_valid)):
        raise ValueError("shortcut needs a bounded query budget, resolution and collision strategy")
    checks = 0

    def query(q):
        nonlocal checks
        checks += 1
        verdict = is_valid(q.copy())
        if not isinstance(verdict, (bool, np.bool_)):
            raise ValueError("collision strategy must return explicit boolean evidence")
        return bool(verdict)

    if not query(path[0]):
        return JointPathResult(False, (), checks, "invalid start")
    chosen, current = [path[0].copy()], 0
    while current < len(path) - 1:
        selected = None
        for candidate in range(len(path) - 1, current, -1):
            extent = float(np.max(np.abs(path[candidate] - path[current])))
            if not math.isfinite(extent) or extent / resolution_rad > max_checks - checks:
                return JointPathResult(False, (), checks, "shortcut query budget exhausted")
            count = max(1, math.ceil(extent / resolution_rad))
            if checks + count > max_checks:
                return JointPathResult(False, (), checks, "shortcut query budget exhausted")
            if all(query(path[current] + (path[candidate] - path[current]) * (i / count))
                   for i in range(1, count + 1)):
                selected = candidate
                break
        if selected is None:
            return JointPathResult(False, (), checks, "no valid continuation")
        chosen.append(path[selected].copy())
        current = selected
    return JointPathResult(True, tuple(chosen), checks, "checked shortcut corridor")


def plan_joint_path(start, goal, lower, upper, is_valid, *, seed=0,
                    max_samples=800, max_checks=10000, step_rad=.18,
                    resolution_rad=.025) -> JointPathResult:
    """Bounded seeded RRT with a caller-owned collision Strategy.

    The callback judges a complete candidate joint vector using its own scene
    and uncertainty model. Edges are sampled no farther apart than resolution
    in any joint; thin obstacles or servo/base tracking error need additional
    clearance at that boundary. Every query, including endpoints, consumes the
    hard budget. No simulator state, collision masks or target poses are written.
    """
    initial = np.asarray(start, dtype=float)
    if initial.ndim != 1 or not 1 <= len(initial) <= 32:
        raise ValueError("start must contain between 1 and 32 joints")
    initial = _vector(initial, len(initial), "start")
    final = _vector(goal, len(initial), "goal")
    lo = _vector(lower, len(initial), "lower bounds")
    hi = _vector(upper, len(initial), "upper bounds")
    if not (hi > lo).all() or not np.isfinite(hi - lo).all():
        raise ValueError("joint bounds must have finite positive width")
    if ((initial < lo) | (initial > hi) | (final < lo) | (final > hi)).any():
        raise ValueError("path endpoints must be within joint bounds")
    for value, minimum, maximum, name in (
        (seed, 0, 2**32 - 1, "seed"), (max_samples, 1, 10000, "sample budget"),
        (max_checks, 2, 1000000, "collision-query budget"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if (not math.isfinite(step_rad) or not math.isfinite(resolution_rad) or
            not 0 < resolution_rad <= step_rad <= math.pi):
        raise ValueError("require 0 < resolution <= step <= pi")
    if not callable(is_valid):
        raise ValueError("collision strategy must be callable")
    checks = 0

    def query(q):
        nonlocal checks
        if checks >= max_checks:
            return False
        checks += 1
        return bool(is_valid(q.copy()))

    def edge(a, b):
        extent = float(np.abs(b - a).max())
        # Reject a segment too large to check rather than allocate/run unbounded.
        if extent / resolution_rad > max_checks - checks:
            return False
        count = max(1, math.ceil(extent / resolution_rad))
        return all(query(a + (b - a) * (i / count)) for i in range(1, count + 1))

    if not query(initial) or not query(final):
        return JointPathResult(False, (), checks, "invalid endpoint")
    if edge(initial, final):
        return JointPathResult(True, (initial, final), checks, "direct corridor")
    rng = np.random.default_rng(seed)
    nodes, parents = [initial], [-1]
    for sample in range(max_samples):
        if checks >= max_checks:
            break
        candidate = final if sample % 5 == 0 else rng.uniform(lo, hi)
        nearest = int(np.argmin([np.linalg.norm(q - candidate) for q in nodes]))
        delta = candidate - nodes[nearest]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-12:
            continue
        next_node = nodes[nearest] + delta * min(1., step_rad / distance)
        if not edge(nodes[nearest], next_node):
            continue
        nodes.append(next_node)
        parents.append(nearest)
        if edge(next_node, final):
            path = [final]
            index = len(nodes) - 1
            while index >= 0:
                path.append(nodes[index])
                index = parents[index]
            return JointPathResult(True, tuple(reversed(path)), checks, "sampled corridor")
    return JointPathResult(False, (), checks, "search budget exhausted")


class PointIK:
    """Bounded inverse kinematics for a body-fixed point and selected joints.

    All other joints, including the base, remain at the supplied measured
    configuration. Targets are world-frame metres. A controller must convert
    base-frame sensor estimates explicitly, gate the residual, and rate-limit
    the result before sending it to servos. An optional body-fixed axis can be
    aligned to a world direction (two orientation conditions) or kept in a
    world plane (one condition). Position and angular errors are reported
    separately so a short arm cannot hide a compromised pose. This is not a
    collision-aware motion planner.
    """

    def __init__(self, model, body_name, joint_names, local_point, *, margin=.01,
                 local_axis=None, axis_scale_m=.05):
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
        self.local_axis = None
        if local_axis is not None:
            axis = _vector(local_axis, 3, "local axis")
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12:
                raise ValueError("local axis must have nonzero length")
            self.local_axis = axis / norm
        if not math.isfinite(axis_scale_m) or axis_scale_m <= 0:
            raise ValueError("axis residual scale must be finite and positive")
        self.axis_scale_m = axis_scale_m
        self.data = mujoco.MjData(model)

    def solve(self, qpos, target, *, axis_target=None, axis_plane_normal=None) -> IKResult:
        state = _vector(qpos, self.model.nq, "qpos")
        goal = _vector(target, 3, "target")
        if axis_target is not None and axis_plane_normal is not None:
            raise ValueError("choose an axis direction or a plane constraint, not both")
        constrained = axis_target is not None or axis_plane_normal is not None
        if (self.local_axis is not None) != constrained:
            raise ValueError("local axis and an orientation constraint must both be supplied")
        direction = None
        if constrained:
            direction = _vector(axis_target if axis_target is not None else axis_plane_normal,
                                3, "target axis or plane normal")
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                raise ValueError("target axis must have nonzero length")
            direction /= norm
        self.data.qpos[:] = state

        def residual(joints):
            self.data.qpos[self.qadr] = joints
            mujoco.mj_kinematics(self.model, self.data)
            rotation = self.data.xmat[self.body].reshape(3, 3)
            error = self.data.xpos[self.body] + rotation @ self.local_point - goal
            if direction is None:
                return error
            if axis_plane_normal is not None:
                return np.concatenate((error, [self.axis_scale_m *
                                               ((rotation @ self.local_axis) @ direction)]))
            return np.concatenate((error, self.axis_scale_m *
                                   (rotation @ self.local_axis - direction)))

        fit = least_squares(residual, np.clip(state[self.qadr], self.lower, self.upper),
                            bounds=(self.lower, self.upper), max_nfev=60,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = residual(fit.x)
        angle = None
        if direction is not None:
            actual = self.data.xmat[self.body].reshape(3, 3) @ self.local_axis
            cosine = np.clip(actual @ direction, -1, 1)
            angle = float(np.arcsin(abs(cosine)) if axis_plane_normal is not None
                          else np.arccos(cosine))
        return IKResult(fit.x.copy(), float(np.linalg.norm(error[:3])), angle)
