"""Sensor-pose grasp proposals and guarded progression, not a grasp certificate.

The geometry model is the known convex SOAP hull, scaled by measured dimensions.
Positions/orientations are in the *current robot base frame*, metres and wxyz.
The producer must transform delayed captures into that frame. No simulator state
is read here. IK/path/closure/lift evidence enters through one checking Strategy.
"""
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from mjrl.manipulation import TargetEstimate
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ObjectPose:
    target: TargetEstimate
    orientation_wxyz: tuple[float, ...] | None
    dimensions_m: tuple[float, ...]

    def __post_init__(self):
        dims = np.asarray(self.dimensions_m, dtype=float)
        if (not isinstance(self.target, TargetEstimate) or dims.shape != (3,) or
                not np.isfinite(dims).all() or not ((dims > 0) & (dims <= 1.)).all()):
            raise ValueError('pose needs a measured target and three finite dimensions in (0, 1] m')
        object.__setattr__(self, 'dimensions_m', tuple(float(v) for v in dims))
        if self.orientation_wxyz is not None:
            q = np.asarray(self.orientation_wxyz, dtype=float)
            if (q.shape != (4,) or not np.isfinite(q).all() or
                    abs(np.linalg.norm(q) - 1.) > 1e-5):
                raise ValueError('orientation must be an explicitly measured unit wxyz quaternion')
            object.__setattr__(self, 'orientation_wxyz', tuple(float(v) for v in q))


@dataclass(frozen=True)
class GraspCandidate:
    contact_b: tuple[float, ...]
    inward_b: tuple[float, ...]
    face_sign: int
    centre_fraction: float

    def __post_init__(self):
        point, axis = np.asarray(self.contact_b, dtype=float), np.asarray(self.inward_b, dtype=float)
        if (point.shape != (3,) or axis.shape != (3,) or
                not np.isfinite(point).all() or not np.isfinite(axis).all() or
                abs(np.linalg.norm(axis)-1.) > 1e-5 or
                type(self.face_sign) is not int or self.face_sign not in (-1, 1) or
                not isinstance(self.centre_fraction, (int, float)) or
                isinstance(self.centre_fraction, bool) or
                not math.isfinite(self.centre_fraction) or not 0 <= self.centre_fraction <= 1):
            raise ValueError('candidate needs a finite point, unit inward axis and bounded face metadata')
        object.__setattr__(self, 'contact_b', tuple(float(v) for v in point))
        object.__setattr__(self, 'inward_b', tuple(float(v) for v in axis))


def grasp_candidates(pose, reference_hull, *, now_s):
    """18 bounded surface proposals on the known shape's +/-Y sides.

    Central *reference points* are heuristics, never proof about either pad's
    actual contact. Convex half-spaces locate the bevelled hull, not a box face.
    Missing orientation is not silently replaced by an upright assumption.
    """
    if (type(now_s) not in (int, float) or not math.isfinite(now_s) or now_s < 0 or
            (pose is not None and not isinstance(pose, ObjectPose))):
        raise ValueError('need a finite nonnegative clock and an ObjectPose or no detection')
    if (pose is None or pose.orientation_wxyz is None or
            not pose.target.usable(now_s, max_age_s=.25, min_confidence=.6)):
        return ()
    vertices = np.asarray(reference_hull, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or
            not 4 <= len(vertices) <= 256 or not np.isfinite(vertices).all()):
        raise ValueError('known shape requires 4..256 finite hull points')
    half = np.abs(vertices).max(axis=0)
    if (half <= 0).any():
        raise ValueError('known shape must have positive extent on every axis')
    vertices = vertices * (np.asarray(pose.dimensions_m) / (2 * half))
    half = np.asarray(pose.dimensions_m) / 2
    try:
        equations = ConvexHull(vertices).equations
    except QhullError as exc:
        raise ValueError('known shape requires a nondegenerate convex hull') from exc
    q = pose.orientation_wxyz
    rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    centre = np.asarray(pose.target.position)
    candidates = []
    for fx in (-.25, 0., .25):
        for fz in (-.25, 0., .25):
            point = np.array([fx * half[0], 0., fz * half[2]])
            constant = equations[:, :3] @ point + equations[:, 3]
            ny = equations[:, 1]
            if ((np.abs(ny) < 1e-10) & (constant > 1e-9)).any():
                continue
            bounds = -constant / np.where(np.abs(ny) > 1e-10, ny, 1.)
            lower = np.where(ny < -1e-10, bounds, -np.inf).argmax()
            upper = np.where(ny > 1e-10, bounds, np.inf).argmin()
            if bounds[lower] > bounds[upper]:
                continue
            for sign, index in ((-1, lower), (1, upper)):
                point[1] = bounds[index]
                contact = centre + rotation @ point
                inward = rotation @ -equations[index, :3]
                candidates.append(GraspCandidate(tuple(contact), tuple(inward), sign,
                                                  max(abs(fx), abs(fz))))
    return tuple(candidates)


@dataclass(frozen=True)
class CandidateChecks:
    ik_pass: bool
    path_pass: bool
    closure_pass: bool
    lift_pass: bool
    reasons: tuple[str, ...]

    def __post_init__(self):
        flags = (self.ik_pass, self.path_pass, self.closure_pass, self.lift_pass)
        if (any(type(v) is not bool for v in flags) or
                not isinstance(self.reasons, tuple) or
                any(not isinstance(r, str) or not r for r in self.reasons) or
                (all(flags) != (len(self.reasons) == 0))):
            raise ValueError('candidate checks need explicit booleans and reasons for rejection')

    @property
    def admitted(self):
        return self.ik_pass and self.path_pass and self.closure_pass and self.lift_pass


@dataclass(frozen=True)
class GraspAdvice:
    selected: GraspCandidate | None
    checks: tuple[CandidateChecks, ...]
    physical_pickup_verified: bool = False


def choose_grasp(candidates, assess):
    """A checker owns kinematics/collision semantics; admission is not success."""
    if (not isinstance(candidates, tuple) or len(candidates) > 18 or
            any(not isinstance(c, GraspCandidate) for c in candidates) or not callable(assess)):
        raise ValueError('advice needs at most 18 candidates and a checking strategy')
    checks = tuple(assess(c) for c in candidates)
    if any(not isinstance(check, CandidateChecks) for check in checks):
        raise ValueError('checking strategy must return CandidateChecks')
    admitted = [c for c, check in zip(candidates, checks, strict=True) if check.admitted]
    selected = min(admitted, key=lambda c: c.centre_fraction) if admitted else None
    return GraspAdvice(selected, checks)


class Phase(Enum):
    WAIT = 'wait_pose'
    APPROACH = 'approach'
    CLOSE = 'close'
    LIFT = 'lift'
    DONE = 'verified_pickup'
    BLOCKED = 'blocked'


class GripGate:
    """Fail-closed State machine; no servo authority or simulator privileges.

    Caller must independently check measured alignment, loaded central contacts,
    runtime clearance and canonical pickup. It may only request closing/lifting
    in the corresponding phase. Samples alone cannot bypass minimum dwell time.
    Native loads are simulation feedback, not an existing hardware tactile sensor.
    """
    def __init__(self):
        self.phase = Phase.WAIT
        self.reason = None
        self.entered = None
        self.last_time = None
        self.count = 0
        self.stable_since = None

    @property
    def finished_successfully(self):
        return self.phase == Phase.DONE

    def _transition(self, phase, now_s, reason=None):
        self.phase, self.entered, self.reason = phase, now_s, reason
        self.count, self.stable_since = 0, None

    def advance(self, now_s, *, pose_usable, plan_admitted, aligned,
                loaded_central, pickup_verified, guard_failed):
        flags = (pose_usable, plan_admitted, aligned, loaded_central, pickup_verified, guard_failed)
        if (type(now_s) not in (int, float) or not math.isfinite(now_s) or now_s < 0 or
                (self.last_time is not None and now_s <= self.last_time) or
                any(type(v) is not bool for v in flags)):
            raise ValueError('gate needs a strictly increasing finite clock and boolean feedback')
        if self.last_time is not None and now_s-self.last_time > .04 + 1e-9:
            self.count, self.stable_since = 0, None
        self.last_time = now_s
        if self.entered is None:
            self.entered = now_s
        if self.phase == Phase.BLOCKED:
            return self.phase
        if guard_failed:
            self._transition(Phase.BLOCKED, now_s, 'runtime safety guard failed')
        elif self.phase == Phase.DONE:
            return self.phase
        elif self.phase == Phase.WAIT:
            if pose_usable and plan_admitted:
                self._transition(Phase.APPROACH, now_s)
            elif pose_usable and not plan_admitted:
                self._transition(Phase.BLOCKED, now_s, 'no admitted grasp plan')
            elif now_s-self.entered > 4.:
                self._transition(Phase.BLOCKED, now_s, 'usable object pose timeout')
        elif not pose_usable or not plan_admitted:
            self._transition(Phase.BLOCKED, now_s, 'pose or plan lost')
        elif self.phase == Phase.APPROACH:
            self.count = self.count + 1 if aligned else 0
            self.stable_since = (now_s if self.count == 1 else self.stable_since) if aligned else None
            if self.count >= 10 and now_s-self.stable_since >= .18 - 1e-9:
                self._transition(Phase.CLOSE, now_s)
            elif now_s-self.entered > 15.:
                self._transition(Phase.BLOCKED, now_s, 'measured alignment timeout')
        elif self.phase == Phase.CLOSE:
            if not aligned:
                self._transition(Phase.BLOCKED, now_s, 'alignment lost during closure')
            else:
                self.count = self.count + 1 if loaded_central else 0
                self.stable_since = (now_s if self.count == 1 else self.stable_since) if loaded_central else None
                if self.count >= 11 and now_s-self.stable_since >= .2 - 1e-9:
                    self._transition(Phase.LIFT, now_s)
                elif now_s-self.entered > 5.:
                    self._transition(Phase.BLOCKED, now_s, 'loaded central pinch timeout')
        elif self.phase == Phase.LIFT:
            if pickup_verified:
                self._transition(Phase.DONE, now_s)
            elif not loaded_central:
                self._transition(Phase.BLOCKED, now_s, 'grip lost during lift')
            elif now_s-self.entered > 5.:
                self._transition(Phase.BLOCKED, now_s, 'physical pickup timeout')
        return self.phase
