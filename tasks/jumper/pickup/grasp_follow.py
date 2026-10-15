"""Measured waypoint progression; clearance and pose alignment stay external."""
import math
from dataclasses import dataclass

import numpy as np
from mjrl.manipulation import joint_segment_step


def measured_grasp_alignment(position_error_m, axis_error_rad):
    """Experimental closing gate: 5 mm measured error, unchanged 10-degree axis.

    Planning still requires 4 mm. Caller must additionally require fresh sensing,
    stable arrival and full-command clearance. Passing is not grip evidence.
    """
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in (position_error_m, axis_error_rad)):
        raise ValueError('alignment requires finite nonnegative position and angular errors')
    return position_error_m <= .005 and axis_error_rad <= math.radians(10.)


@dataclass(frozen=True)
class TargetMotion:
    goal_world: np.ndarray
    inward_world: np.ndarray
    position_error_m: float
    goal_error_m: float
    rotation_error_rad: float

    @property
    def admitted(self):
        return bool(self.position_error_m <= .008 and self.goal_error_m <= .008 and
                    self.rotation_error_rad <= math.radians(10.) + 1e-12)


class ObjectTargetTracker:
    """Bounded pre-contact object-relative target, in a fixed world frame.

    Inputs are measured poses, not simulator truth. Rotation maps object-local
    points into world coordinates. Limits are relative to the ORIGINAL plan:
    8 mm object/goal displacement and 10 degrees rotation, never a rolling bound.
    Missing/stale packets yield no target. Large motion latches replan-required;
    this class cannot authorise replanning or execute commands. First contact
    freezes the goal permanently for this approach, avoiding contact chasing.
    Caller owns IK, path clearance, timing, contact validity and lift/slip checks.
    """
    @staticmethod
    def _pose(position, rotation):
        p, r = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
        if (p.shape != (3,) or r.shape != (3, 3) or
                not np.isfinite(p).all() or not np.isfinite(r).all() or
                not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0) or
                abs(np.linalg.det(r)-1.) > 1e-6):
            raise ValueError('tracking needs a finite world position and proper rotation')
        return p.copy(), r.copy()

    def __init__(self, position_world, rotation_world, goal_world, inward_world):
        self.origin, self.rotation = self._pose(position_world, rotation_world)
        goal, axis = np.asarray(goal_world, dtype=float), np.asarray(inward_world, dtype=float)
        if (goal.shape != (3,) or axis.shape != (3,) or
                not np.isfinite(goal).all() or not np.isfinite(axis).all() or
                abs(np.linalg.norm(axis)-1.) > 1e-6):
            raise ValueError('tracking needs a finite goal and unit inward axis')
        self.initial_goal = goal.copy()
        self.goal, self.axis = goal.copy(), axis.copy()
        self.local_goal = self.rotation.T @ (goal-self.origin)
        self.local_axis = self.rotation.T @ axis
        self.observed_at = None
        self.last_time = None
        self.contact_latched = False
        self.replan_required = False

    def assess(self, position_world, rotation_world):
        """Preview the original-plan envelope without accepting or latching it."""
        position, rotation = self._pose(position_world, rotation_world)
        goal = position + rotation @ self.local_goal
        angle = math.acos(np.clip((np.trace(rotation @ self.rotation.T)-1.)/2., -1., 1.))
        return TargetMotion(goal, rotation @ self.local_axis,
                            float(np.linalg.norm(position-self.origin)),
                            float(np.linalg.norm(goal-self.initial_goal)), angle)

    def update(self, now, *, position_world=None, rotation_world=None,
               observed_at=None, contact=False):
        if (type(now) not in (int, float) or not math.isfinite(now) or now < 0 or
                (self.last_time is not None and now <= self.last_time) or
                type(contact) is not bool):
            raise ValueError('tracking needs an increasing finite clock and boolean contact')
        packet = (position_world, rotation_world, observed_at)
        if any(value is not None for value in packet):
            if (any(value is None for value in packet) or
                    type(observed_at) not in (int, float) or
                    not math.isfinite(observed_at) or not 0 <= observed_at <= now):
                raise ValueError('tracking requires a complete, past-timestamped pose packet')
            assessment = self.assess(position_world, rotation_world)
        self.last_time = now
        self.contact_latched |= contact
        if self.replan_required or observed_at is None or now-observed_at > .25:
            return None
        if self.observed_at is not None and observed_at < self.observed_at:
            return None
        if self.observed_at is None or observed_at > self.observed_at:
            if not assessment.admitted:
                self.replan_required = True
                return None
            self.observed_at = observed_at
            if not self.contact_latched:
                self.goal = assessment.goal_world
                self.axis = assessment.inward_world
        return self.goal.copy(), self.axis.copy()


def grasp_finger_target(phase, current, opened, closed, loaded, dt):
    """Canonical LF angle: close only with permission; hold once loaded.

    Caller owns phase transitions, contact evidence and full-command clearance.
    This 0.5 rad/s target bound supplements, not replaces, the squeeze limiter.
    """
    from .grasp_advisor import Phase
    if (not isinstance(phase, Phase) or type(loaded) is not bool or
            any(type(x) not in (int, float) or not math.isfinite(x)
                for x in (current, opened, closed, dt)) or
            not opened <= current <= closed or opened >= closed or not 0 < dt <= .02):
        raise ValueError('finger command needs a phase, bounded LF angle, load flag and valid dt')
    if phase == Phase.CLOSE and not loaded:
        return min(closed, current+.5*dt)
    return current


def tracking_trim(desired, current_targets, measured, lower, upper, dt):
    """Estimate steady servo lag without integrating it; caller checks clearance.

    Opt-in terminal tracking only. Cap compensation at 0.02 rad per joint and
    preserve the existing 0.45 rad/s command rate and supplied joint bounds.
    Not a force controller: stale sensing or contact must be handled externally.
    """
    desired, current, actual, lo, hi = (
        np.asarray(v, dtype=float) for v in (desired, current_targets, measured, lower, upper))
    if (any(v.shape != (4,) or not np.isfinite(v).all()
            for v in (desired, current, actual, lo, hi)) or not (hi > lo).all() or
            ((current < lo) | (current > hi) | (desired < lo) | (desired > hi)).any() or
            type(dt) not in (int, float) or not math.isfinite(dt) or not 0 < dt <= .02):
        raise ValueError('trim requires bounded finite four-joint targets and dt in (0, .02]')
    compensated = np.clip(desired + np.clip(current-actual, -.02, .02), lo, hi)
    return joint_segment_step(current, compensated, max_delta=.45*dt)


class TrackingCorrection:
    """One approach's fixed lag estimate; clearance can only attenuate it.

    Create a fresh instance for each admitted approach. This is not integral
    control or a force estimator. Each emitted command is independently checked;
    a held correction does not certify a changed scene or remove phase deadlines.
    """
    def __init__(self):
        self._offset = None

    def _command(self, desired, current, measured, lower, upper, dt, check):
        # Inputs have passed the refresh and tracking_trim boundary checks.
        offset = (np.clip(current-measured, -.02, .02) if self._offset is None
                  else self._offset)
        for scale in (1., .5, .25, .125, 0.):
            reduced = offset * scale
            target = np.clip(desired+reduced, lower, upper)
            candidate = joint_segment_step(current, target, max_delta=.45*dt)
            verdict = check(candidate.copy())
            if type(verdict) is not bool:
                raise ValueError('command clearance must return a boolean verdict')
            if verdict:
                self._offset = reduced.copy()
                return candidate
        return None


def refreshed_approach_target(ik, qpos, goal, inward, current_targets, endpoint, dt, *,
                              compensate_tracking=False, command_clearance=None,
                              tracking_state=None):
    """Locally refit the terminal pose at the measured base, never switch branches.

    The caller owns fresh perception, measured arrival at the final waypoint,
    and clearance checks of every emitted command. This does not certify grip.
    The 0.05 rad envelope is relative to the original checked endpoint, not the
    previous fit, so repeated refreshes cannot silently wander off the plan.
    Optional tracking trim acts on the nominal fit BEFORE rate limiting; trimming
    a rate-limited intermediate target can otherwise create outward target drift.
    Optional command_clearance must check the complete measured-to-command path,
    including current stance/jaw targets. Rejected compensation may fall back to
    the independently checked nominal command; neither pass implies grasp success.
    An opt-in TrackingCorrection latches the initial bias and tries bounded
    attenuation instead of alternating full and nominal corrections each frame.
    """
    current, original = (np.asarray(v, dtype=float) for v in (current_targets, endpoint))
    if (type(compensate_tracking) is not bool or
            (tracking_state is not None and (not isinstance(tracking_state, TrackingCorrection)
             or not compensate_tracking or command_clearance is None)) or
            (command_clearance is not None and not callable(command_clearance)) or
            current.shape != (4,) or original.shape != (4,) or
            not np.isfinite(current).all() or not np.isfinite(original).all() or
            type(dt) not in (int, float) or not math.isfinite(dt) or not 0 < dt <= .02):
        raise ValueError('refresh needs finite four-joint targets and dt in (0, .02]')
    fit = ik.solve(qpos, goal, axis_target=inward)
    joints = np.asarray(fit.joints, dtype=float)
    if (joints.shape != (4,) or not np.isfinite(joints).all() or
            not math.isfinite(fit.error_m) or not 0 <= fit.error_m <= .004 or
            fit.axis_error_rad is None or not math.isfinite(fit.axis_error_rad) or
            not 0 <= fit.axis_error_rad <= math.radians(10.) or
            np.max(abs(joints-original)) > .05):
        return None
    nominal = joint_segment_step(current, joints, max_delta=.45*dt)
    preferred = (tracking_trim(joints, current, np.asarray(qpos)[ik.qadr],
                               ik.lower, ik.upper, dt) if compensate_tracking else nominal)
    if tracking_state is not None:
        return tracking_state._command(joints, current, np.asarray(qpos)[ik.qadr],
                                       ik.lower, ik.upper, dt, command_clearance)
    if command_clearance is None:
        return preferred
    candidates = [preferred]
    if compensate_tracking and not np.array_equal(preferred, nominal):
        candidates.append(nominal)
    for candidate in candidates:
        verdict = command_clearance(candidate.copy())
        if type(verdict) is not bool:
            raise ValueError('command clearance must return a boolean verdict')
        if verdict:
            return candidate
    return None


class PlanPoseGate:
    """Confirm poses, holding suspect packets until accepted evidence expires.

    Callers supply either legacy positional error or a complete pose-admission
    verdict from their geometry authority. Neither can refresh a repeated packet.
    """
    def __init__(self, *, recovery_samples=1):
        if type(recovery_samples) is not int or recovery_samples not in (1, 2):
            raise ValueError('pose recovery requires one or two fresh confirmations')
        self.confirmed, self.last_packet = None, None
        self.held = True
        self.recovery_samples = recovery_samples
        self.recovery_count, self.recovery_at = 0, None

    def expired(self, now):
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
            raise ValueError('pose clock must be finite and nonnegative')
        return self.confirmed is None or not 0 <= now-self.confirmed <= .25

    def update(self, now, *, observed_at=None, error_m=None, pose_admissible=None):
        self.expired(now)
        if observed_at is not None:
            if (type(observed_at) not in (int, float) or not math.isfinite(observed_at) or
                    not 0 <= observed_at <= now):
                raise ValueError('pose packet needs a finite past timestamp and nonnegative error')
            if ((error_m is None) == (pose_admissible is None) or
                    (error_m is not None and (type(error_m) not in (int, float) or
                     not math.isfinite(error_m) or error_m < 0)) or
                    (pose_admissible is not None and type(pose_admissible) is not bool)):
                raise ValueError('supply either a nonnegative position error or explicit pose admission')
            if self.last_packet is None or observed_at > self.last_packet:
                self.last_packet = observed_at
                self.held = error_m > .008 if pose_admissible is None else not pose_admissible
                recovering = self.confirmed is not None and self.expired(now)
                if recovering and self.recovery_samples > 1:
                    if self.held or now-observed_at > .25:
                        self.recovery_count, self.recovery_at = 0, None
                    else:
                        if self.recovery_at is None or observed_at-self.recovery_at > .25:
                            self.recovery_count = 0
                        self.recovery_count += 1
                        self.recovery_at = observed_at
                    self.held = self.recovery_count < self.recovery_samples
                if not self.held:
                    self.confirmed = observed_at
                    self.recovery_count, self.recovery_at = 0, None
        elif error_m is not None or pose_admissible is not None:
            raise ValueError('pose error requires a packet timestamp')
        return not self.held and not self.expired(now)


class ApproachFollower:
    def __init__(self, waypoints):
        points = np.asarray(waypoints, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 4 or not 1 <= len(points) <= 128 or
                not np.isfinite(points).all()):
            raise ValueError('approach needs 1..128 finite four-joint waypoints')
        self.points = points.copy()
        self.index, self.arrived = 0, False

    def command(self, current_targets, measured_joints, dt):
        current = np.asarray(current_targets, dtype=float)
        measured = np.asarray(measured_joints, dtype=float)
        if (current.shape != (4,) or measured.shape != (4,) or
                not np.isfinite(current).all() or not np.isfinite(measured).all() or
                type(dt) not in (int, float) or not math.isfinite(dt) or not 0 < dt <= .02):
            raise ValueError('approach needs finite four-joint feedback and dt in (0, .02]')
        target = self.points[self.index]
        # Measured servo lag does not authorize cutting a commanded corner.
        # 1e-6 rad permits float32 target roundoff, not a geometric shortcut.
        reached = (np.max(abs(measured-target)) <= .02 and
                   np.max(abs(current-target)) <= 1e-6)
        if reached and self.index < len(self.points)-1:
            self.index += 1
            target = self.points[self.index]
            reached = False
        self.arrived = bool(reached and self.index == len(self.points)-1)
        return joint_segment_step(current, target, max_delta=.45*dt)
