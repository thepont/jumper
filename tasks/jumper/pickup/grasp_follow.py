"""Measured waypoint progression; clearance and pose alignment stay external."""
import math

import numpy as np
from mjrl.manipulation import joint_segment_step


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


def refreshed_approach_target(ik, qpos, goal, inward, current_targets, endpoint, dt, *,
                              compensate_tracking=False):
    """Locally refit the terminal pose at the measured base, never switch branches.

    The caller owns fresh perception, measured arrival at the final waypoint,
    and clearance checks of every emitted command. This does not certify grip.
    The 0.05 rad envelope is relative to the original checked endpoint, not the
    previous fit, so repeated refreshes cannot silently wander off the plan.
    Optional tracking trim acts on the nominal fit BEFORE rate limiting; trimming
    a rate-limited intermediate target can otherwise create outward target drift.
    """
    current, original = (np.asarray(v, dtype=float) for v in (current_targets, endpoint))
    if (type(compensate_tracking) is not bool or
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
    if compensate_tracking:
        return tracking_trim(joints, current, np.asarray(qpos)[ik.qadr], ik.lower, ik.upper, dt)
    return joint_segment_step(current, joints, max_delta=.45*dt)


class PlanPoseGate:
    """Hold on a suspect position packet; expire rather than moving blindly."""
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

    def update(self, now, *, observed_at=None, error_m=None):
        self.expired(now)
        if observed_at is not None:
            if (type(observed_at) not in (int, float) or not math.isfinite(observed_at) or
                    not 0 <= observed_at <= now or type(error_m) not in (int, float) or
                    not math.isfinite(error_m) or error_m < 0):
                raise ValueError('pose packet needs a finite past timestamp and nonnegative error')
            if self.last_packet is None or observed_at > self.last_packet:
                self.last_packet = observed_at
                self.held = error_m > .008
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
        elif error_m is not None:
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
