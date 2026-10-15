"""Measured waypoint progression; clearance and pose alignment stay external."""
import math

import numpy as np
from mjrl.manipulation import joint_segment_step


class PlanPoseGate:
    """Hold on a suspect position packet; expire rather than moving blindly."""
    def __init__(self):
        self.confirmed, self.last_packet = None, None
        self.held = True

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
                if not self.held:
                    self.confirmed = observed_at
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
        reached = max(np.max(abs(measured-target)), np.max(abs(current-target))) <= .02
        if reached and self.index < len(self.points)-1:
            self.index += 1
            target = self.points[self.index]
            reached = False
        self.arrived = bool(reached and self.index == len(self.points)-1)
        return joint_segment_step(current, target, max_delta=.45*dt)
