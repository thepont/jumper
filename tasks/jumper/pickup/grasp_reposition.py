"""Measured progression for one 30 mm support-leg preparation experiment.

This State gate emits world-frame base references, never simulator pose writes.
The caller must solve support-foot IK, check the resulting joint corridor and
apply bounded servo targets. Support and clearance are measured inputs, not
inferred from a successful IK fit. This is not a walking controller.
"""
import math

import numpy as np


def support_joint_names(prefix, available):
    """Translate a support foot with J0..J2, keeping RF wrist/finger targets held.

    Native preparation exposed RF forearm/finger clearance loss when the free
    wrist participated in position-only IK. Three joints suffice to attempt
    foot translation; residual and collision guards still decide feasibility.
    """
    if (prefix not in ('RF', 'LM', 'RM', 'LR', 'RR') or
            not isinstance(available, (tuple, list)) or
            any(not isinstance(name, str) for name in available) or
            len(set(available)) != len(available)):
        raise ValueError('support mapping needs unique named support-leg joints')
    names = tuple(f'{prefix}_J{i}_joint' for i in range(3))
    if not set(names).issubset(available):
        raise ValueError('support foot requires its three positioning joints')
    return names


class SupportShift:
    def __init__(self):
        self.phase, self.reason = 'settle', None
        self.last_time, self.entered = None, None
        self.stable_since, self.stable_samples = None, 0
        self.start, self.anchors = None, None

    def block(self, reason):
        self.phase, self.reason = 'blocked', reason

    def _dwell(self, now, condition):
        if not condition:
            self.stable_since, self.stable_samples = None, 0
            return False
        if self.stable_since is None:
            self.stable_since = now
        self.stable_samples += 1
        return self.stable_samples >= 10 and now-self.stable_since >= .18-1e-9

    def update(self, now, *, base, feet, support_count, clear, tilt_rad):
        base, feet = np.asarray(base, dtype=float), np.asarray(feet, dtype=float)
        if (type(now) not in (int, float) or not math.isfinite(now) or now < 0 or
                (self.last_time is not None and now <= self.last_time) or
                base.shape != (3,) or feet.shape != (5, 3) or
                not np.isfinite(base).all() or not np.isfinite(feet).all() or
                type(support_count) is not int or not 0 <= support_count <= 5 or
                type(clear) is not bool or type(tilt_rad) not in (int, float) or
                not math.isfinite(tilt_rad) or tilt_rad < 0):
            raise ValueError('support shift needs finite, ordered and complete measurements')
        gap = self.last_time is not None and now-self.last_time > .040001
        self.last_time = now
        if self.phase == 'blocked':
            return None
        tilt_ok = tilt_rad <= math.radians(3.)
        if not clear or (self.phase != 'settle' and not tilt_ok):
            self.block('clearance or body tilt rejected')
            return None
        if gap:
            self.stable_since, self.stable_samples = None, 0
            if self.phase != 'settle':
                self.block('measurement interval exceeded 40 ms')
                return None
        if self.phase == 'settle':
            if self._dwell(now, now >= 1. and support_count == 5 and tilt_ok):
                self.phase, self.entered = 'moving', now
                self.start, self.anchors = base.copy(), feet.copy()
                self.stable_since, self.stable_samples = None, 0
            elif now > 4.:
                self.block('five-foot settling timeout')
            return None
        if support_count < 4 or np.linalg.norm(feet-self.anchors, axis=1).max() > .008:
            self.block('support lost or foot drift exceeds 8 mm')
            return None
        elapsed = now-self.entered
        target = self.start + np.array([.03, 0., 0.])
        if self.phase == 'moving':
            if self._dwell(now, elapsed >= 6. and np.max(abs(base-target)) < .003):
                self.phase = 'ready'
            elif elapsed > 12.:
                self.block('measured repositioning timeout')
                return None
        return self.start + np.array([min(.03, .005*elapsed), 0., 0.])
