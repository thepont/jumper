"""Pre-contact approach proposals from observations and an admitted Rust motion.

The host supplies world-frame measured poses, independent scene coverage and a
full measured-to-command clearance Strategy. This module has no simulator,
camera, joint stepping, collision or actuator implementation.
"""
from dataclasses import dataclass
import math

from .grasp_follow import ObjectTargetTracker


def _nanoseconds(seconds):
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('observation clocks require finite nonnegative seconds')
    return round(seconds * 1_000_000_000)


@dataclass(frozen=True)
class ApproachCommand:
    joints: tuple[float, ...]
    goal_world: tuple[float, ...]
    inward_world: tuple[float, ...]


class ObservedApproach:
    """Application coordinator: existing tracker + SceneMotion + host guard.

    The motion must have been admitted for the supplied scene revision. Coverage
    is independent of target visibility: the older timestamp constrains movement.
    The Rust authority owns freshness, joint rates, geometry checks and the stop
    latch. Any unavailable/rejected observation is submitted as a missing scene,
    consuming that motion. Recovery needs a newly admitted motion/coordinator.
    Contact ends this pre-contact executor; it cannot command closing or lifting.
    The host serializes observations and dispatch and owns the hold response.
    """

    def __init__(self, *, motion, tracker):
        if (not isinstance(tracker, ObjectTargetTracker) or
                not callable(getattr(motion, 'step', None))):
            raise ValueError('observed approach needs a target tracker and a scene motion')
        self._motion, self._tracker = motion, tracker

    @property
    def arrived(self):
        return self._motion.arrived

    def command(self, current, measured, dt, *, now_s, position_world, rotation_world,
                observed_at_s, coverage_at_s, scene_revision, contact, clear):
        packet, tracked, now_ns, rejection = None, None, 0, None
        try:
            now_ns = _nanoseconds(now_s)
            coverage_ns = _nanoseconds(coverage_at_s)
            if coverage_at_s > now_s:
                raise ValueError('scene coverage timestamp is in the future')
            tracked = self._tracker.update(now_s, position_world=position_world,
                rotation_world=rotation_world, observed_at=observed_at_s, contact=contact)
            if contact:
                rejection = 'contact ends pre-contact approach; fresh phase admission required'
            elif tracked is None:
                rejection = ('object motion outside original plan; replan required'
                             if self._tracker.replan_required else 'usable object observation lost')
            else:
                packet = (scene_revision, min(_nanoseconds(observed_at_s), coverage_ns))
        except (ValueError, TypeError, OverflowError) as error:
            rejection = str(error)
        # Even parsing/tracking errors pass through the Rust stop latch. There
        # is no second Python stop state or fallback motion policy to diverge.
        try:
            joints = self._motion.step(current, measured, dt, packet, now_ns, clear)
        except (ValueError, TypeError, OverflowError) as error:
            if rejection is not None:
                raise ValueError(rejection) from error
            raise
        goal, inward = tracked
        return ApproachCommand(tuple(joints), tuple(goal), tuple(inward))
