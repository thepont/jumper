"""Pre-contact approach proposals from observations and an admitted Rust motion.

The host supplies world-frame measured poses, independent scene coverage and a
full measured-to-command clearance Strategy. This module has no simulator,
camera, joint stepping, collision or actuator implementation.
"""
from dataclasses import dataclass
import math

from .grasp_follow import ObjectTargetTracker, PlanPoseGate


def _nanoseconds(seconds):
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('observation clocks require finite nonnegative seconds')
    return round(seconds * 1_000_000_000)


@dataclass(frozen=True)
class ApproachCommand:
    joints: tuple[float, ...]
    goal_world: tuple[float, ...]
    inward_world: tuple[float, ...]
    holding: bool = False


class ObservedApproach:
    """Application coordinator: existing tracker + SceneMotion + host guard.

    The motion must have been admitted for the supplied scene revision. Coverage
    is independent of target visibility: the older timestamp constrains movement.
    The Rust authority owns freshness, joint rates, geometry checks and the stop
    latch. A suspect/missing pose packet holds through the existing PlanPoseGate;
    it cannot refresh the last accepted capture time. Expiry or other failures
    consume that motion. Recovery then needs a newly admitted motion/coordinator.
    Contact ends this pre-contact executor; it cannot command closing or lifting.
    The host serializes observations and dispatch and owns the hold response.
    """

    def __init__(self, *, motion, tracker):
        if (not isinstance(tracker, ObjectTargetTracker) or
                any(not callable(getattr(motion, name, None)) for name in ('step', 'hold'))):
            raise ValueError('observed approach needs a target tracker and a scene motion')
        self._motion, self._tracker = motion, tracker
        self._pose_gate = PlanPoseGate()

    @property
    def arrived(self):
        return self._motion.arrived

    def command(self, current, measured, dt, *, now_s, position_world, rotation_world,
                observed_at_s, coverage_at_s, scene_revision, contact, clear):
        packet, tracked, now_ns, rejection, holding = None, None, 0, None, False
        try:
            now_ns = _nanoseconds(now_s)
            coverage_ns = _nanoseconds(coverage_at_s)
            if coverage_at_s > now_s:
                raise ValueError('scene coverage timestamp is in the future')
            if type(contact) is not bool:
                raise ValueError('contact must be boolean')
            if observed_at_s is None:
                if position_world is not None or rotation_world is not None:
                    raise ValueError('tracking requires a complete pose packet')
                self._pose_gate.update(now_s)
                pose_ok = False
            else:
                if position_world is None or rotation_world is None:
                    raise ValueError('tracking requires a complete pose packet')
                assessment = self._tracker.assess(position_world, rotation_world)
                pose_ok = self._pose_gate.update(now_s, observed_at=observed_at_s,
                                                 pose_admissible=assessment.admitted)
            if contact:
                rejection = 'contact ends pre-contact approach; fresh phase admission required'
            elif not pose_ok:
                if self._pose_gate.expired(now_s):
                    rejection = 'confirmed object observation expired; fresh plan required'
                else:
                    holding = True
                    tracked = (self._tracker.goal.copy(), self._tracker.axis.copy())
                    packet = (scene_revision,
                              min(_nanoseconds(self._pose_gate.confirmed), coverage_ns))
            else:
                tracked = self._tracker.update(now_s, position_world=position_world,
                    rotation_world=rotation_world, observed_at=observed_at_s, contact=contact)
                if tracked is None:
                    rejection = ('object motion outside original plan; replan required'
                                 if self._tracker.replan_required
                                 else 'usable object observation lost')
                else:
                    packet = (scene_revision, min(_nanoseconds(observed_at_s), coverage_ns))
        except (ValueError, TypeError, OverflowError) as error:
            rejection = str(error)
        # Even parsing/tracking errors pass through the Rust stop latch. There
        # is no second Python stop state or fallback motion policy to diverge.
        try:
            operation = self._motion.hold if holding else self._motion.step
            joints = operation(current, measured, dt, packet, now_ns, clear)
        except (ValueError, TypeError, OverflowError) as error:
            if rejection is not None:
                raise ValueError(rejection) from error
            raise
        goal, inward = tracked
        return ApproachCommand(tuple(joints), tuple(goal), tuple(inward), holding)
