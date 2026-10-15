"""Native scene/IK/path Adapter for the Rust tracking supervisor, not a servo."""
import numpy as np

from .grasp_advisor import GraspCandidate
from .grasp_follow import measured_grasp_alignment
from .grasp_planning import NativeGraspChecks, fit_grasp_waypoint, grasp_waypoint_goals


class NativeTrackingPlanner:
    def __init__(self, env, estimate, reference, *, standoff_m, grasp_standoff_m=None,
                 checker_factory=NativeGraspChecks, axis_tolerance_rad=None):
        if (not callable(estimate) or not callable(checker_factory) or
                not isinstance(reference, GraspCandidate)):
            raise ValueError('native tracking requires estimate and checking Strategies')
        grasp_waypoint_goals([0., 0., 0.], [1., 0., 0.], standoff_m)
        if grasp_standoff_m is None:
            grasp_standoff_m = standoff_m
        grasp_waypoint_goals([0., 0., 0.], [1., 0., 0.], grasp_standoff_m)
        self.env, self.estimate, self.checker_factory = env, estimate, checker_factory
        self.reference, self.standoff_m = reference, standoff_m
        self.grasp_standoff_m = grasp_standoff_m
        self.ik_options = {} if axis_tolerance_rad is None else {
            'axis_tolerance_rad': axis_tolerance_rad}
        self.closure_ready = False
        self.records = []
        self.outer_reached = False

    def __call__(self, request, *, closing=False):
        if type(closing) is not bool:
            raise ValueError('planning phase must be explicit')
        self.closure_ready = False
        standoff = self.grasp_standoff_m if closing else self.standoff_m
        estimate = self.estimate()
        checker = self.checker_factory(self.env, estimate, standoff_m=standoff, **self.ik_options)
        axis, goal = np.asarray(request['axis']), np.asarray(request['position'])
        point, measured_axis = np.asarray(request['tool_position']), np.asarray(request['tool_axis'])
        contact = goal + axis*standoff
        outer, _, _ = grasp_waypoint_goals(contact, axis, standoff)
        angle = float(np.arccos(np.clip(measured_axis@axis, -1., 1.)))
        self.outer_reached |= closing or measured_grasp_alignment(
            float(np.linalg.norm(point-outer)), angle)
        target = goal if self.outer_reached else outer
        fit, attempts = fit_grasp_waypoint(checker.ik, checker.qpos, target, axis, **self.ik_options)
        selected = next(row for row in attempts if row['selected'])
        record = dict(captured_ns=request['observed_at_ns'], remaining_ns=request['remaining_ns'],
                      stage='terminal' if self.outer_reached else 'outer',
                      fit_error_m=fit.error_m, axis_error_rad=fit.axis_error_rad,
                      admitted=False, closure_ready=False)
        self.records.append(record)
        if not selected['admitted']:
            record['reason'] = 'current approach IK rejected'
            return None
        # JointMotion advances from requested targets. Re-anchoring its route at
        # lagged measurements would command backwards on each fresh capture.
        # The executor's guard still checks the measured-to-proposed corridor.
        path = checker._path(request['current'], fit.joints, jaw=float(checker.qpos[checker.jaw_adr]))
        record.update(reason=path.reason, route_checks=path.checks, admitted=path.found)
        if not path.found:
            return None
        if measured_grasp_alignment(float(np.linalg.norm(point-goal)), angle):
            closure = self.checker_factory(self.env, estimate, standoff_m=self.grasp_standoff_m,
                                           **self.ik_options)
            candidate = GraspCandidate(tuple(closure.base_rotation.T@(contact-closure.base_position)),
                tuple(closure.base_rotation.T@axis), self.reference.face_sign,
                self.reference.centre_fraction)
            verdict = closure.assess(candidate)
            self.closure_ready = verdict.admitted
            record.update(closure_ready=verdict.admitted, closure_reasons=verdict.reasons)
            record['closure_details'] = closure.details
            record['closure_planning_qpos'] = closure.qpos.tolist()
        return [list(q) for q in path.waypoints]
