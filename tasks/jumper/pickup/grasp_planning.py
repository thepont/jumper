"""Native, known-bench collision adapter for estimated-pose grasp proposals.

Uses the actual curved insert model, existing IK and sampled path authorities.
All object-pose substitutions are in private planning state. Static bench CAD
and ideal robot proprioception are assumed; this is not an arbitrary-scene map.
Frozen-object closure and rigid-carriage lift are conservative screening models,
not proof of dynamic seating or impossibility when no proposal survives.
"""
import math

import mujoco
import numpy as np
from mjrl.clearance import GeometryClearance
from mjrl.manipulation import PointIK, plan_joint_path
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .collision import non_claw_object_pairs, non_insert_object_pairs, nonadjacent_arm_pairs
from .grasp_advisor import CandidateChecks, ObjectPose


def grasp_waypoint_goals(contact, inward, standoff_m):
    """World-frame approach, grasp and lift references; not contact evidence."""
    point, axis = (np.asarray(v, dtype=float) for v in (contact, inward))
    if (point.shape != (3,) or axis.shape != (3,) or
            not np.isfinite(point).all() or not np.isfinite(axis).all() or
            abs(np.linalg.norm(axis)-1.) > 1e-5 or
            type(standoff_m) not in (int, float) or not math.isfinite(standoff_m) or
            not .002 <= standoff_m <= .01):
        raise ValueError('waypoints need finite world point, unit axis and 2..10 mm stand-off')
    grasp = point-axis*standoff_m
    return point-axis*.017, grasp, grasp+np.array([0., 0., .05])


def fit_grasp_waypoint(ik, qpos, goal, inward, *, alternate_starts=True):
    """Try ten deterministic starts through the existing IK authority.

    Only arm seed joints change in private copies. Admission retains separate
    position/alignment limits. This samples local solutions, not every branch;
    the chosen waypoint still needs collision and continuity checks.
    """
    if type(alternate_starts) is not bool:
        raise ValueError('alternate_starts must be boolean')
    seeds = [np.asarray(qpos, dtype=float).copy()]
    if alternate_starts:
        rng = np.random.default_rng(410019)
        for joints in [(ik.lower + ik.upper) / 2,
                       *rng.uniform(ik.lower, ik.upper, size=(8, len(ik.qadr)))]:
            seed = seeds[0].copy()
            seed[ik.qadr] = joints
            seeds.append(seed)
    fits, records = [], []
    for seed in seeds:
        fit = ik.solve(seed, goal, axis_target=inward)
        fits.append(fit)
        records.append({'error_m': float(fit.error_m),
                        'axis_error_rad': float(fit.axis_error_rad),
                        'admitted': bool(fit.error_m <= .004 and
                                         fit.axis_error_rad <= math.radians(10.))})

    def rank(index):
        fit, record = fits[index], records[index]
        residual = max(fit.error_m / .004, fit.axis_error_rad / math.radians(10.))
        travel = float(np.linalg.norm(fit.joints - seeds[0][ik.qadr]))
        return (not record['admitted'], travel if record['admitted'] else residual, index)

    selected = min(range(len(fits)), key=rank)
    for index, record in enumerate(records):
        record['selected'] = index == selected
    return fits[selected], records


class NativeGraspChecks:
    def __init__(self, env, estimate, *, standoff_m=.002):
        grasp_waypoint_goals([0., 0., 0.], [0., 1., 0.], standoff_m)
        self.standoff_m = standoff_m
        if (not isinstance(estimate, ObjectPose) or estimate.orientation_wxyz is None or
                not estimate.target.usable(env.common_step_counter * env.step_dt,
                                           max_age_s=.25, min_confidence=.6)):
            raise ValueError('planning needs a fresh complete estimated object pose')
        if env.num_envs != 1 or not hasattr(env.sim, 'env_mjdata'):
            raise ValueError('prototype planning requires one native world')
        self.term = env.action_manager.get_term('claw')
        if self.term.cfg.side != 'left':
            raise ValueError('prototype insert-frame audit covers only the left claw')
        live_model, _ = env.sim.env_mjdata(0)
        name = self.term.object_names[int(self.term.object_index[0])]
        object_geom_name = name + '/' + ('soap' if name == 'soap' else name)
        planning_spec = env.scene.spec.copy()
        geom = planning_spec.geom(object_geom_name)
        if geom is None or not geom.meshname:
            raise ValueError('known shape requires an independently compiled planning mesh')
        hull = np.asarray(self.term.cfg.hull)
        half = np.abs(hull).max(axis=0)
        estimated_vertices = hull * (np.asarray(estimate.dimensions_m) / (2 * half))
        mesh = planning_spec.mesh(geom.meshname)
        mesh.uservert = estimated_vertices.flatten().tolist()
        self.model = planning_spec.compile()
        if (self.model.nq != live_model.nq or
                any(self.model.joint(i).name != live_model.joint(i).name
                    for i in range(live_model.njnt))):
            raise ValueError('planning clone changed joint identity/order')
        self.data = mujoco.MjData(self.model)
        self.qpos = env.sim.data.qpos[0].cpu().numpy().copy()
        self.data.qpos[:] = self.qpos
        mujoco.mj_kinematics(self.model, self.data)
        base = self.model.body('robot/base_link').id
        self.base_position = self.data.xpos[base].copy()
        self.base_rotation = self.data.xmat[base].reshape(3, 3).copy()
        body = self.model.body(name + '/' + ('soap' if name == 'soap' else name)).id
        joint = int(self.model.body_jntadr[body])
        if self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('known object requires a free planning pose')
        self.object_adr = int(self.model.jnt_qposadr[joint])
        q = estimate.orientation_wxyz
        orientation = self.base_rotation @ Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        world_q = Rotation.from_matrix(orientation).as_quat()
        self.qpos[self.object_adr:self.object_adr+3] = (
            self.base_position + self.base_rotation @ estimate.target.position)
        self.qpos[self.object_adr+3:self.object_adr+7] = world_q[[3, 0, 1, 2]]
        # Never retain the active object's live true pose inside planning queries.
        self.palm = self.model.body('robot/LF_palm_link').id
        self.pads = [self.model.geom(f'robot/LF_{p}_grip_insert_link_meshcol').id
                     for p in ('palm', 'finger')]
        self.object_geom = self.model.geom(object_geom_name).id
        # Resolve from the compiled model, including geom-frame compensation.
        self.data.qpos[:] = self.qpos
        mujoco.mj_kinematics(self.model, self.data)
        fixed = self.pads[0]
        mesh = int(self.model.geom_dataid[fixed])
        start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
        vertices = self.model.mesh_vert[start:start+count].astype(float)
        world = vertices @ self.data.geom_xmat[fixed].reshape(3, 3).T + self.data.geom_xpos[fixed]
        local = (world-self.data.xpos[self.palm]) @ self.data.xmat[self.palm].reshape(3, 3)
        equations = ConvexHull(local).equations
        reference = np.asarray(self.term.cfg.local_point)
        distance = np.abs(equations[:, :3] @ reference + equations[:, 3])
        if float(distance.min()) > .0001:
            raise ValueError('calibrated contact reference is not on an active insert facet')
        moving_local = ((self.data.geom_xpos[self.pads[1]]-self.data.xpos[self.palm]) @
                        self.data.xmat[self.palm].reshape(3, 3))
        choices = np.flatnonzero(distance <= distance.min() + 1e-6)
        chosen = max(choices, key=lambda i: equations[i, :3] @ (moving_local-reference))
        self.local_axis = equations[chosen, :3].copy()
        self.ik = PointIK(self.model, 'robot/LF_palm_link',
                          tuple(f'robot/LF_J{i}_joint' for i in range(4)),
                          reference, local_axis=self.local_axis, axis_scale_m=.06)
        self.jaw_adr = int(self.model.joint('robot/LF_J4_joint').qposadr[0])
        pairs = list(nonadjacent_arm_pairs(self.model))
        floors = [self.model.geom(g).name for g in range(self.model.ngeom)
                  if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        arm = [self.model.geom(g).name for g in range(self.model.ngeom)
               if (self.model.geom(g).name or '').startswith('robot/LF_') and
               '_meshcol' in self.model.geom(g).name]
        pairs += [(a, floor) for a in arm for floor in floors]
        self.robot_guard = GeometryClearance(self.model, tuple(pairs), margin_m=.001)
        object_pairs = set(non_claw_object_pairs(self.model, name))
        object_pairs.update(non_insert_object_pairs(self.model, name))
        self.object_guard = GeometryClearance(self.model, tuple(sorted(object_pairs)), margin_m=.001)
        self.max_queries, self.queries = 512, 0
        self.details = []
        self.rejections = []

    def _valid(self, qpos):
        if self.queries >= self.max_queries:
            self.rejections.append({'reason': 'query budget exhausted'})
            return False
        self.queries += 1
        # Keep both predicate results, rather than hiding concurrent obstruction.
        robot = self.robot_guard.check(qpos)
        obj = self.object_guard.check(qpos)
        if not robot.passed or not obj.passed:
            self.rejections.append({'reason': 'clearance rejected',
                                    'robot_pairs': list(robot.distances) if not robot.passed else [],
                                    'object_pairs': list(obj.distances) if not obj.passed else []})
        return robot.passed and obj.passed

    def _path(self, start, goal, *, jaw, carry_from=None):
        def valid(joints):
            pose = self.qpos.copy()
            pose[self.ik.qadr], pose[self.jaw_adr] = joints, jaw
            if carry_from is not None:
                self.data.qpos[:] = pose
                mujoco.mj_kinematics(self.model, self.data)
                old_position, old_rotation = carry_from
                rotation = self.data.xmat[self.palm].reshape(3, 3) @ old_rotation.T
                old_object = self.qpos[self.object_adr:self.object_adr+3]
                pose[self.object_adr:self.object_adr+3] = (
                    self.data.xpos[self.palm] + rotation @ (old_object-old_position))
                old_q = self.qpos[self.object_adr+3:self.object_adr+7]
                new_q = Rotation.from_matrix(rotation @ Rotation.from_quat(old_q[[1, 2, 3, 0]]).as_matrix()).as_quat()
                pose[self.object_adr+3:self.object_adr+7] = new_q[[3, 0, 1, 2]]
            return self._valid(pose)
        return plan_joint_path(start, goal, self.ik.lower, self.ik.upper, valid,
                               max_samples=64, max_checks=512, resolution_rad=.025)

    def assess(self, candidate):
        contact = self.base_position + self.base_rotation @ candidate.contact_b
        inward = self.base_rotation @ candidate.inward_b
        goals = grasp_waypoint_goals(contact, inward, self.standoff_m)
        fits = []
        attempts = []
        seed = self.qpos.copy()
        for goal in goals:
            fit, records = fit_grasp_waypoint(self.ik, seed, goal, inward)
            fits.append(fit)
            attempts.append(records)
            seed[self.ik.qadr] = fit.joints
        ik_pass = all(f.error_m <= .004 and f.axis_error_rad <= math.radians(10.) for f in fits)
        detail = {'goals_world': [g.tolist() for g in goals],
                  'standoff_m': self.standoff_m,
                  'ik_errors_m': [f.error_m for f in fits],
                  'axis_errors_rad': [f.axis_error_rad for f in fits],
                  'ik_attempts': attempts,
                  'path': 'not_run', 'closure': 'not_run', 'lift': 'not_run'}
        reasons, path_pass, closure_pass, lift_pass = [], False, False, False
        if not ik_pass:
            reasons.extend(('IK position/alignment rejected',
                            'approach, closure and lift clearance not evaluated without admitted IK'))
        else:
            first = self._path(self.qpos[self.ik.qadr], fits[0].joints, jaw=self.term.cfg.open_angle)
            second = self._path(fits[0].joints, fits[1].joints, jaw=self.term.cfg.open_angle)
            path_pass = first.found and second.found
            detail['path'] = [first.reason, second.reason]
            detail['approach_waypoints'] = [
                [q.tolist() for q in segment.waypoints] for segment in (first, second)]
            detail['path_checks'] = [first.checks, second.checks]
            if not path_pass:
                reasons.append('sampled approach corridor rejected or query budget exhausted')
            closure_pose = self.qpos.copy()
            closure_pose[self.ik.qadr] = fits[1].joints
            selected_jaw = None
            samples = []
            for jaw in np.linspace(self.term.cfg.open_angle, self.term.cfg.closed_angle, 33):
                closure_pose[self.jaw_adr] = jaw
                clearance = self._valid(closure_pose)
                self.data.qpos[:] = closure_pose
                mujoco.mj_forward(self.model, self.data)
                distances = [float(mujoco.mj_geomDistance(self.model, self.data, pad,
                                                        self.object_geom, .01, None)) for pad in self.pads]
                samples.append({'jaw': float(jaw), 'clearance': clearance, 'pad_distances_m': distances})
                if not clearance or min(distances) < -.002:
                    break
                if max(distances) <= .002:
                    closure_pass, selected_jaw = True, float(jaw)
                    break  # Never require closing through an already seated object.
            detail['closure'] = samples
            if not closure_pass:
                reasons.append('curved-jaw frozen-object closure screen rejected; not a dynamic impossibility proof')
            if closure_pass:
                closure_pose[self.jaw_adr] = selected_jaw
                self.data.qpos[:] = closure_pose
                mujoco.mj_kinematics(self.model, self.data)
                carry = (self.data.xpos[self.palm].copy(), self.data.xmat[self.palm].reshape(3, 3).copy())
                lift = self._path(fits[1].joints, fits[2].joints, jaw=selected_jaw, carry_from=carry)
                lift_pass = lift.found
                detail['lift'] = lift.reason
                detail['lift_waypoints'] = [q.tolist() for q in lift.waypoints]
                if not lift_pass:
                    reasons.append('sampled rigid-carriage lift corridor rejected')
            else:
                reasons.append('lift clearance not evaluated without admitted closure')
        self.details.append(detail)
        return CandidateChecks(ik_pass, path_pass, closure_pass, lift_pass, tuple(reasons))
