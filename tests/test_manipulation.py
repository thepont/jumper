"""Manipulation boundaries: impossible/stale targets must not look successful."""

import unittest

import mujoco
import numpy as np

from mjrl.manipulation import PointIK, TargetEstimate


class JointShortcutTests(unittest.TestCase):
    def test_clear_path_removes_unnecessary_stops_without_mutating_input(self):
        from mjrl.manipulation import shortcut_joint_path
        path = np.array([[0., 0.], [.2, .3], [.7, .4], [1., 1.]])
        before = path.copy()
        result = shortcut_joint_path(path, lambda q: True, resolution_rad=.05)
        self.assertTrue(result.found)
        self.assertEqual(len(result.waypoints), 2)
        np.testing.assert_array_equal(path, before)
        np.testing.assert_array_equal(result.waypoints[-1], path[-1])

    def test_shortcut_cannot_cut_through_an_obstacle(self):
        from mjrl.manipulation import shortcut_joint_path
        path = [[0., 0.], [0., 1.], [1., 1.], [1., 0.]]
        def valid(q):
            return not (.3 < q[0] < .7 and q[1] < .8)
        self.assertFalse(valid(np.array([.5, 0.])))
        result = shortcut_joint_path(path, valid, resolution_rad=.02)
        self.assertTrue(result.found)
        self.assertGreater(len(result.waypoints), 2)
        for a, b in zip(result.waypoints, result.waypoints[1:]):
            self.assertTrue(all(valid(q) for q in np.linspace(a, b, 101)))

    def test_query_budget_exhaustion_never_returns_a_partial_success(self):
        from mjrl.manipulation import shortcut_joint_path
        result = shortcut_joint_path([[0.], [1.]], lambda q: True,
                                     max_checks=2, resolution_rad=.01)
        self.assertFalse(result.found)
        self.assertEqual(result.waypoints, ())
        self.assertLessEqual(result.checks, 2)

    def test_invalid_path_or_callback_results_are_rejected(self):
        from mjrl.manipulation import shortcut_joint_path
        for path in ([], [[0.]], [[0.], [float('nan')]], [[0., 1.], [1.]]):
            with self.assertRaises(ValueError):
                shortcut_joint_path(path, lambda q: True)
        with self.assertRaises(ValueError):
            shortcut_joint_path([[0.], [1.]], lambda q: None)
        for resolution in (True, 0., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                shortcut_joint_path([[0.], [1.]], lambda q: True,
                                    resolution_rad=resolution)


class ManipulationTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''
          <mujoco><compiler angle="radian"/><worldbody><body name="hand">
            <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
            <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
            <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
            <geom type="sphere" size=".01" mass="1"/>
          </body></worldbody></mujoco>''')
        self.ik = PointIK(self.model, "hand", ("x", "y", "z"), (0, 0, .02))

    def test_reaches_target_without_mutating_input_state(self):
        qpos = self.model.qpos0.copy()
        result = self.ik.solve(qpos, (.23, -.17, .42))
        self.assertLess(result.error_m, 1e-4)
        np.testing.assert_allclose(result.joints, (.23, -.17, .40), atol=1e-4)
        np.testing.assert_array_equal(qpos, self.model.qpos0)

    def test_unreachable_target_reports_residual_and_respects_limits(self):
        result = self.ik.solve(self.model.qpos0, (2, 0, .02))
        self.assertGreater(result.error_m, 1)
        self.assertLessEqual(result.joints[0], .99)

    def test_missing_or_duplicate_joints_fail_loudly(self):
        for joints in (("missing",), ("x", "x")):
            with self.assertRaises(ValueError):
                PointIK(self.model, "hand", joints, (0, 0, 0))

    def test_nonfinite_targets_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.ik.solve(self.model.qpos0, (value, 0, 0))

    def test_bad_state_and_missing_body_are_rejected(self):
        for state in ((0, 0), (0, float("nan"), 0)):
            with self.assertRaises(ValueError):
                self.ik.solve(state, (0, 0, 0))
        with self.assertRaises(ValueError):
            PointIK(self.model, "missing", ("x",), (0, 0, 0))

    def test_invalid_margin_and_unbounded_joints_are_rejected(self):
        for margin in (0, -1, 1, float("nan")):
            with self.assertRaises(ValueError):
                PointIK(self.model, "hand", ("x",), (0, 0, 0), margin=margin)
        self.model.jnt_limited[0] = False
        with self.assertRaises(ValueError):
            PointIK(self.model, "hand", ("x",), (0, 0, 0))

    def test_repeated_solves_use_the_new_target(self):
        self.ik.solve(self.model.qpos0, (.3, .2, .4))
        result = self.ik.solve(self.model.qpos0, (-.4, -.1, .1))
        np.testing.assert_allclose(result.joints, (-.4, -.1, .08), atol=1e-4)

    def test_axis_constraint_keeps_the_gripper_aligned_at_the_target(self):
        model = mujoco.MjModel.from_xml_string('''
          <mujoco><compiler angle="radian"/><worldbody><body name="hand">
            <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
            <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
            <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
            <joint name="yaw" type="hinge" axis="0 0 1" range="-3 3"/>
            <geom type="sphere" size=".01" mass="1"/>
          </body></worldbody></mujoco>''')
        ik = PointIK(model, "hand", ("x", "y", "z", "yaw"), (0, 0, .02),
                     local_axis=(1, 0, 0))
        state = model.qpos0.copy()
        state[3] = .7
        result = ik.solve(state, (.23, -.17, .42), axis_target=(0, 1, 0))
        self.assertLess(result.error_m, 1e-4)
        self.assertLess(result.axis_error_rad, 1e-4)
        self.assertAlmostEqual(result.joints[3], np.pi / 2, places=4)
        np.testing.assert_array_equal(state, (.0, .0, .0, .7))

    def test_axis_constraint_rejects_incomplete_or_nonunit_configuration(self):
        for axis in ((0, 0, 0), (float("nan"), 0, 1)):
            with self.assertRaises(ValueError):
                PointIK(self.model, "hand", ("x",), (0, 0, 0), local_axis=axis)
        with self.assertRaises(ValueError):
            self.ik.solve(self.model.qpos0, (0, 0, 0), axis_target=(1, 0, 0))

    def test_impossible_axis_is_reported_without_hiding_position_error(self):
        ik = PointIK(self.model, "hand", ("x", "y", "z"), (0, 0, 0),
                     local_axis=(1, 0, 0))
        result = ik.solve(self.model.qpos0, (.1, .2, .3), axis_target=(0, 1, 0))
        self.assertLess(result.error_m, 1e-4)
        self.assertAlmostEqual(result.axis_error_rad, np.pi / 2)

    def test_axis_can_stay_in_a_plane_without_overconstraining_its_yaw(self):
        model = mujoco.MjModel.from_xml_string('''
          <mujoco><compiler angle="radian"/><worldbody><body name="hand">
            <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
            <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
            <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
            <joint name="pitch" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <geom type="sphere" size=".01" mass="1"/>
          </body></worldbody></mujoco>''')
        ik = PointIK(model, "hand", ("x", "y", "z", "pitch"), (0, 0, .02),
                     local_axis=(1, 0, 0))
        state = model.qpos0.copy()
        state[3] = .7
        result = ik.solve(state, (.23, -.17, .42), axis_plane_normal=(0, 0, 1))
        self.assertLess(result.error_m, 1e-4)
        self.assertLess(result.axis_error_rad, 1e-4)
        self.assertAlmostEqual(result.joints[3], 0., places=4)

    def test_two_orientation_constraints_cannot_be_silently_mixed(self):
        ik = PointIK(self.model, "hand", ("x", "y", "z"), (0, 0, 0),
                     local_axis=(1, 0, 0))
        with self.assertRaises(ValueError):
            ik.solve(self.model.qpos0, (0, 0, 0), axis_target=(1, 0, 0),
                     axis_plane_normal=(0, 0, 1))

    def test_sensor_estimates_expire_and_do_not_accept_future_samples(self):
        estimate = TargetEstimate((.2, .1, .05), 1., .9)
        self.assertTrue(estimate.usable(1.1, max_age_s=.2, min_confidence=.8))
        self.assertFalse(estimate.usable(1.3, max_age_s=.2, min_confidence=.8))
        self.assertFalse(estimate.usable(.9, max_age_s=.2, min_confidence=.8))
        self.assertFalse(estimate.usable(1.1, max_age_s=.2, min_confidence=.95))

    def test_malformed_estimates_are_rejected(self):
        for position, timestamp, confidence in (
            ((0, 0), 1., .9), ((float("nan"), 0, 0), 1., .9),
            ((0, 0, 0), -1., .9), ((0, 0, 0), 1., 1.1),
        ):
            with self.assertRaises(ValueError):
                TargetEstimate(position, timestamp, confidence)

    def test_bad_freshness_thresholds_fail_instead_of_allowing_motion(self):
        estimate = TargetEstimate((.1, .2, .3), 1., .9)
        for now, age, confidence in ((1., 0., .8), (1., .2, -1.),
                                     (float("nan"), .2, .8)):
            with self.assertRaises(ValueError):
                estimate.usable(now, max_age_s=age, min_confidence=confidence)


if __name__ == "__main__":
    unittest.main()
