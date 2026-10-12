"""Given obstacles, a bounded joint path must not cut through them."""

import unittest

import numpy as np

from mjrl.manipulation import plan_joint_path


class JointPathTests(unittest.TestCase):
    def test_clear_corridor_goes_directly_to_the_goal(self):
        start, goal = np.array([-.7, -.3]), np.array([.6, .4])
        result = plan_joint_path(start, goal, [-1, -1], [1, 1], lambda q: True)
        self.assertTrue(result.found)
        np.testing.assert_array_equal(result.waypoints[0], start)
        np.testing.assert_array_equal(result.waypoints[-1], goal)
        np.testing.assert_array_equal(start, [-.7, -.3])

    def test_blocked_straight_line_routes_around_the_obstacle(self):
        def valid(q):
            return not (-.12 < q[0] < .12 and q[1] < .45)

        result = plan_joint_path([-.7, 0], [.7, 0], [-1, -1], [1, 1], valid,
                                 seed=4, max_samples=800)
        self.assertTrue(result.found, result.reason)
        # A straight-line-only implementation would cross the barrier here.
        self.assertGreater(max(q[1] for q in result.waypoints), .45)
        for a, b in zip(result.waypoints, result.waypoints[1:]):
            for t in np.linspace(0, 1, 101):
                self.assertTrue(valid(a + t * (b - a)))

    def test_closed_barrier_never_claims_a_path_or_exceeds_budget(self):
        calls = 0

        def valid(q):
            nonlocal calls
            calls += 1
            return abs(q[0]) >= .12

        result = plan_joint_path([-.7, 0], [.7, 0], [-1, -1], [1, 1], valid,
                                 max_samples=100, max_checks=200)
        self.assertFalse(result.found)
        self.assertEqual(result.waypoints, ())
        self.assertLessEqual(calls, 200)
        self.assertEqual(result.checks, calls)

    def test_invalid_endpoint_is_rejected_before_search(self):
        for start, goal in (([0], [.5]), ([-.5], [0])):
            result = plan_joint_path(start, goal, [-1], [1], lambda q: abs(q[0]) > .1)
            self.assertFalse(result.found)
            self.assertEqual(result.reason, "invalid endpoint")

    def test_repeated_seed_produces_the_same_safe_path(self):
        def valid(q):
            return not (-.12 < q[0] < .12 and q[1] < .45)
        args = ([-.7, 0], [.7, 0], [-1, -1], [1, 1], valid)
        a, b = plan_joint_path(*args, seed=4), plan_joint_path(*args, seed=4)
        self.assertTrue(a.found and b.found)
        np.testing.assert_array_equal(a.waypoints, b.waypoints)

    def test_malformed_boundary_input_fails_before_collision_queries(self):
        def unexpected(q):
            self.fail("invalid input reached the collision strategy")

        for start, goal, lo, hi in (([float("nan")], [0], [-1], [1]),
                                    ([0], [2], [-1], [1]),
                                    ([0], [0], [1], [-1]),
                                    ([0, 0], [0], [-1], [1])):
            with self.assertRaises(ValueError):
                plan_joint_path(start, goal, lo, hi, unexpected)
        for options in ({"max_samples": 0}, {"max_checks": 1}, {"seed": -1},
                        {"step_rad": 0}, {"resolution_rad": float("nan")}):
            with self.assertRaises(ValueError):
                plan_joint_path([0], [.5], [-1], [1], unexpected, **options)
