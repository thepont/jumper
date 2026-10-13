# Foundations and conventions

Original operational summaries based on the user-supplied third edition and
primary documentation checked on 2026-10-13. No uploaded PDF is needed at runtime.

- John J. Craig, *Introduction to Robotics: Mechanics and Control*, 3rd ed.,
  Pearson, 2005: chapter 2 covers frames and transformations; chapters 3–4 cover
  forward/inverse kinematics; chapter 5 covers Jacobians, singularities and static
  forces; chapter 6 covers dynamics; chapters 7, 9–11 cover trajectories and
  control. Its manipulator derivations require care when applied to a floating
  base with unilateral contacts; they are not a complete legged balance model.
- [MuJoCo functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html):
  consult `mj_forward`, `mj_jac`, `mj_jacSite`, `mj_integratePos` and
  `mj_differentiatePos` for derived-state updates, Jacobians and quaternion-aware
  position operations. Check the installed version before relying on an API.
- [ROS REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst):
  provides explicit unit/frame conventions including different camera optical
  axes. It is a reference for boundary documentation, not Jumper's model authority.

The asymmetric-pose checks, source tracing and reporting procedure adapt these
foundations to the repository's model and deployment boundaries.
