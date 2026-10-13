---
name: robot-model-check
description: Diagnose or validate robot geometry, coordinate transforms, joint mappings, kinematics, Jacobians, dynamics and controller timing in Jumper. Use for incorrect stance, reversed motion, model conversion and unstable control. Appearance generation belongs to Design; policy tuning belongs to rl-experiment; deployment belongs to deploy.
---

# Robot model and control checks

Establish the mechanical and numerical contract before tuning around a mismatch.
Read `CLAUDE.md`, `docs/DESIGN.md` and the relevant asset source/tools; use
`deploy/BUNDLE.md` when checking the controller boundary. Read
[sources](references/sources.md) for transform or Jacobian derivations. Paths in
prose and commands are repository-relative unless linked.

## Locate the authority

Trace source model, generated MJCF, task constants, simulator state and exported
joint/observation/action layout. Identify the writer for each value. Use existing
asset generation and calibration tools rather than hand-editing generated geometry
or introducing another kinematics authority. Check tool help/source before running
anything that writes assets.

Record joint names and order, joint axes, zero offsets, sign conventions, limits,
gear mapping, action scale, gains and effort/velocity limits. Resolve joints by name
and model address; joint index is not a `qpos` index, and `nq` need not equal `nv`.
Include base and tool/foot frames and sensor extrinsics when relevant.

State units, handedness, quaternion component order, transform direction and
composition order. For example, define `T_A_B` by `p_A = T_A_B p_B` before using
it. Distinguish points from directions and poses from twists/wrenches. If using
Denavit–Hartenberg parameters, name the exact convention; Craig's frame assignment
must not be silently mixed with another DH convention or assumed to match MJCF.

## Choose checks for the reported failure

- **Transforms:** identity, inverse and round-trip checks plus a known nonzero
  translation/rotation. A round trip alone can hide two consistently wrong maps.
- **Forward kinematics:** compare expected foot/tool poses with MuJoCo at home
  and several asymmetric configurations. Symmetric poses can hide swapped limbs.
- **Jacobians/IK:** compare with finite differences at the same point and in the
  same frame. Perturb generalized velocity coordinates with the appropriate
  position integration operation for free/ball joints. Inspect singular values,
  unreachable targets and joint-limit margins; bound steps and residuals rather
  than blindly inverting a singular Jacobian.
- **Forces:** check virtual power consistency, `tau · qdot = wrench · twist`,
  with matching frame, point, component ordering and force sign convention.
  Static `tau = J^T wrench` reasoning does not include all dynamic or contact loads.
- **Dynamics/stance:** inspect mass, inertia, COM, gravity, contacts, support and
  actuator capability. An assisted hold or a plausible render is insufficient
  evidence of an unassisted stance.
- **Control:** inspect physics timestep, policy decimation, servo update rate,
  delay, saturation and filtering. Continuous-time stability intuition does not
  establish stability of the sampled controller with those limits and delays.

Use only the checks relevant to the failure. After changing state, update MuJoCo's
derived quantities before reading poses/Jacobians. Pick numeric tolerances from
units, scale and precision, and report the observed residuals.

## Reuse workspace checks and report limits

Existing asset diagnostics include:

```bash
python assets/jumper/tools/check_stance.py
python assets/jumper/tools/check_model_slim.py
```

Inspect them first: the model-slim diagnostic allocates many model copies and
prints failures, so exit code alone does not establish success. Choose a bounded
probe if its memory cost is unsuitable. Keep task/asset probes outside `tests/`
where required by `CLAUDE.md`; shared machinery regressions belong in the suite.

Deliver source revisions, mappings, configurations checked, measured residuals,
limits and unresolved mechanical assumptions. For a Train/Design comparison,
follow `docs/WORKFLOWS.md`, record the independent Design revision and preserve any
mechanical difference explicitly. Route export or live robot work through
[deploy](../deploy/SKILL.md); this diagnostic skill does not itself authorize
actuating hardware.
