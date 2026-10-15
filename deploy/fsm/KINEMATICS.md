# Portable grasp-controller plan

## First slice: non-actuating forward kinematics

`kinematics::SerialArm` is an Adapter around pinned OpenRR `k` 0.32.0.
Library types remain private. It accepts a serial hinge descriptor, a base pose
and joint positions keyed by name; it returns the tool point and direction in
world coordinates. It does not command joints or certify collision clearance.

All lengths are metres and angles radians. Quaternions are scalar-first `wxyz`.
An origin means `p_parent = translation + R * p_body`. Each hinge composes
`T_origin * T_anchor * R(axis, q-reference) * T_-anchor`. Descriptor order is
parent to child, not alphabetic name order. Any fixed intermediate transforms
must be folded into the next origin by the model exporter; they cannot be omitted.
The descriptor must come from the authoritative compiled model, not a separately
hand-maintained robot geometry table.

Malformed descriptors, non-unit directions/quaternions, unknown or missing
joints, non-finite inputs and out-of-limit positions fail closed. Queries own
their library state, so rejected queries cannot contaminate later results.
This deliberately allocates per query: it is a validation foundation, not yet a
real-time servo-loop component. Performance and host integration remain gates.

## Behavioural evidence

Local Linux workspace, 2026-10-15, Rust 1.96.0:

- Existing baseline: 183 core tests passed.
- Initial adapter tests: three failed against the unimplemented boundary; the
  malformed-input rejection control passed.
- Implemented slice: 190 core tests passed, including seven adapter tests.
- Tests cover rotated tool points/directions, off-origin anchors, joint reference
  offsets, transformed bases, asymmetric two-joint chains, non-alphabetic joint
  order, exact limits, malformed inputs and recovery after rejected queries.
- Host `cargo check --no-default-features --features web --locked` passed. This
  is not a wasm-target build or browser execution test.
- Clippy is blocked by the unchanged `control.rs:327` `never_loop` error and
  reports existing warnings elsewhere; it reported none in the new adapter.

Reproduce the core gate:

```sh
cargo test --manifest-path deploy/fsm/Cargo.toml --no-default-features --lib --locked
```

## Remaining acceptance gates, in order

1. Add thin Python/WASM query bindings and compare against MuJoCo's compiled
   model at home and asymmetric joint/base poses. Keep all existing regression
   tests. No replacement of the current planner before parity passes.
2. Test Jacobians against finite differences; introduce bounded constrained IK
   with explicit residuals and termination reasons. Reject unreachable targets.
3. Integrate collision/path checking using unchanged clearance requirements;
   test approach and closure separately, including moving objects.
4. Add a State machine for approach, alignment, closure, lift and safe stop.
   Pause/resume must revalidate fresh measured state and object pose, not replay
   stale targets. State transitions earn this pattern; no speculative hierarchy.
5. Demonstrate the actual controller through the existing virtual interface on
   multiple object locations and sizes. Record physical pickup witnesses,
   failures, clearance and timing, not just animation or solver convergence.

No IK, pickup, hardware safety, deployment, resume behaviour or full virtual-demo
success is claimed by this first slice. Existing policies, physics and safety
thresholds are unchanged.

## Python query boundary and compiled-model parity

The `py` feature now exports `mjrl_fsm.SerialArm(descriptor_json)`. Its
`forward(translation, rotation_wxyz, positions_by_name)` returns
`(world_point, world_direction)` as two numeric lists. Invalid inputs raise
`ValueError`; there is no fallback and no actuator access. JSON is parsed only
at construction, and the wrapper delegates calculations to the same Rust core.
This does not yet connect a planner to `play --app`.

Measured on `music-room`, Linux, 2026-10-15, MuJoCo 3.11.0:

- Eight boundary tests failed first because the export did not exist, then
  passed after binding it. Three additional synthetic compiled-MuJoCo cases
  pin nonzero anchors/references and rotated body frames: 11 tests pass.
- A separate non-actuating LF asset probe checked five arm configurations
  (home plus four asymmetric configurations) under two base poses. All ten
  passed a 1e-10 tolerance. Maximum position difference was
  7.110250420355456e-17 m; maximum direction-vector difference was
  2.9893669801409083e-16.
- Asset source: `assets/jumper/jumper.xml`, SHA256
  `3342ca15a5bbcaf40e1f16dedb60044b91360dddae5f7f03f05eb3e6c830cf37`.
  The descriptor was extracted from the compiled model; the point was the
  existing calibrated fixed-contact reference. An asymmetric unit vector tested
  direction transformation, not contact-normal calibration.
- This probe used the raw asset, not the assembled pickup task scene. Static
  scratch states are FK queries, not feasible paths or physical executions.
- All 190 Rust core tests still pass. Ruff was unavailable in the local virtual
  environment; no Python lint pass is claimed.
- Existing `test_fsm_extension.py` / `test_app_play.py` run: 14 passed,
  one module skipped for its absent exported-policy fixture, 38 warnings.
  The new boundary suite builds independently and does not require that policy.

Reproduce the shared boundary and synthetic-MuJoCo tests with
`python -m pytest tests/test_kinematics_extension.py -q`.
WASM bindings, task-scene parity, Jacobian checks and closed-loop movement remain
pending; the earlier acceptance gates still apply.

## First Rust-driven physical simulation segment

`joint_motion::JointMotion` now owns feedback-gated segment progression, exposed
as `mjrl_fsm.JointMotion(points, lower, upper, rate, timeout)`.
`step(current, measured, dt, guard)` proposes the next rate-bounded point and
calls the host's guard before returning it. The guard must check the complete
measured-to-command corridor, support and any required sensing. It must return
a boolean; rejection, exception, malformed feedback or expiry latches Stop.
There is no blind resume: a new validated plan needs a new instance.

This uses State for Moving/Arrived/Stopped and a callback Strategy for scene
checks. The existing recorded `Trajectory` is time-indexed, so it cannot replace
measured waypoint progression without changing its existing behaviour. No
existing trajectory or grasp-follower tests were changed. The Python grasp
prototype remains intact; it is not yet replaced by this integration primitive.

Rate is a joint increment bound in rad/s, not a torque or physical velocity
guarantee. The host owns servo application, sensor timestamps, a wall deadline
and stopping actuators. The Rust timeout counts supplied simulation timesteps;
it is not a wall-clock watchdog. `arrived` means joint tracking within 0.02 rad
and commanded position within 1e-6 rad, never qualified grasp alignment.

On `music-room`, 2026-10-15, one bounded run in
`logs/jumper/jumper.pickup/rust-motion-demo-2026-10-15`:

- Open claw, LF shoulder target -0.20 rad from its starting command; other arm
  joint targets unchanged. This is a motion connection test, not a new grasp plan.
- Rust produced 49 checked commands at 0.45 rad/s maximum target progression.
  Measured arrival and a half-second hold completed; 107 physics-control steps
  were recorded as a 2.16-second MP4 using the shared ReplayVideo renderer.
- Smallest reported checked gap: 2 mm; no rejected clearance queries. Maximum
  support-foot drift: 0.747 mm. Motion-period maximum tilt: 1.317 degrees.
  Overall tilt peaked at 4.260 degrees during initial settling, before motion
  admission. Motion required the unchanged <=3-degree and support gates.
- Task-scene FK matched MuJoCo throughout: maximum point difference
  1.442222014787673e-16 m, direction-vector difference 9.142502916948183e-16.
- No post-reset pose writes drove the robot; Rust proposals passed through the
  existing Python action/servo mechanism and MuJoCo dynamics. Python retained
  scene construction, guard queries and rendering. This is not a browser WASM
  integration or a standalone Rust grasp controller.
- No closure, lift or pickup was attempted. The rejected grasp preparation
  remains unresolved. Adjacent/rigid interfaces and rear-only collision pairs
  are outside this guard's scope; no hardware safety claim is made.

TDD: four core tests failed against the stub, then passed; one rejection control
already passed. Four Python boundary tests failed before the export existed,
then passed. Current gates: 195 Rust core tests; 29 combined interface/app-play
tests passed (38 existing warnings). No full-framework or Python lint pass is
claimed. The driver, protocol, trace, report and MP4 are preserved together.
