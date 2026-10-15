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
