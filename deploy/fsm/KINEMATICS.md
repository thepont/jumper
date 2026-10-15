# Portable grasp-controller plan

## Pickup and secure hold, without placement (2026-10-16)

`capture::Capture` owns the opt-in capture phase policy in Rust. Its exhaustive
State enum is Tracking / Closing / Lifting / Verified / Stopped. There is no
put-down, release or tray-placement state. The legacy Python gate remains for
reproducing earlier experiments; the Rust-backed host does not call it for
capture decisions. Remove that compatibility path only after its old evidence
can still be reproduced and the replacement is physically qualified.

The host composes `Tracking`, `Capture`, existing IK/path Strategies and the
existing action/squeeze limiter. Fresh relative alignment plus a separately
admitted grasp can begin closing immediately, including while the target moves.
Waypoint arrival is not the closing criterion. Temporary pre-contact loss can
pause jaw advancement and recover inside the original closing deadline; contact
prohibits blind realignment. An unanticipated tracking-phase contact, malformed
feedback, safety failure or deadline latches stop. Neither alignment recovery
nor loaded-contact samples reset phase deadlines.

Loaded capture needs the configured consecutive count AND duration; gaps over
40 ms reset that evidence. Only an independent physical pickup witness can enter
Verified after lifting. Verified continues monitoring qualified load, sensing
and plan validity if the host keeps stepping: a dropped grip cannot masquerade
as a secure hold. This gate neither computes nor fabricates force/clearance
evidence, and is not itself an actuator or hardware safety certificate.

The Python Adapter exports `Capture(policy_json, start_ns)`, with policy fields
`approach`, `close`, `lift`, `recovery`, `loaded_samples`, `loaded_duration`.
Durations use the same `{secs, nanos}` representation as tracking. Call
`advance(now_ns, (fresh, plan, aligned, contact, loaded, pickup, unsafe))` to obtain
`(phase, advance_jaw)`. Flags must be actual booleans; malformed input consumes
the adapter. The host must obey `advance_jaw`, not infer jaw movement from phase
alone. All combined arm/finger proposals still need corridor admission before
dispatch.

`NativeTrackingPlanner` is a task-owned simulator/IK/path Adapter, not another
motion controller. It checks the current arm route independently from future
grasp feasibility. Requiring a complete future frozen-object closure/lift for
every outer approach update caused hold starvation in the first replay. Later
grasp admission remains mandatory before closing. Pregrasp and contact offsets
are explicit separate proposals, not altered acceptance tolerances. The adapter
reuses the existing multi-start IK, convex SOAP geometry and path checker.
Tracking planner requests now also include measured `tool_position` and
`tool_axis`, validated in Rust, to judge the outer-to-terminal path transition.

Expected pad/object contact uses the new opt-in
`GeometryClearance.check_contact_depth(qpos, max_penetration_m=...)`. It reuses
the positive-clearance query and complete-pose validation. An unresolved zero
needs either signed compiled-contact depth or a separating-plane certificate;
missing/disabled-mask evidence and excessive depth reject. Forbidden robot
contacts continue using positive clearance. This is neither a force witness
nor permission to disregard object/finger-body or robot self-collision.

The bounded external integration host and physical outcomes are recorded in
`tasks/jumper/pickup/README.md`. This API alone does not establish a Jumper
pickup, long-term retention, arbitrary moving-object interception or a catch.

## Recoverable observed-target tracking (2026-10-16)

`tracking::Tracking<P>` is an opt-in Application Service around the existing
`JointMotion` and `SceneMotion`. A `Planner` Strategy returns freshly checked
joint waypoints or `None` when the current target is temporarily infeasible.
The geometry Strategy checks every measured-to-proposed corridor, including
held targets. No new IK, collision engine, trajectory interpolation or servo
rate algorithm is introduced. Existing callers are unchanged.

Behaviour:

- Follow new reachable observations rather than terminating at waypoint arrival.
- Discard the old route before planning for a new capture/scene. If planning is
  infeasible, hold under a new geometry check and retry on new evidence. Do not
  reuse the old route or busy-retry an unchanged packet.
- Missing, stale or reordered target observations produce checked holds. A fresh
  packet can reacquire the target within the same session and absolute deadline.
- Planner infrastructure errors, malformed input, unsafe command corridors and
  expired sessions latch stop. `None` is the recoverable planning verdict; an
  exception is not silently reclassified as infeasibility.
- Replanning never resets the original monotonic deadline or accumulated control
  budget. Unsampled time does not count towards relative-alignment dwell.

All target/tool points and directions are in the WORLD frame: metres and unit
vectors. Feedback must contain measured FK, not FK of requested joint targets.
Capture timestamps and scene revisions identify immutable observations. Reusing
a capture time with different contents is invalid. Scene coverage remains the
host's responsibility: fresh target detection does not certify fresh obstacle
coverage. Planning/checking must be bounded, updates and dispatch serialized,
and the host must recheck freshness after slow work before physical dispatch.

`Following`, `Holding` and `Aligned` are typed command statuses. `Aligned` means
the measured hand/object RELATIVE pose satisfies the configured limits, with an
optional relative-error dwell. It NEVER requires a stationary object; dwell may
be zero. It is not permission to close or evidence of capture. The grasp phase
must still check closure geometry, relative velocity and closure/sensing latency,
then measure contacts and the canonical lift witness. Fast interception outside
the observed pose envelope is not implemented by this supervisor.

The Python extension exposes this SAME Rust authority:

```python
session = mjrl_fsm.Tracking(policy_json, start_ns, planner)
joints, status = session.step(
    (current, measured, tool_point_w, tool_axis_w, now_ns, dt),
    (goal_w, inward_w, scene_revision, captured_at_ns),  # or None
    corridor_clear,
)
```

The policy JSON has exactly `lower`, `upper`, `rate`, `timeout`, `freshness`,
`position_tolerance`, `axis_tolerance` and `settle`. Durations use Rust/Serde
`{"secs": integer, "nanos": integer}`; the direction tolerance is radians.
The planner receives a dictionary with `position`, `axis`, `current`, `measured`,
`now_ns`, `observed_at_ns`, `scene_revision` and `remaining_ns`. It returns a list
of checked joint waypoints or `None`. Geometry must return an actual boolean.
Malformed packets consume the adapter session before parsing can fail, preventing
silent resumption. The adapter contains no tracking or retry policy of its own.

This does not yet replace the pickup runner's stop-at-arrival path, export a
browser tracking binding, dispatch hardware, add speed scheduling, or establish
successful physical catching. A clearance-dependent speed governor needs actual
hazard distances, latency and stopping constraints; target distance alone is not
an adequate speed policy. The existing maximum joint-command rate is retained.

## Scene-bound motion proposals (2026-10-15)

`joint_motion::SceneMotion` is an opt-in Decorator around `JointMotion`, not a
second planner or servo. It binds one admitted joint path to a host scene
revision and a positive observation-age budget (`std::time::Duration`). The
existing geometry callback remains the Strategy for checking the whole
measured-to-command corridor. Existing callers of `JointMotion` are unchanged.

Behaviour contract:

- Given a checked path, when the scene revision changes, no new command is
  returned and the geometric checker is not called for the obsolete path.
- Given an unchanged scene, stale/future timestamps, clock rollback or an older
  observation stop the motion. Repeated fresh sensor packets are allowed because
  sensor and control rates differ. The exact freshness boundary is inclusive.
- Once rejected, returning to the old revision cannot resume the motion. Recovery
  requires a newly checked path and a new motion instance, not clearing a flag.
- With a fresh unchanged scene, existing synchronized joint rates, feedback gates,
  geometric checks and latched failures retain their existing authority.

The host owns revision assignment: advance it for relevant target/obstacle changes,
use a new identity after a scene-source restart, and never reuse an old revision.
All times must share one monotonic epoch. An observation stamp must represent
coverage of the required scene, not merely a fresh target detection while obstacle
data has expired. Unknown coverage must fail closed at the host boundary. Scene
updates and dispatch must be serialized or rechecked immediately before dispatch;
this borrowed snapshot does not prevent concurrent updates during a slow check.
The host must implement the physical hold/braking response to a rejected proposal.

Measured validation: three new behavioural tests failed against pass-through
behaviour, then all four new tests and all 195 existing core tests passed (199
total). Cases cover revision invalidation, stale/future data, ordering, exact age
boundary, repeated packets, zero age budget, rate preservation and guard-stop
latching. The 250ms budget used in tests is a test fixture, not a robot safety
recommendation. *The Art of Unit Testing*, section 1.6, informed the small
red/green steps; no book content or assets are included in the repository.

This API is not yet wired to a perception stream, WASM binding or hardware
dispatch. The Python boundary and injected-scene integration tests are described
below. It does not implement obstacle detection, target tracking, re-planning,
braking-distance guarantees or successful pickup. Next integrate a real scene
producer into the pickup simulation host, then assess a library-backed local controller.
Reuse the existing OpenRR `k` FK adapter and collision checker; do not build a new
robot model or collision engine. Physics, safety margins and saved policies remain
unchanged.

### Python boundary and simulated dispatch

The existing PyO3 extension now exports:

```python
motion = mjrl_fsm.SceneMotion(points, lower, upper, rate, timeout,
                             revision, max_age_ns)
candidate = motion.step(current, measured, dt,
                        (revision, observed_at_ns), now_ns, corridor_clear)
```

Joint values and `dt` retain `JointMotion`'s radians/seconds contract. Scene
timestamps use integer nanoseconds in one monotonic epoch; scene identity and
times reject booleans, negative values, floats and overflow. The scene packet is
a two-element tuple. Only a successful return may be dispatched. `arrived` is
false after rejection. A malformed packet, failed conversion, invalid feedback
or geometry callback error consumes the binding's motion, so a later good packet
cannot silently resume it. Callback results must be actual Python booleans.
The wrapper delegates scene validity and motion computation to the Rust core;
it does not recreate either algorithm in Python.

`SceneMotion.hold` has the same input/scene/guard contract as `step`, but proposes
the current targets without consuming or advancing a waypoint. It uses the same
feedback validation, scene checks, geometry callback, elapsed-motion budget and
stop latch as normal advancement. Holds therefore cannot extend a motion's budget
or bypass stale coverage. This is a target hold; the host still checks the entire
measured-to-held corridor and supplies the actual physical response. Five Python
hold cases were observed failing before the API existed, then passed; an added
Rust case pins non-advancement even at a reached waypoint (200 core tests pass).

Measured locally, 2026-10-15: 15 new binding cases failed first because the export
was absent, then passed. Three further MuJoCo integration cases start a single
hinge moving through the real extension, inject target-revision, obstacle-revision
or expired-coverage events, and verify that rejection leaves the actuator's prior
target unchanged without invoking obsolete geometry checks. The obstacle fixture
moves a mocap obstacle; events are injected, not detected by a camera. The fixture
uses MuJoCo distance queries and `implicitfast` for its small-inertia servo:
explicit integration at the initial test gains was unstable before the event.
This changed only the synthetic test fixture, not the Jumper model or margins.

Validation: 33 kinematics/scene-boundary tests plus 14 app-play regressions passed;
the legacy FSM extension module skipped because its exported tripod policy is
absent (47 passed, 1 skipped, 38 existing warnings). All 199 Rust core tests and
the host web-feature check passed. Ruff is unavailable in this environment.
Clippy remains blocked by the unchanged `control.rs:327` `never_loop` error;
PyO3-generated conversion warnings also occur on the new binding as on existing
bindings. No clean lint or full-suite result is claimed.
These are command-dispatch tests, not measurements of braking distance, dynamic
obstacle avoidance, whole-robot grasp completion or the full application using
live scene updates. Existing app consumers remain unchanged until a scene
producer supplies the documented coverage and revision guarantees.

### Observed pickup approach coordinator (2026-10-16)

`tasks/jumper/pickup/grasp_execution.py` composes `ObjectTargetTracker`, an admitted
Rust `SceneMotion` and the host's existing clearance callback. The module returns
an immutable joint proposal plus tracked world goal/axis. It owns sequencing only;
sensor capture, transforms, IK, collision geometry and command dispatch remain
with their existing owners. It imports neither a simulator nor a Rust extension
loader. The host injects the actual motion instance, keeping build/loading policy
outside task logic. This is a small Application Service using the existing State,
Decorator and Strategy boundaries, without another motion or stop implementation.

`ObservedApproach.command` takes current/measured joints and `dt`, a monotonic
clock, measured world position/rotation and capture time, independent scene
coverage time/revision, contact and the clearance callback. It reuses the tracker's
original-plan 8 mm/10 degree envelope. The older target/coverage timestamp is
passed to Rust in integer nanoseconds; fresh target data cannot mask expired
coverage. The host owns coverage validity and revision ordering. Expired/malformed
observations, scene changes and unexpected contact go through the Rust stop latch. This executor
is for pre-contact movement; closing and lifting require their own phase admission.

Within the admitted envelope, Rust follows the existing joint waypoints and the
returned world goal tracks the observation. This does not itself refit IK or steer
the waypoints toward that goal; terminal correction remains a separate existing
planner operation. The adapter cannot automatically replan or restart a stopped
motion. The host must serialize scene reads/checks/dispatch and implement holding.

Fifteen external task-level cases were observed red before the module existed,
then passed against the actual extension. They pin normal proposals, transformed
goal updates, target/revision change, contact, missing/expired/future coverage,
malformed clocks/poses, repeated/reordered packets and geometric rejection.
Combined with shared boundary/app-play checks: 62 passed, 38 existing warnings.
One bounded known-bench replay then exercised 17 Rust commands from real delayed,
noisy synthetic target packets before an estimated-motion rejection. See the
pickup README and its preserved report for physical outcomes and limits.

#### Bounded confirmation holds (2026-10-16)

A same-seed sensor audit separated a 0.010 mm physical object displacement from
an 11.095 mm estimated shift; frame reconstruction error stayed below 4.15e-8 m.
The coordinator now reuses `PlanPoseGate` before committing a pose to the tracker.
`ObjectTargetTracker.assess` previews position/goal/rotation errors against the
original plan without updating or latching it. The same assessment owns the
existing 8 mm/10 degree decision in normal tracker updates. `PlanPoseGate` accepts
that boolean decision, retaining its quantified-position API for existing callers.

A suspect or missing packet produces `ApproachCommand.holding=True` through
Rust's `hold` operation, with the last confirmed capture timestamp. It never
advances a waypoint, refreshes confirmation from rejected/duplicate data, or
extends the motion budget. New acceptable data may continue an unexpired,
unstopped motion after a fresh geometry check. Expiry, scene/geometry errors or
contact still consume it. The host must check and dispatch held proposals through
the same measured-to-target corridor as moving proposals. No smoothing, noise
reduction, relaxed tolerance, or automatic replay of a stopped plan was added.

Two saved-noise/lifetime regressions and three angle/goal/missing-packet cases
were observed failing before their respective corrections. With legacy tracker,
confirmation, sensor, extension and app checks, 140 Python cases passed (38
existing warnings); all 200 Rust core tests and the host web-feature check passed.
The final known-bench replay reached the approach endpoint after 100 proposals,
15 of them checked holds. Saved-state position error was 5.479 mm against the
5 mm closing gate, so this is approach completion with terminal correction still
required. Closing, lifting and pickup were not executed.

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
