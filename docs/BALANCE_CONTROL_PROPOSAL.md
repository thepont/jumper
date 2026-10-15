# Balance and payload control proposal

Status: proposed architecture and reference list, 2026-10-16. Not an implemented
or hardware-qualified balance controller. Keep pickup repair as the immediate
workstream; do not fold general balance behaviour into its grasp state machine.

## Responsibilities

Balance should remain active during standing, locomotion, reaching, lifting and
holding. It should respond to payload changes, external pushes and uneven ground
within an explicitly tested operating envelope. Separate software responsibilities
must not become physically independent arm and leg controllers.

| Owner | Responsibility | Information exchanged |
| --- | --- | --- |
| State estimation | Estimate body motion, contact/support state and available load information. | Timestamped states, uncertainty and sensor validity. |
| Pickup and manipulation | Track the object, request arm/gripper motion, establish grip and verify pickup/hold. | Intended arm trajectory, expected/measured load and uncertainty; receive motion constraints. |
| Balance and locomotion | Maintain feasible body/contact behaviour; choose posture, support forces and, when needed and feasible, foot placement. | Allowed body/arm motion envelope, degraded status, pause/recovery request. |
| Command arbitration | Emit one coordinated command after all applicable checks. | Requested, accepted and applied targets plus explicit rejection reasons. |

Pickup must not write individual support-leg targets behind the balance
controller's back. Balance must account for the arm configuration, movement and
load rather than treating it as an unrelated mechanism. Reuse existing sensing,
kinematic and action authorities; do not add a second actuator writer.

## Proposed interface contract

Manipulation supplies arm state and intended motion, payload estimates or bounded
unknowns, and any measured external wrench available. A wrench must identify its
reference point, frame, force units (N), moment units (N m), timestamp and
uncertainty. Never equate geometric object centre with centre of mass.

Balance returns a time-limited feasible motion envelope and explicit status.
Both sides use the same monotonic time and frame/joint conventions. Unknown load,
stale feedback, infeasible support and actuator saturation must be distinguishable
from a verified safe command. Define bounded recovery behaviour before execution;
a pause must not silently open the claw or certify successful pickup.

The balance implementation could use model-based control, learning, or a hybrid.
No solver or NN is selected here. MPC papers often assume torque/force interfaces
and state estimates that Jumper has not yet been shown to expose. First audit
hardware telemetry, actuator command modes, timing, gains and effort limits.
Motor current alone is not a calibrated object-weight sensor. Simulator contact
forces remain evaluator inputs unless equivalent deployable sensing is established.

## References and how they inform the proposal

Links checked on 2026-10-16. These notes describe the published abstracts or
official overview, not a completed reproduction or implementation audit. Link the
original publications rather than copying paper text or PDFs into the source tree.

### Hierarchical adaptive control for changing loads and terrain

Mohsen Sombolestan and Quan Nguyen, *Hierarchical Adaptive Loco-manipulation
Control for Quadruped Robots*, ICRA 2023; arXiv v2, 2023-03-11.
[Paper and metadata](https://arxiv.org/abs/2209.13145v2) ·
[PDF](https://arxiv.org/pdf/2209.13145v2).

Combines adaptive manipulation-force control with MPC that includes interaction
forces in the robot dynamics. Reports quadruped experiments with changing loads
and unknown terrain parameters. Closest reference for coordinating reusable
balance and manipulation responsibilities; its manipulation results do not
establish an object pickup or payload capacity for Jumper.

### Predictive ground force control

Jared Di Carlo, Patrick M. Wensing, Benjamin Katz, Gerardo Bledt and Sangbae Kim,
*Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*,
IROS 2018.
[MIT record and accepted manuscript](https://dspace.mit.edu/entities/publication/bc8c7e1e-5830-443f-a879-787947111fcf) ·
[DOI](https://doi.org/10.1109/IROS.2018.8594448).

Uses simplified three-dimensional dynamics to optimize ground-reaction forces
for a torque-controlled quadruped. A foundation for balance/contact-force planning,
not a ready-made grasp controller. Audit actuator compatibility before adopting
its low-level control assumptions.

### Counterpoint to poorly coordinated modular controllers

Zipeng Fu, Xuxin Cheng and Deepak Pathak, *Deep Whole-Body Control: Learning a
Unified Policy for Manipulation and Locomotion*, CoRL 2022.
[Paper and metadata](https://arxiv.org/abs/2210.10044v1) ·
[PDF](https://arxiv.org/pdf/2210.10044v1).

Proposes a unified learned whole-body policy and identifies coordination problems
with decoupled manipulation/locomotion pipelines. Treat this as a countercheck
against assuming modularity alone solves the dynamics. It does not require us
to replace the existing pickup controller with an NN.

### Boston Dynamics explanation of feedback across behaviours

Boston Dynamics, *Flipping the Script with Atlas*.
[Official technical overview](https://bostondynamics.com/blog/flipping-the-script-with-atlas/).

Explains online MPC adjustments to forces, posture and timing, including changes
across behaviour boundaries. Useful architectural context, not a research paper,
Spot source code, or a specification of Jumper's achievable recovery performance.

## TDD and BDD implementation sequence

1. Audit available sensors and command interfaces. Specify rates, delays,
   uncertainty, frames and failure semantics before choosing an algorithm.
2. Write shared contract tests for fresh/stale/invalid load messages, frame and
   unit validation, command ownership, expiry, infeasible requests and saturation.
   Verify that no rejected request leaks to actuators and no old route is resumed
   without renewed admission.
3. Qualify balance independently in bounded simulation: nominal standing,
   asymmetric payloads at different arm extensions, gradual and abrupt load
   changes, lateral/fore-aft pushes, slopes, uneven footholds and contact loss.
   Freeze magnitudes, durations, seeds and acceptance thresholds before each study.
4. Integrate a slow pickup and a separately measured continued-hold window.
   Test load transfer while leaving the ground, a push during lift/hold, and
   disagreement or loss of sensing. Balance may constrain or reject a lift;
   manipulation must not bypass it or release the payload implicitly.
5. Record falls, slips, support changes, body motion, actuator saturation,
   collision clearance and task outcomes independently. A low tilt or completed
   grasp alone does not establish dynamic balance. Preserve unsuccessful cases.

Use deterministic shared tests for message/arbitration policy and external bounded
probes for task-specific physical evidence, following the repository testing rule.
No training, disturbance trials, hardware actions or put-down behaviour are
authorized merely by adding this proposal. Current pickup evidence and remaining
handoff defects are recorded in [the pickup notes](../tasks/jumper/pickup/README.md).
