# Pickup research compatibility review

Reviewed 2026-10-16 against source revision `2c8d17f` and the preserved
`rust-capture-continuity-2026-10-16` evidence. Four independent read-only agents
each examined one paper and compared it with the executed controller. This is
a compatibility assessment, not a reproduction of their published performance.

## Paper findings and decisions

| Reference | Useful principle | Compatibility decision |
| --- | --- | --- |
| Morrison, Corke and Leitner, [Closing the Loop for Robotic Grasping](https://arxiv.org/abs/1804.05172), RSS 2018; [PDF](https://arxiv.org/pdf/1804.05172v2), sections III–VI | Separate grasp proposals from feedback execution; continuously update the approach. | Retain Rust Tracking/JointMotion with numerical planning Adapters. Defer GG-CNN inference until actual depth sensing and proposal inadequacy are demonstrated. Top-down parallel-jaw width is not a direct command for Jumper's hinged claw. |
| Chaumette and Hutchinson, [Visual Servo Control Part I](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf), 2006, pp. 82–90, equations 1–21 | Explicit feature/pose errors, frames, Jacobian and feedback assumptions. | Adapt relative tool/object pose correction through existing IK. Do not apply an eye-in-hand camera-velocity Jacobian to the four arm joints: the modeled onboard camera is on the chassis. Delayed synthetic poses do not establish real image-based servoing or the paper's convergence conditions. |
| Ferrari and Canny, [Planning Optimal Grasps](https://people.eecs.berkeley.edu/~jfc/papers/92/FCicra92.pdf), ICRA 1992, pp. 2290–2295, sections 2–5 | Evaluate contact arrangements against forces and moments, not geometric centrality alone. | Defer a non-actuating, task-specific load-resistance diagnostic until the execution defects are repaired. Declare friction, COM uncertainty, force/moment units and shared pad/actuator budgets. Two ideal point contacts can support a particular load without full spatial force closure. No new success or closing gate is justified now. |
| Calandra et al., [More Than a Feeling](https://arxiv.org/abs/1805.11085), RAL 2018; [author PDF](https://www.robertocalandra.com/papers/2018_calandra_ral.pdf), sections III–VII, equation 1 | Close/probe, assess contact, conditionally adjust, then verify before lifting. | Adapt the sequencing principle only. Defer learned visuo-tactile adjustment and force optimisation: GelSight sensing and commanded fingertip force are not established on Jumper. Position-lead limiting is not measured force control. |

The reactive-grasp reviewer read the complete arXiv v2 author manuscript because
the proceedings PDF timed out; final-typesetting equivalence was not certified.
The contact-feedback reviewer used complete arXiv v2 HTML, not high-resolution
PDF figure inspection. The visual-servo and grasp-quality reviewers read their
complete primary PDFs. Publication years above are not crawler timestamps.

## Implementation order

All four reviews identify the same first integration defect: supervision reads
one detection stream while dispatch gates arm and finger targets on another.
The continuity trace directly recorded 15 withheld proposals in 144 tracking
ticks, with 25 object-validity disagreements overall. Low IK residuals or a new
grasp network cannot repair that command-delivery contract.

1. Provide one timestamped perception authority to the opt-in host and action
   term. Keep default training/replay behaviour and squeeze limiting unchanged.
   Test expiry, reset, dropout, accepted targets and old observation/checkpoint
   interfaces before another physical comparison.
2. Separately distinguish the next combined arm/jaw occupancy check from future
   lift feasibility. Preserve expected-contact versus forbidden-contact rules.
   Enter Closing before requiring load; normal first contact need not already
   be a qualified bilateral grip. Lift still needs its independent admission.
3. Qualify pickup and a predeclared continued-hold window independently. A latched
   pickup event is not evidence of subsequent retention.
4. Only then evaluate improved proposal quality, load-resistance diagnostics or
   contact-driven adjustment, with their sensor and model assumptions explicit.

Do not use a positive auxiliary central score as COM estimation, force closure
or hardware sensing. Keep all collision, support, rate and time limits; do not
resume stale routes or add blind closure to obtain a favourable result.

## Existing code and factory compatibility

`ClawActionCfg.detections_factory` is an opt-in Factory Method composition seam
for an existing sensor family, not another detector implementation. The action
term constructs it once; observations, dispatch and the host can reuse
`term.detections`. Absent a factory, construction still uses the original
`SimulatedDetections`. Existing methods, five-action layout, 13-value detection
observation, reset and squeeze-limiter paths remain in place.

Different closing Strategies may justify a factory later. No alternative closing
algorithm, NN, new force interface or automatic retry is introduced by this repair.
See [measured controls and robot-test results](../tasks/jumper/pickup/README.md).

## Accompanying code review

The [author GG-CNN repository](https://github.com/dougsm/ggcnn) distinguishes its
original `RSS2018` Keras branch from the later PyTorch `master` branch. Its README
links separate [original Kinova control](https://github.com/dougsm/ggcnn_kinova_grasping)
and [later ROS control](https://github.com/dougsm/mvp_grasp) repositories.
The commit-pinned implementation/licence review is still in progress. No upstream
code, model weights, dataset or dependencies have been installed or imported.
