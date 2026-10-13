---
name: robot-task-design
description: Design or review a robot decision task before implementing rewards, observations, perception or planning. Use for task objectives, reward exploitation, partial observability, sensor requirements and choosing between a controller, planner and learned policy. Task scaffolding belongs to new-task; appearance and scene creation belong to the Design workflow.
---

# Robot task design

Turn the requested behaviour into a measurable decision problem. Read `CLAUDE.md`
and `docs/WORKFLOWS.md`; inspect the nearest task's `env_cfg.py`, `rl_cfg.py` and
observation/action terms before proposing another mechanism. Paths here are
repository-relative unless linked. Read [sources](references/sources.md) when
checking the underlying theory or making claims beyond the existing contracts.

## Establish the problem

Write a compact PEAS description: performance measures, environment, actuators
and sensors. Separate what counts as success from the training reward. Include
failure conditions and a bounded episode; record unresolved requirements instead
of silently choosing them. Proceed with reversible work that does not depend on
missing answers.

Describe state, actions, observations, transition assumptions, reset distribution,
reward and episode endings. Distinguish simulator state from information available
to the actor on the robot. A privileged critic may use training-only information;
the deployed actor needs a supported way to obtain every observation.

For each sensor-derived observation, record units, frame, shape, update rate,
latency, valid range and behaviour for missing, stale or invalid readings. Check
the existing sensor implementation and controller observation contract before
promising that a camera or depth input can deploy.

For manipulation or mirrored modes, also inspect the task's `deploy/lib.rs`, the
manifest's hook selection and control bindings: hook-side mirroring and claw
commands can change action semantics beyond the policy's action terms.

## Choose a mechanism that fits the information

- A known local response may need an existing controller or FSM transition.
- A goal with explicit transitions and collision constraints may need a planner.
- An uncertain transition model with repeated experience may justify learning.
- If indistinguishable observations require different actions, identify the hidden
  information. Consider an estimator, observation history or recurrent policy;
  verify support in the runner and export path before selecting one. Naming the
  problem a POMDP does not supply an implementable belief estimator.

State why the selected approach fits this task. Keep planning, perception and
actuation responsibilities explicit without introducing a new framework solely
to express the textbook taxonomy.

## Review incentives and endings

For each reward term, explain how an agent could increase it without completing
the task. Check doing nothing, oscillation, early reset, repeated partial progress,
object displacement out of view, and simulator-only assistance where relevant.
Measure completion, failures, limits and resource use separately from reward.

Distinguish success/failure termination from an external time limit. Trace the
actual timeout and bootstrap path through the environment and runner; do not
change the API to resemble Gymnasium. If the objective itself has a finite horizon,
account for remaining time in the state/observation design where needed.

For randomization, name the parameter, distribution, physically justified range,
application timing and held-out evaluation conditions. A curriculum teaches within
a changing distribution; it does not establish performance at its final range.

## Deliver and route implementation

Deliver the decision specification, measurable acceptance criteria, likely reward
exploits, unavailable observations and the smallest useful experiment. Record which
claims need simulation evidence and which need hardware evidence.

Use [new-task](../new-task/SKILL.md) to create a task,
[controls](../controls/SKILL.md) to bind controls,
[rl-experiment](../rl-experiment/SKILL.md) to compare trained policies and
[deploy](../deploy/SKILL.md) for export and hardware work. Keep tuning in each task
as `CLAUDE.md` requires. Shared contract changes need automated checks; task tuning
uses bounded training/replay and one-off probes outside the repository.
