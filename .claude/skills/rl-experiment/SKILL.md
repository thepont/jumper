---
name: rl-experiment
description: Plan, run and evaluate bounded reinforcement-learning experiments in Jumper, including reward or curriculum tuning, policy comparisons and robustness evaluation. Use when deciding whether a trained policy improved or diagnosing learning failure. Environment repair belongs to setup-env; export and robot deployment belong to deploy.
---

# RL experiments

Produce evidence that answers a stated question. Read `CLAUDE.md`,
`docs/USAGE.md` and the relevant task configs. Run commands from the repository root
in its training environment. Read [sources](references/sources.md) for statistical
or reproducibility claims. Use [setup-env](../setup-env/SKILL.md) if the environment
is broken, rather than repairing it incidentally inside an experiment.

## Make the comparison interpretable

State one hypothesis, its baseline, the intended change, outcome metrics and a
bounded compute budget. Preserve unrelated configuration. For locomotion siblings,
check the parity requirements in `CLAUDE.md` before altering comparisons.

Fix the evaluation protocol before selecting the winning checkpoint: commands,
initial states, episode count or simulated duration, noise, terrain and failure
criteria. Hold evaluation scenarios apart from tuning. Use multiple independent
training seeds when the budget supports an improvement claim; many episodes from
one trained policy are not independent training runs. A single run supports a
smoke check or tentative observation, with that limitation stated.

Record the repository commit and dirty diff, resolved task/runner configuration,
seed settings and where they are applied, dependency/simulator versions, backend,
device, hardware, environment count, physics/control rates, rollout length,
iterations, wall time and exact checkpoint path. Do not invent a `--seed` flag:
inspect the current parser and runner config. Same seed alone does not establish
determinism across devices, physics engines or versions.

## Run in stages

Resolve with `train.py --list` and `--dry-run`; inspect `--help` before using an
unfamiliar option. Use a bounded smoke run to detect non-finite observations,
actions or rewards, invalid resets and ineffective commands before a long run.
The smoke run's small batch settings are not evidence for tuned learning quality.

Monitor reward components and independent outcome metrics, termination reasons,
command tracking and limit violations. Stop at the stated budget or a defined
failure condition, preserve logs, and report failures rather than repeating until
one run looks successful. Resume is continuation of a run, not a fresh seed;
`--max-iterations` counts additional iterations on resume.

Use an explicit checkpoint for comparisons; the newest-checkpoint default can
silently select a different policy. For a compatible velocity task, this is an
example after replacing the paths and choosing a command inside its trained range:

```bash
python scripts/play.py --task jumper.tripod --checkpoint <model.pt> \
  --backend native --device cpu --headless --steps 500 \
  --command 0.1 0 0 --evaluation-out <evaluation.json> \
  --replay-info-out <replay-info.json>
```

The existing fixed-command evaluator reports velocity errors and resets. Inspect
its output contract; it is not a generic evaluator for manipulation or every task.
Keep per-run outputs in distinct paths. Record observation noise and replay physics
settings explicitly: replay is not automatically the training distribution. Use
`play.py --video` for recordings; inspect video alongside metrics.

## Interpret and hand over

Preserve per-run results and report sample counts, failures, central tendency and
variability. Use interval estimates appropriate to the independent sampling unit;
do not treat correlated timesteps as independent samples. For a broad benchmark,
consider performance profiles or interquartile means; do not impose them on a
single-task smoke run. Label best-seed or best-checkpoint selection explicitly.

Report whether the hypothesis is supported, inconclusive or contradicted, with
commands, artifact paths and remaining gaps. Keep task-specific probes outside the
repository and measured tuning evidence beside the decision. Add regression tests
when shared machinery changes. Use [deploy](../deploy/SKILL.md) for transfer:
simulation robustness and a successful replay do not establish hardware readiness.
