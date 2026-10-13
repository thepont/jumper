# Foundations and current practices

Original summaries based on sources checked on 2026-10-13. The user-supplied book
is identified by edition and chapter so the skill does not depend on its upload.

- Russell and Norvig, *Artificial Intelligence: A Modern Approach*, 3rd ed.,
  chapters 17 and 21: sequential reward, policy evaluation, exploration and
  generalization. This is foundational guidance, not a current PPO tuning manual.
- [Agarwal et al., Deep RL at the Edge of the Statistical Precipice, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html):
  few-run comparisons need uncertainty reporting; aggregate benchmark tools include
  performance profiles and interquartile means. Metric choice depends on scope.
- [Isaac Lab reproducibility](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html):
  seeds, physics configuration, hardware and version affect reproducibility.
  Its implementation and timing details are Isaac-specific; Jumper uses MuJoCo.
- [Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/):
  verify termination/truncation semantics when diagnosing value-target errors.

The staged run procedure, manifest fields and CLI example are adaptations to this
workspace. Inspect the checked-out parser, task and evaluator before executing;
these sources supply neither Jumper flags nor numerical success thresholds.
