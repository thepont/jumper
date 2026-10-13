# Foundations and scope

These are original operational summaries, not extracts from the books. The user
supplied both third editions on 2026-10-13. Chapters use printed book numbering;
the uploaded PDFs and extracted text are not required at skill runtime.

- Russell and Norvig, *Artificial Intelligence: A Modern Approach*, 3rd ed.:
  section 2.3 motivates PEAS and environment classification; chapter 15 covers
  temporal estimation; section 17.4 distinguishes partial observations from state
  and introduces probabilistic belief states; chapter 21 addresses learning and
  exploration; sections 25.3–25.6 connect robot perception, planning and movement.
  These foundations do not prescribe Jumper's PPO settings or deployment support.
- [Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/):
  termination and truncation differ for bootstrapping. Apply the semantics to the
  installed runner rather than copying an unrelated API signature.
- [ROS REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst):
  units and coordinate conventions make integration assumptions explicit. ROS
  conventions are a comparison reference, not authority to change this robot.

The reward-exploitation examples and routing are Jumper-specific design guidance.
Repository source and measured behaviour decide what is actually implemented.

## Skill authoring

The three workspace skills follow the [Agent Skills specification](https://agentskills.io/specification)
and [Anthropic's authoring guidance](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices):
precise discovery descriptions, concise workflows, references loaded as needed,
and validation against realistic requests. Book material is paraphrased into
procedures; manuals and upload-specific paths are not bundled into agent context.
