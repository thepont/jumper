"""Hyper-parameters for jumper.gesture_paw.

Based on jumper.dance's settings, with lower initial exploration noise for the
narrow joint-tracking reward on this clip. Written out rather than imported so
that changing one clip's training does not retune the other.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config: `logs/<model>/jumper.gesture_paw/<date-time>`."""
    return jumper_ppo_baseline(
        experiment_name="jumper.gesture_paw",
        # Off, and not a tuning choice: the gesture is one-handed, and mirroring its
        # samples would train the average of offering the left paw and the right.
        # `symmetry.py` would also refuse the reference terms.
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # The action scale is 0.25 rad; std=1.0 injects 0.25 rad of joint noise,
        # nearly twice this clip's 0.13-rad tracking width. In short runs the
        # resulting action-change penalties outweighed tracking, and training
        # collapsed into short episodes. Start within half that width instead.
        init_std=0.25,
        # Half the baseline's: imitation needs less exploration than locomotion,
        # and a rising `Policy/mean_std` costs precision terms first.
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def runner_cls() -> type:
    """mjlab's motion-tracking runner, as jumper.dance: it logs the tracking metrics
    and the adaptive sampler's entropy, and bakes the clip into the exported ONNX."""
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    return MotionTrackingOnPolicyRunner
