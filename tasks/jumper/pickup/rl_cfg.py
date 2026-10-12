"""Initial pickup PPO settings; independent of every existing walking run."""
from ..common.ppo import jumper_ppo_baseline

DISCOUNT = .999


def agent_cfg():
    # At 50 Hz, gamma=.999 gives a ~20 s discount horizon for a ~21 s bench
    # sequence. These are starting choices, not a measured convergence claim.
    return jumper_ppo_baseline(
        experiment_name="jumper.pickup", actor_hidden_dims=(128, 128),
        critic_hidden_dims=(128, 128), activation='elu', obs_normalization=True,
        init_std=.6, std_type='scalar', entropy_coef=.002,
        learning_rate=3e-4, schedule='adaptive', desired_kl=.01, gamma=DISCOUNT, lam=.97,
        clip_param=.2, value_loss_coef=1., use_clipped_value_loss=True, max_grad_norm=1.,
        num_learning_epochs=4, num_mini_batches=4, num_steps_per_env=64,
        max_iterations=3000, save_interval=25, symmetry=False)


def runner_cls():
    from .runner import PickupRunner
    return PickupRunner
