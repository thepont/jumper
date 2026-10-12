"""Initialize the paw PPO actor from the clip's own joint targets.

Collect real simulator observations while the recorded command drives the
ordinary absolute joint-position action. Fit the actor to those actions, save
a standard PPO checkpoint, then continue with ``scripts/train.py --checkpoint``.
The policy and replay action contracts are unchanged; no reference feed-forward
is present when the checkpoint runs.

For example::

    python3 -m tasks.jumper.gesture_paw.tools.pretrain_actor --backend mjx --device cuda:0 --output logs/jumper/jumper.gesture_paw/pretrain_paw.pt
    python3 scripts/train.py --task jumper.gesture_paw --backend mjx --device cuda:0 --checkpoint logs/jumper/jumper.gesture_paw/pretrain_paw.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjrl.backend.resolve import resolve
from mjrl.backend.select import use_backend
from tensordict import TensorDict

from tasks.jumper.common.constants import ACTION_SCALE
from tasks.registry import load_agent_cfg, load_env_cfg, load_runner_cls

TASK = "jumper.gesture_paw"


def reference_actions(
    reference: torch.Tensor, default: torch.Tensor, *, scale: float
) -> torch.Tensor:
    """Map absolute reference joints to the task's HOME-relative raw actions."""
    if reference.ndim != 2 or reference.shape != default.shape:
        raise ValueError("reference and default joints must have the same 2-D shape")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("action scale must be finite and positive")
    return (reference - default) / scale


def pretrain(
    *,
    output: Path,
    num_envs: int,
    rollout_steps: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    backend: str = "native",
    device: str = "cpu",
) -> tuple[float, float]:
    """Collect reference-driven states, fit the actor, and save a PPO checkpoint."""
    if min(num_envs, rollout_steps, epochs, batch_size) < 1:
        raise ValueError("worlds, rollout steps, epochs and batch size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if output.suffix != ".pt" or output.exists() or not output.parent.is_dir():
        raise ValueError("output must be a new .pt file in an existing directory")

    torch.manual_seed(0)
    resolved = resolve(backend=backend, device=device, num_envs=num_envs)
    use_backend(resolved)
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = num_envs
    agent_cfg = load_agent_cfg(TASK)
    agent_cfg.logger = "tensorboard"
    runner_cls = load_runner_cls(TASK)
    if runner_cls is None:
        raise RuntimeError(f"{TASK} has no runner")

    env = ManagerBasedRlEnv(cfg=cfg, device=resolved.device)
    try:
        wrapped = RslRlVecEnvWrapper(env)
        runner = runner_cls(wrapped, asdict(agent_cfg), str(output.parent), resolved.device)
        actor = runner.alg.actor
        robot = env.scene["robot"]
        command = env.command_manager.get_term("motion")
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        obs = wrapped.get_observations()
        with torch.inference_mode():
            for _ in range(rollout_steps):
                target = reference_actions(
                    command.joint_pos, robot.data.default_joint_pos, scale=ACTION_SCALE
                )
                observations.append(obs["actor"].clone())
                actions.append(target.clone())
                obs, _, _, _ = wrapped.step(target)

        inputs = torch.cat(observations, dim=0)
        labels = torch.cat(actions, dim=0)
        permutation = torch.randperm(len(inputs), device=inputs.device)
        cut = max(1, int(len(inputs) * 0.9))
        if cut == len(inputs):
            raise ValueError("need at least two collected examples for holdout")
        training, validation = permutation[:cut], permutation[cut:]
        actor.update_normalization(
            TensorDict({"actor": inputs[training]}, batch_size=[len(training)])
        )
        optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)

        for _ in range(epochs):
            order = training[torch.randperm(len(training), device=training.device)]
            for indices in order.split(batch_size):
                batch = TensorDict({"actor": inputs[indices]}, batch_size=[len(indices)])
                predicted = actor(batch)
                loss = torch.square(predicted - labels[indices]).mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite imitation loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()

        with torch.inference_mode():
            held_out = TensorDict({"actor": inputs[validation]}, batch_size=[len(validation)])
            predicted = actor(held_out)
            mse = float(torch.square(predicted - labels[validation]).mean())
            zero_mse = float(torch.square(labels[validation]).mean())
        if not math.isfinite(mse) or mse >= min(0.1, zero_mse / 10):
            raise RuntimeError(
                f"reference imitation did not fit: MSE={mse:.4f}, zero={zero_mse:.4f}"
            )
        runner.save(str(output))
        return mse, zero_mse
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("native", "mjx"), default="native")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    mse, zero_mse = pretrain(
        output=args.output,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backend=args.backend,
        device=args.device,
    )
    print(
        f"[paw pretrain] saved {args.output}; held-out MSE {mse:.4f} vs zero-action {zero_mse:.4f}"
    )


if __name__ == "__main__":
    main()
