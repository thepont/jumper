"""Minimal MJX trainer for AMD-oriented acceleration experiments."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
from pathlib import Path
import time
from typing import Any

from .observer import EventSink, NullEventSink, ProbeEvent


@dataclass(frozen=True)
class MJXTrainConfig:
    iterations: int = 200
    horizon: int = 64
    learning_rate: float = 1.0e-3
    action_scale: float = 0.4
    action_penalty: float = 1.0e-3
    seed: int = 0
    log_every: int = 20
    optimizer: str = "grad"
    es_sigma: float = 0.02
    es_samples: int = 8
    es_batch_size: int = 4
    strip_visual_meshes: bool = False
    strip_noncolliding_geoms: bool = False
    collision_keep_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class MJXTrainReport:
    iterations: int
    final_loss: float
    checkpoint: Path
    model_path: Path


class MJXLinearPolicyTrainer:
    """Train a tiny linear policy directly through MJX dynamics.

    This is an isolated parallel track for AMD MJX bring-up, not a replacement for
    the existing warp/native PPO stack.
    """

    def __init__(self, *, sink: EventSink | None = None) -> None:
        self._sink = sink or NullEventSink()

    def train(
        self,
        *,
        model_path: Path,
        out_dir: Path,
        cfg: MJXTrainConfig,
    ) -> MJXTrainReport:
        if cfg.iterations <= 0 or cfg.horizon <= 0:
            raise ValueError("iterations and horizon must be positive")
        if cfg.optimizer not in {"grad", "es"}:
            raise ValueError("optimizer must be 'grad' or 'es'")
        if cfg.es_batch_size <= 0:
            raise ValueError("es_batch_size must be positive")

        jax, jnp, mujoco, mjx = _import_stack()
        out_dir.mkdir(parents=True, exist_ok=True)
        self._sink.on_event(
            ProbeEvent(
                name="training.started",
                payload={
                    "model": str(model_path),
                    "iterations": cfg.iterations,
                    "horizon": cfg.horizon,
                },
            )
        )

        t0 = time.perf_counter()
        mj_model = _load_model(
            mujoco=mujoco,
            model_path=model_path,
            strip_visual_meshes=cfg.strip_visual_meshes,
            strip_noncolliding_geoms=cfg.strip_noncolliding_geoms,
            collision_keep_patterns=cfg.collision_keep_patterns,
        )
        validate_trainable_model(mj_model)
        self._sink.on_event(
            ProbeEvent(
                name="training.stage",
                payload={"stage": "model_loaded", "seconds": time.perf_counter() - t0},
            )
        )
        t1 = time.perf_counter()
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)
        self._sink.on_event(
            ProbeEvent(
                name="training.stage",
                payload={"stage": "data_initialized", "seconds": time.perf_counter() - t1},
            )
        )

        t2 = time.perf_counter()
        model = mjx.put_model(mj_model)
        self._sink.on_event(
            ProbeEvent(
                name="training.stage",
                payload={"stage": "mjx_model_put", "seconds": time.perf_counter() - t2},
            )
        )
        t3 = time.perf_counter()
        data0 = mjx.put_data(mj_model, mj_data)
        self._sink.on_event(
            ProbeEvent(
                name="training.stage",
                payload={"stage": "mjx_data_put", "seconds": time.perf_counter() - t3},
            )
        )

        obs_dim = int(mj_model.nq + mj_model.nv)
        act_dim = int(mj_model.nu)
        key = jax.random.PRNGKey(cfg.seed)
        key, wk, bk = jax.random.split(key, 3)
        params = {
            "w": 0.01 * jax.random.normal(wk, (obs_dim, act_dim)),
            "b": jax.random.normal(bk, (act_dim,)) * 0.0,
        }

        def policy(p: dict[str, Any], obs):
            # Avoid backend BLAS matmul kernels that can be unavailable on some
            # ROCm stacks by keeping this as elementwise+reduce primitives.
            linear = jnp.sum(obs[:, None] * p["w"], axis=0) + p["b"]
            return cfg.action_scale * jnp.tanh(linear)

        def rollout_reward_fn(p: dict[str, Any]):
            def body(d, _):
                obs = jnp.concatenate([d.qpos, d.qvel], axis=0)
                act = policy(p, obs)
                d = d.replace(ctrl=act)
                d = mjx.step(model, d)
                # Dense locomotion starter reward, enough to prove MJX training loop.
                reward = d.qvel[0] - cfg.action_penalty * jnp.mean(act * act)
                return d, reward

            d_last, rewards = jax.lax.scan(body, data0, None, length=cfg.horizon)
            del d_last
            return jnp.mean(rewards)

        def loss_fn(p: dict[str, Any], key_in):
            del key_in
            avg_reward = rollout_reward_fn(p)
            return -avg_reward, avg_reward

        last_loss = None
        if cfg.optimizer == "grad":
            grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
            self._sink.on_event(
                ProbeEvent(name="training.stage", payload={"stage": "jit_built"})
            )
            for step in range(cfg.iterations):
                key, sub = jax.random.split(key)
                step_t0 = time.perf_counter()
                (loss, avg_reward), grads = grad_fn(params, sub)
                if step == 0:
                    self._sink.on_event(
                        ProbeEvent(
                            name="training.stage",
                            payload={
                                "stage": "first_grad_complete",
                                "seconds": time.perf_counter() - step_t0,
                            },
                        )
                    )
                params = jax.tree_util.tree_map(
                    lambda p, g: p - cfg.learning_rate * g, params, grads
                )
                last_loss = float(loss)
                if (step + 1) % max(1, cfg.log_every) == 0:
                    self._sink.on_event(
                        ProbeEvent(
                            name="training.iteration",
                            payload={
                                "step": step + 1,
                                "loss": float(loss),
                                "avg_reward": float(avg_reward),
                            },
                        )
                    )
        else:
            sigma = float(cfg.es_sigma)
            samples = int(cfg.es_samples)
            batch_size = int(min(cfg.es_batch_size, samples))
            eval_jit = jax.jit(rollout_reward_fn)
            self._sink.on_event(
                ProbeEvent(
                    name="training.stage",
                    payload={
                        "stage": "es_jit_built",
                        "sigma": sigma,
                        "samples": samples,
                        "batch_size": batch_size,
                    },
                )
            )

            def perturb_eval(p: dict[str, Any], eps_w, eps_b):
                plus = {"w": p["w"] + sigma * eps_w, "b": p["b"] + sigma * eps_b}
                minus = {"w": p["w"] - sigma * eps_w, "b": p["b"] - sigma * eps_b}
                return eval_jit(plus), eval_jit(minus)

            def batched_eval(p: dict[str, Any], w_chunk, b_chunk):
                return jax.vmap(perturb_eval, in_axes=(None, 0, 0))(p, w_chunk, b_chunk)

            v_perturb_eval = jax.jit(batched_eval)
            for step in range(cfg.iterations):
                key, wk, bk = jax.random.split(key, 3)
                eps_w = jax.random.normal(wk, (samples, obs_dim, act_dim))
                eps_b = jax.random.normal(bk, (samples, act_dim))
                step_t0 = time.perf_counter()
                grad_w_sum = jnp.zeros_like(params["w"])
                grad_b_sum = jnp.zeros_like(params["b"])
                reward_sum = jnp.zeros((), dtype=params["w"].dtype)
                for i in range(0, samples, batch_size):
                    w_chunk = eps_w[i : i + batch_size]
                    b_chunk = eps_b[i : i + batch_size]
                    r_plus, r_minus = v_perturb_eval(params, w_chunk, b_chunk)
                    advantage = (r_plus - r_minus) / (2.0 * sigma)
                    grad_w_sum = grad_w_sum + jnp.sum(
                        advantage[:, None, None] * w_chunk, axis=0
                    )
                    grad_b_sum = grad_b_sum + jnp.sum(
                        advantage[:, None] * b_chunk, axis=0
                    )
                    reward_sum = reward_sum + 0.5 * jnp.sum(r_plus + r_minus)
                if step == 0:
                    self._sink.on_event(
                        ProbeEvent(
                            name="training.stage",
                            payload={
                                "stage": "first_es_complete",
                                "seconds": time.perf_counter() - step_t0,
                            },
                        )
                    )
                inv = 1.0 / float(samples)
                grad_w = grad_w_sum * inv
                grad_b = grad_b_sum * inv
                params = {
                    "w": params["w"] + cfg.learning_rate * grad_w,
                    "b": params["b"] + cfg.learning_rate * grad_b,
                }
                avg_reward = reward_sum * inv
                try:
                    avg_reward_host = float(jax.device_get(avg_reward))
                except Exception:  # noqa: BLE001
                    avg_reward_host = 0.0
                last_loss = -avg_reward_host
                if (step + 1) % max(1, cfg.log_every) == 0:
                    self._sink.on_event(
                        ProbeEvent(
                            name="training.iteration",
                            payload={
                                "step": step + 1,
                                "loss": float(last_loss),
                                "avg_reward": float(avg_reward_host),
                            },
                        )
                    )

        ckpt = out_dir / "mjx_linear_policy_latest.npz"
        import numpy as np

        np.savez(
            ckpt,
            w=np.asarray(jax.device_get(params["w"])),
            b=np.asarray(jax.device_get(params["b"])),
            iterations=np.asarray(cfg.iterations),
            horizon=np.asarray(cfg.horizon),
            learning_rate=np.asarray(cfg.learning_rate),
        )
        self._sink.on_event(
            ProbeEvent(name="training.saved", payload={"checkpoint": str(ckpt)})
        )
        self._sink.on_event(
            ProbeEvent(
                name="training.finished",
                payload={"final_loss": float(last_loss if last_loss is not None else 0.0)},
            )
        )
        return MJXTrainReport(
            iterations=cfg.iterations,
            final_loss=float(last_loss if last_loss is not None else 0.0),
            checkpoint=ckpt,
            model_path=model_path,
        )


def validate_trainable_model(model: Any) -> None:
    """Reject MuJoCo models that cannot produce a trainable control policy."""
    actuator_count = int(getattr(model, "nu", 0))
    if actuator_count < 1:
        raise ValueError(
            "MJX training requires a model with at least one actuator; "
            "the supplied model contains none"
        )


def _import_stack():
    try:
        import jax
        import jax.numpy as jnp
        import mujoco
        import mujoco.mjx as mjx
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MJX stack unavailable: install jax+jaxlib and mujoco[mjx] compatible with your platform"
        ) from exc
    return jax, jnp, mujoco, mjx


def _load_model(
    *,
    mujoco,
    model_path: Path,
    strip_visual_meshes: bool,
    strip_noncolliding_geoms: bool,
    collision_keep_patterns: tuple[str, ...],
):
    if not strip_visual_meshes:
        if not strip_noncolliding_geoms:
            return mujoco.MjModel.from_xml_path(str(model_path))
        spec = mujoco.MjSpec.from_file(str(model_path))
        spec = _strip_noncolliding_geoms(spec, mujoco=mujoco)
        if collision_keep_patterns:
            spec = _keep_only_collisions(spec, mujoco=mujoco, patterns=collision_keep_patterns)
        return spec.compile()

    spec = mujoco.MjSpec.from_file(str(model_path))
    spec = _strip_visual_meshes(spec, mujoco=mujoco)
    if strip_noncolliding_geoms:
        spec = _strip_noncolliding_geoms(spec, mujoco=mujoco)
    if collision_keep_patterns:
        spec = _keep_only_collisions(spec, mujoco=mujoco, patterns=collision_keep_patterns)
    return spec.compile()


def _strip_visual_meshes(spec, *, mujoco):
    import numpy as np

    before = spec.compile()
    mesh_t = int(mujoco.mjtGeom.mjGEOM_MESH)

    def _collides(model, i: int) -> bool:
        return bool(model.geom_contype[i] or model.geom_conaffinity[i])

    ref = {
        "body_mass": np.array(before.body_mass),
        "body_ipos": np.array(before.body_ipos),
        "body_iquat": np.array(before.body_iquat),
        "body_inertia": np.array(before.body_inertia),
    }
    n_col_before = sum(_collides(before, i) for i in range(before.ngeom))
    doomed = {
        before.geom(i).name
        for i in range(before.ngeom)
        if before.geom_type[i] == mesh_t and not _collides(before, i)
    }
    if not doomed:
        return spec

    kept_meshes: set[str] = set()
    for i in range(before.ngeom):
        if before.geom_type[i] == mesh_t and before.geom(i).name not in doomed:
            did = int(before.geom_dataid[i])
            if did >= 0:
                kept_meshes.add(before.mesh(did).name)

    for body in spec.bodies:
        for g in list(body.geoms):
            if g.name in doomed:
                spec.delete(g)
    for m in list(spec.meshes):
        if m.name not in kept_meshes:
            spec.delete(m)

    after = spec.compile()
    for field, base in ref.items():
        got = np.asarray(getattr(after, field))
        if got.shape != base.shape or np.abs(got - base).max() > 1e-9:
            raise RuntimeError(
                f"stripping visual meshes changed {field}; model likely depends on geom-derived inertial"
            )
    n_col_after = sum(_collides(after, i) for i in range(after.ngeom))
    if n_col_after != n_col_before:
        raise RuntimeError("stripping visual meshes changed number of colliding geoms")
    return spec


def _strip_noncolliding_geoms(spec, *, mujoco):
    import numpy as np

    before = spec.compile()

    def _collides(model, i: int) -> bool:
        return bool(model.geom_contype[i] or model.geom_conaffinity[i])

    ref = {
        "body_mass": np.array(before.body_mass),
        "body_ipos": np.array(before.body_ipos),
        "body_iquat": np.array(before.body_iquat),
        "body_inertia": np.array(before.body_inertia),
    }
    n_col_before = sum(_collides(before, i) for i in range(before.ngeom))
    doomed = {
        before.geom(i).name for i in range(before.ngeom) if not _collides(before, i)
    }
    if not doomed:
        return spec

    for body in spec.bodies:
        for g in list(body.geoms):
            if g.name in doomed:
                spec.delete(g)

    after = spec.compile()
    for field, base in ref.items():
        got = np.asarray(getattr(after, field))
        if got.shape != base.shape or np.abs(got - base).max() > 1e-9:
            raise RuntimeError(
                f"deleting non-colliding geoms changed {field}; model likely depends on geom-derived inertial"
            )
    n_col_after = sum(_collides(after, i) for i in range(after.ngeom))
    if n_col_after != n_col_before:
        raise RuntimeError("deleting non-colliding geoms changed number of colliding geoms")
    return spec


def _keep_only_collisions(spec, *, mujoco, patterns: tuple[str, ...]):
    import numpy as np

    before = spec.compile()

    def _collides(model, i: int) -> bool:
        return bool(model.geom_contype[i] or model.geom_conaffinity[i])

    ref = {
        "body_mass": np.array(before.body_mass),
        "body_ipos": np.array(before.body_ipos),
        "body_iquat": np.array(before.body_iquat),
        "body_inertia": np.array(before.body_inertia),
    }
    keep = set()
    for i in range(before.ngeom):
        if not _collides(before, i):
            continue
        name = before.geom(i).name
        if any(fnmatch.fnmatch(name, p) for p in patterns):
            keep.add(name)

    for body in spec.bodies:
        for g in body.geoms:
            if g.name not in keep:
                g.contype = 0
                g.conaffinity = 0

    after = spec.compile()
    for field, base in ref.items():
        got = np.asarray(getattr(after, field))
        if got.shape != base.shape or np.abs(got - base).max() > 1e-9:
            raise RuntimeError(
                f"collision filtering changed {field}; model likely depends on geom-derived inertial"
            )
    return spec
