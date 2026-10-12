"""Bounded CPU manipulation experiment, NOT a trained or vision-driven policy.

One floor object, one bin, free-standing robot, and real collision/friction.
Only reset may write poses. The controller writes rate-limited arm/finger servo
targets. Object truth is isolated in an explicitly synthetic estimate producer
and in the evaluator; kinematics uses ideal robot pose and joint readings.
The producer is NOT a calibrated camera simulator.

Run from the repository root with the editable environment:
  python -m tasks.jumper.five_foot.tools.pickup_check --simulated-estimates
Add --video demos/pickup-check.mp4 to use the shared replay recorder.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from functools import partial
import itertools
import json
import math
from pathlib import Path

import numpy as np

TRAY_INNER_HALF = .065


def bench_tray():
    """Same physical tray construction, sized to a stationary arm's workspace."""
    from tasks.jumper.five_foot.objects import BIN_WALL_T, _bin_spec

    spec = _bin_spec()
    for geom in spec.geoms:
        for axis in (0, 1):
            if geom.size[axis] > BIN_WALL_T:
                geom.size[axis] = TRAY_INNER_HALF + BIN_WALL_T
            if geom.pos[axis] != 0:
                geom.pos[axis] = math.copysign(TRAY_INNER_HALF + BIN_WALL_T / 2,
                                              geom.pos[axis])
    return spec


def config(seed=7):
    from tasks.jumper.five_foot.claw import ARM_JOINTS, GRIPPER_OPEN
    from tasks.jumper.five_foot.env_cfg import env_cfg
    from tasks.jumper.five_foot.objects import (
        OBJECTS_NCONMAX, OBJECTS_NJMAX, PROP_CONE, PROP_IMPRATIO, SOLIDS,
        _place_props, _place_the_robot, _prop, _solid_spec,
    )

    cfg = env_cfg(play=True)
    cfg.seed = seed
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 60
    cfg.rewards = {}
    cfg.curriculum = {}
    # No competing teleop/hold writer or camera rays no controller consumes.
    cfg.events.pop("gripper_teleop", None)
    cfg.events.pop("claw_hold", None)
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "tof")
    ready = dict(zip(ARM_JOINTS, (-.88, -1.57, -.524, .23), strict=True))
    cfg.events["hold_carried_arm"].params["ranges"] = (
        tuple((ready[j], ready[j]) for j in ARM_JOINTS) + ((GRIPPER_OPEN, GRIPPER_OPEN),)
    )
    soap = next(s for s in SOLIDS if s.name == "soap")
    props = {
        "soap": _prop(partial(_solid_spec, soap), (.23, .10, .051)),
        "dropbin": _prop(bench_tray, (.23, .245, .0)),
    }
    cfg.scene.entities.update(props)
    _place_props(cfg, props)
    _place_the_robot(cfg)
    cfg.sim.nconmax = max(int(cfg.sim.nconmax or 0), OBJECTS_NCONMAX)
    cfg.sim.njmax = max(int(cfg.sim.njmax or 0), OBJECTS_NJMAX)
    cfg.sim.mujoco.cone = PROP_CONE
    cfg.sim.mujoco.impratio = PROP_IMPRATIO
    return cfg


class SimulatedEstimates:
    """10 Hz, 60 ms latency, noisy estimates: explicitly privileged fixture.

    This only approximates a front sensor's field of view. No rendering,
    occlusion, detector, RGB/depth calibration or hardware validation is claimed.
    Retains the frame in which the measurement was made so old coordinates are
    never silently treated as fresh robot-frame coordinates.
    """

    def __init__(self, *, noise_m, seed, dropout):
        self.noise_m = noise_m
        self.rng = np.random.default_rng(seed)
        self.dropout = dropout
        self.pending = []
        self.latest = None
        self.next_exposure = 0.

    def sample(self, now, world_point, base_pos, base_rot):
        from mjrl.manipulation import TargetEstimate

        if now + 1e-9 >= self.next_exposure:
            self.next_exposure = now + .1
            relative = base_rot.T @ (world_point - base_pos)
            visible = (relative[0] > .05 and
                       abs(math.atan2(relative[1], relative[0])) < math.radians(55))
            estimate = None
            if visible and not self.dropout:
                estimate = TargetEstimate(
                    tuple(relative + self.rng.normal(0, self.noise_m, 3)), now, .9
                )
            self.pending.append((now + .06, estimate, base_pos.copy(), base_rot.copy()))
        while self.pending and self.pending[0][0] <= now + 1e-9:
            _, estimate, pos, rot = self.pending.pop(0)
            self.latest = None if estimate is None else (estimate, pos, rot)
        if self.latest is None:
            return None
        estimate, pos, rot = self.latest
        if not estimate.usable(now, max_age_s=.25, min_confidence=.8):
            return None
        return pos + rot @ np.array(estimate.position)


def run(args):
    import mujoco
    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend
    from mjrl.manipulation import PointIK

    from tasks.jumper.five_foot.claw import (
        ARM_JOINTS, FINGER_JOINT, GRIPPER_CLOSED, GRIPPER_OPEN,
    )
    from tasks.jumper.five_foot.mdp.grasp import Claw
    from tasks.jumper.five_foot.mdp.gripper import squeeze_limited
    from tasks.jumper.five_foot.objects import BIN_WALL_T

    res = resolve(backend="native", device="cpu", num_envs=1)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv

    with ExitStack() as stack, torch.inference_mode():
        env = ManagerBasedRlEnv(cfg=config(args.seed), device="cpu")
        stack.callback(env.close)
        env.reset(seed=args.seed)
        model = env.sim.mj_model
        robot, soap, tray = (env.scene[n] for n in ("robot", "soap", "dropbin"))
        arm = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
        finger = robot.find_joints([FINGER_JOINT])[0]
        claw = Claw(env)
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/LF_palm_link")
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
        object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soap/soap")
        tray_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dropbin/dropbin")
        if min(palm, base, object_body, tray_body) < 0:
            raise ValueError("required robot bodies are absent")
        from tasks.jumper.five_foot.objects import SOLIDS
        solid = next(s for s in SOLIDS if s.name == "soap")
        object_corners = np.array(list(itertools.product(*zip(*solid.bounds, strict=True))))
        measured = mujoco.MjData(model)

        def refresh():
            measured.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mujoco.mj_kinematics(model, measured)

        refresh()
        # A known-object grasp offset, derived from the existing jaw geometry.
        # This does not set or move the object. Orientation handling is limited
        # to the upright soap fixture; arbitrary object poses are not supported.
        seat = claw.seat(env, "soap")[0].cpu().numpy()
        local = measured.xmat[palm].reshape(3, 3).T @ (seat - measured.xpos[palm])
        ik = PointIK(model, "robot/LF_palm_link", tuple(f"robot/{n}" for n in ARM_JOINTS[:3]),
                     local)
        zero = torch.zeros(env.action_space.shape)
        target_arm = robot.data.joint_pos[0, arm].clone()
        video = None
        if args.video:
            from mjrl.viewer.video import ReplayVideo
            video = stack.enter_context(ReplayVideo(
                env, args.video, fps=25, width=800, height=600,
                camera_distance=1.15, azimuth=135, elevation=-35,
                follow=False, lookat_height=.07,
            ))
        source = SimulatedEstimates(noise_m=args.noise_mm / 1000, seed=args.seed,
                                    dropout=args.sensor_dropout)
        bin_source = SimulatedEstimates(noise_m=args.noise_mm / 1000, seed=args.seed + 1,
                                        dropout=args.sensor_dropout)
        stage, entered = "settle", 0.
        transitions = [{"stage": stage, "at_s": 0.}]
        initial_z = float(soap.data.root_link_pos_w[0, 2])
        peak_z = initial_z
        picked = False
        reason = "deadline"
        held_goal = None
        observed_floor_z = None
        close_open = args.negative_control
        dt = env.step_dt
        max_error = 0.
        max_joint_step = 0.
        last_error = None
        stale_steps = 0
        settled_in_bin = 0
        tilt_at_close = None

        def transition(next_stage, now):
            nonlocal stage, entered
            stage, entered = next_stage, now
            transitions.append({"stage": stage, "at_s": round(now, 3)})
            print(f"[pickup-check] {now:.2f}s {stage}", flush=True)

        for step in range(math.ceil(args.seconds / dt)):
            now = step * dt
            refresh()
            rot = measured.xmat[base].reshape(3, 3)
            obj_truth = soap.data.root_link_pos_w[0].cpu().numpy().copy()
            bin_truth = tray.data.root_link_pos_w[0].cpu().numpy().copy()
            obj = source.sample(now, obj_truth, measured.xpos[base], rot)
            bin_pos = bin_source.sample(now, bin_truth, measured.xpos[base], rot)
            grip_point = measured.xpos[palm] + measured.xmat[palm].reshape(3, 3) @ local
            peak_z = max(peak_z, float(obj_truth[2]))
            goal = None
            closing = stage in ("close", "lift", "carry", "lower") and not close_open
            if stage == "settle" and now >= 1.:
                transition("reach", now)
            missing = ((stage in ("reach", "close", "lift") and obj is None) or
                       (stage in ("carry", "lower") and bin_pos is None))
            if missing:
                # Stop advancing all targets, retaining grip if already carrying.
                stale_steps += 1
                if now - entered > 3.:
                    reason = "target estimate unavailable or expired"
                    transition("failed", now)
            elif stage == "reach":
                goal = obj
                if np.linalg.norm(goal - grip_point) < .008:
                    # Evaluator diagnostic only: not supplied to the controller.
                    tilt_at_close = math.degrees(math.acos(float(np.clip(
                        measured.xmat[object_body].reshape(3, 3)[2, 2], -1, 1
                    ))))
                    observed_floor_z = float(obj[2])
                    held_goal = goal.copy()
                    transition("close", now)
            elif stage == "close":
                goal = held_goal
                if now - entered >= 1.:
                    held_goal = grip_point.copy() + (0, 0, .08)
                    transition("lift", now)
            elif stage == "lift":
                goal = held_goal
                if now - entered >= 2.:
                    if obj[2] - observed_floor_z < .035:
                        reason = "object did not follow the lift"
                        transition("failed", now)
                    else:
                        picked = True
                        transition("carry", now)
            elif stage == "carry":
                # Centre of the small bench tray, above its rim.
                goal = bin_pos + (0, 0, .13)
                if np.linalg.norm(goal - grip_point) < .012:
                    held_goal = goal.copy()
                    held_goal[2] = bin_pos[2] + .095
                    transition("lower", now)
            elif stage == "lower":
                goal = held_goal
                if np.linalg.norm(goal - grip_point) < .012:
                    transition("release", now)
            elif stage == "release":
                goal = held_goal
                if now - entered >= 1.:
                    held_goal = grip_point.copy() + (0, 0, .06)
                    transition("verify", now)
            elif stage == "verify":
                goal = held_goal
                # Evaluator-only truth: the complete object footprint is inside,
                # resting near the tray floor, released and still for half a second.
                object_rotation = measured.xmat[object_body].reshape(3, 3)
                bin_rotation = measured.xmat[tray_body].reshape(3, 3)
                corners_in_bin = ((object_corners @ object_rotation.T + obj_truth - bin_truth)
                                  @ bin_rotation)
                inside = (bool((np.abs(corners_in_bin[:, :2]) < TRAY_INNER_HALF).all()) and
                          -.004 < corners_in_bin[:, 2].min() < BIN_WALL_T + .008)
                still = float(soap.data.root_com_lin_vel_w[0].norm()) < .02
                separated = np.linalg.norm(obj_truth - grip_point) > .045
                settled_in_bin = settled_in_bin + 1 if inside and still and separated else 0
                if settled_in_bin * dt >= .5:
                    reason = "released object settled inside bin"
                    transition("done", now)
                elif now - entered > 4.:
                    reason = "placement verification failed"
                    transition("failed", now)

            if goal is not None and stage not in ("failed", "done"):
                result = ik.solve(measured.qpos, goal)
                last_error = result.error_m
                max_error = max(max_error, result.error_m)
                if result.error_m > .015:
                    reason = f"unreachable target (IK residual {result.error_m:.4f} m)"
                    transition("failed", now)
                else:
                    change = np.clip(result.joints - target_arm[:3].numpy(), -.45 * dt, .45 * dt)
                    max_joint_step = max(max_joint_step, float(np.abs(change).max()))
                    target_arm[:3] += torch.tensor(change, dtype=target_arm.dtype)
            if stage not in ("done", "failed", "settle") and now - entered > 8.:
                reason = f"{stage} timed out"
                transition("failed", now)
            robot.set_joint_position_target(target_arm[None, :], joint_ids=arm)
            angle = robot.data.joint_pos[:, finger]
            speed = robot.data.joint_vel[:, finger]
            target = torch.full_like(angle, GRIPPER_CLOSED if closing else GRIPPER_OPEN)
            robot.set_joint_position_target(squeeze_limited(target, angle, speed=speed),
                                            joint_ids=finger)
            _, _, terminated, truncated, _ = env.step(zero)
            if video is not None:
                video.update()
            if bool((terminated | truncated).any()):
                reason = "environment terminated; attempt not counted as success"
                transition("failed", now)
            if stage in ("done", "failed"):
                break

        if stage not in ("done", "failed"):
            transition("failed", now + dt)

        return {
            "success": stage == "done", "picked_up": picked, "reason": reason,
            "source": "simulated position estimates; NOT camera perception",
            "controller": "scripted bounded arm IK; zero-action standing, NOT trained cleanup",
            "negative_control": close_open, "sensor_dropout": args.sensor_dropout,
            "seed": args.seed, "noise_mm": args.noise_mm,
            "transitions": transitions, "max_object_rise_m": peak_z - initial_z,
            "max_ik_error_m": max_error, "last_ik_error_m": last_error,
            "max_arm_target_step_rad": max_joint_step, "stale_steps": stale_steps,
            "object_tilt_at_close_deg": tilt_at_close,
            "final_object_position_m": soap.data.root_link_pos_w[0].tolist(),
            "final_grip_point_m": grip_point.tolist(),
            "final_arm_joints_rad": robot.data.joint_pos[0, arm].tolist(),
            "final_arm_targets_rad": target_arm.tolist(),
            "final_base_position_m": robot.data.root_link_pos_w[0].tolist(),
            "seconds": round(now + dt, 3),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulated-estimates", action="store_true", required=True,
                        help="acknowledge that this is a privileged perception fixture")
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-mm", type=float, default=2.)
    parser.add_argument("--negative-control", action="store_true",
                        help="keep the finger open: pickup must not succeed")
    parser.add_argument("--sensor-dropout", action="store_true",
                        help="supply no detections: movement must stop")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        parser.error("seconds must be finite and in [1, 60]")
    if not math.isfinite(args.noise_mm) or not 0 <= args.noise_mm <= 20:
        parser.error("noise-mm must be finite and in [0, 20]")
    if not 0 <= args.seed <= 2**32 - 2:
        parser.error("seed must be in [0, 2**32 - 2]")
    if args.video and args.video.suffix.lower() != ".mp4":
        parser.error("video must name an MP4")
    report = run(args)
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
