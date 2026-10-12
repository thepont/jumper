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
from collections import deque
from contextlib import ExitStack
from functools import partial
import itertools
import json
import math
from pathlib import Path

import numpy as np

TRAY_INNER_HALF = .125
# The existing deployed arm preset opens 90 degrees from shut, inside the
# measured finger range [-1.6, .1]. This bench does not use a walking policy.
BENCH_OPEN = .10 - math.pi / 2
# Collision-hull measurement, native MuJoCo 3.11.0, 2026-10-13: the sloped
# face of LF_palm_grip_insert_link, in palm coordinates. At the reference pose
# the upright soap clears every active robot mesh; fixed/moving insert gaps
# are at least 1.79/6.08 mm. The legacy anvil omits this separate grip insert.
FIXED_CONTACT_PALM = np.array([-.002474635, .099238620, -.003243786])
GRASP_OFFSET = np.array([-.029345020, .019968078, .034235365])
GRASP_ARM = np.array([-1.04377350, -1.53823883, .43236852, -.61391009])


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


def fixture_positions(side="left", *, object_offset_mm=(0, 0), bin_offset_mm=(0, 0)):
    """Validate bench-frame XY millimetres, then reflect the whole right fixture.

    The upright, unrotated soap must have a separating XY axis from the
    tray's outer footprint. The evaluator checks the actual hull again at reset.
    """
    from tasks.jumper.five_foot.objects import BIN_WALL_T, SOLIDS

    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    positions = []
    for name, offset, origin in (("object-offset-mm", object_offset_mm, (.25, .08, .051)),
                                 ("bin-offset-mm", bin_offset_mm, (.13, .235, 0.))):
        if not isinstance(offset, (tuple, list)) or len(offset) != 2:
            raise ValueError(f"{name} must contain exactly two XY values")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not -50 <= v <= 50 or not math.isfinite(v) for v in offset):
            raise ValueError(f"{name} must be finite XY millimetres in [-50, 50]")
        positions.append(np.array(origin) + np.array([*offset, 0.]) / 1000)
    soap = next(s for s in SOLIDS if s.name == "soap")
    lo, hi = soap.bounds
    relative = positions[0] - positions[1]
    outer = TRAY_INNER_HALF + BIN_WALL_T
    if not any(relative[axis] + lo[axis] > outer or
               relative[axis] + hi[axis] < -outer for axis in (0, 1)):
        raise ValueError("pickup fixture must start with the whole object outside the tray")
    mirror = np.array([1., -1. if side == "right" else 1., 1.])
    return tuple(tuple(p * mirror) for p in positions)


def config(seed=7, side="left", *, object_offset_mm=(0, 0), bin_offset_mm=(0, 0)):
    object_position, bin_position = fixture_positions(
        side, object_offset_mm=object_offset_mm, bin_offset_mm=bin_offset_mm)
    from tasks.jumper.five_foot.claw import ARM_JOINTS, LF_GRASP
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
    cfg.events["hold_carried_arm"].params["ranges"] = (
        tuple((LF_GRASP[j], LF_GRASP[j]) for j in ARM_JOINTS)
        + ((BENCH_OPEN, BENCH_OPEN),)
    )
    if side == "right":
        from .pickup_side import right_bench
        right_bench(cfg)
    soap = next(s for s in SOLIDS if s.name == "soap")
    props = {
        "soap": _prop(partial(_solid_spec, soap), object_position),
        "dropbin": _prop(bench_tray, bin_position),
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

    def __init__(self, *, noise_m, seed, dropout, visibility_radius_m=0.):
        self.noise_m = noise_m
        self.rng = np.random.default_rng(seed)
        self.dropout = dropout
        self.pending = []
        self.latest = None
        self.next_exposure = 0.
        self.visibility_radius_m = visibility_radius_m

    def sample(self, now, world_point, base_pos, base_rot):
        from mjrl.manipulation import TargetEstimate

        if now + 1e-9 >= self.next_exposure:
            self.next_exposure = now + .1
            relative = base_rot.T @ (world_point - base_pos)
            distance = float(np.linalg.norm(relative[:2]))
            angular_radius = math.asin(min(1., self.visibility_radius_m / max(distance, .001)))
            visible = (relative[0] > .05 and
                       abs(math.atan2(relative[1], relative[0])) < math.radians(55) + angular_radius)
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
        ARM_JOINTS, FINGER_JOINT, GRIPPER_CLOSED,
    )
    from tasks.jumper.five_foot.mdp.grasp import Claw
    from tasks.jumper.five_foot.mdp.gripper import squeeze_limited
    from tasks.jumper.five_foot.objects import BIN_WALL_T

    side = getattr(args, "side", "left")
    object_offset_mm = getattr(args, "object_offset_mm", (0, 0))
    bin_offset_mm = getattr(args, "bin_offset_mm", (0, 0))
    prefix = "RF" if side == "right" else "LF"
    mirror = np.array([1., -1. if side == "right" else 1., 1.])
    # Actual LF/RF palm, finger and insert vertices match exactly under this
    # local z half-turn (CPU mesh probe, 2026-10-13). This is NOT the world
    # reflection above; the right palm's frame is also rotated.
    local_mirror = np.array([-1., -1., 1.]) if side == "right" else np.ones(3)
    joint_signs = np.array([-1., 1., -1., -1.]) if side == "right" else np.ones(4)
    finger_sign = -1. if side == "right" else 1.
    ARM_JOINTS = tuple(n.replace("LF", prefix) for n in ARM_JOINTS)
    FINGER_JOINT = FINGER_JOINT.replace("LF", prefix)

    res = resolve(backend="native", device="cpu", num_envs=1)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv

    with ExitStack() as stack, torch.inference_mode():
        env = ManagerBasedRlEnv(cfg=config(
            args.seed, side, object_offset_mm=object_offset_mm,
            bin_offset_mm=bin_offset_mm), device="cpu")
        stack.callback(env.close)
        env.reset(seed=args.seed)
        model = env.sim.mj_model
        robot, soap, tray = (env.scene[n] for n in ("robot", "soap", "dropbin"))
        arm = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
        finger = robot.find_joints([FINGER_JOINT])[0]
        claw = Claw(env)
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{prefix}_palm_link")
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
        object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soap/soap")
        tray_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dropbin/dropbin")
        if min(palm, base, object_body, tray_body) < 0:
            raise ValueError("required robot bodies are absent")
        from tasks.jumper.five_foot.objects import SOLIDS
        solid = next(s for s in SOLIDS if s.name == "soap")
        # Use the physical hull, not imaginary AABB corners. A rounded bar
        # rotated onto its side has box corners below the floor and outside
        # the tray even when its actual collision surface is fully contained.
        object_points = (np.array(solid.hull) if solid.hull else
                         np.array(list(itertools.product(*zip(*solid.bounds, strict=True)))))
        measured = mujoco.MjData(model)

        def refresh():
            measured.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mujoco.mj_kinematics(model, measured)

        refresh()
        initial_object_position = soap.data.root_link_pos_w[0].tolist()
        initial_bin_position = tray.data.root_link_pos_w[0].tolist()
        initial_points = (object_points @ measured.xmat[object_body].reshape(3, 3).T
                          + soap.data.root_link_pos_w[0].numpy())
        initial_in_tray_frame = ((initial_points - tray.data.root_link_pos_w[0].numpy())
                                 @ measured.xmat[tray_body].reshape(3, 3))
        outer = TRAY_INNER_HALF + BIN_WALL_T
        initially_outside_bin = any(
            initial_in_tray_frame[:, axis].min() > outer or
            initial_in_tray_frame[:, axis].max() < -outer for axis in (0, 1)
        )
        if not initially_outside_bin:
            raise ValueError("pickup fixture must start with the whole object outside the tray")
        # Choose the measured, collision-clear grasp branch on a private
        # kinematic state. This never changes robot or object simulator state.
        reference = mujoco.MjData(model)
        reference.qpos[:] = measured.qpos
        for joint, value in zip(ARM_JOINTS, GRASP_ARM * joint_signs, strict=True):
            reference.qpos[model.jnt_qposadr[model.joint(f"robot/{joint}").id]] = value
        local = FIXED_CONTACT_PALM * local_mirror
        ik = PointIK(model, f"robot/{prefix}_palm_link", tuple(f"robot/{n}" for n in ARM_JOINTS),
                     local, local_axis=claw.closing_palm.numpy() * local_mirror, axis_scale_m=.06)
        zero = torch.zeros(env.action_space.shape)
        target_arm = robot.data.joint_pos[0, arm].clone()
        raised_arm = target_arm.numpy().copy()
        raised_arm[1] = -2.4
        video = None
        if args.video:
            from mjrl.viewer.video import ReplayVideo
            video = stack.enter_context(ReplayVideo(
                env, args.video, fps=25, width=800, height=600,
                camera_distance=1.15, azimuth=135, elevation=-35,
                follow=False, lookat_height=.07,
            ))
        source = SimulatedEstimates(noise_m=args.noise_mm / 1000, seed=args.seed,
                                    dropout=args.sensor_dropout,
                                    visibility_radius_m=float(np.linalg.norm(solid.size[:2])))
        bin_source = SimulatedEstimates(noise_m=args.noise_mm / 1000, seed=args.seed + 1,
                                        dropout=args.sensor_dropout,
                                        visibility_radius_m=TRAY_INNER_HALF)
        stage, entered = "settle", 0.
        transitions = [{"stage": stage, "at_s": 0.}]
        initial_z = float(soap.data.root_link_pos_w[0, 2])
        peak_z = initial_z
        picked = False
        clear_grasp_steps = 0
        peak_floor_clearance = 0.
        reason = "deadline"
        held_goal = None
        observed_floor_z = None
        close_open = args.negative_control
        dt = env.step_dt
        max_error = 0.
        max_joint_step = 0.
        max_axis_error = 0.
        last_ik_joints = reference.qpos[ik.qadr].copy()
        cartesian_reference = None
        last_error = None
        stale_steps = 0
        settle_window = deque(maxlen=math.ceil(.5 / dt) + 1)
        tilt_at_close = None
        initial_robot_clear = True
        final_inside = False
        final_released = False
        settling_drift = None

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
            world_points = (object_points @ measured.xmat[object_body].reshape(3, 3).T
                             + obj_truth)
            floor_clearance = float(world_points[:, 2].min() - env.scene.env_origins[0, 2])
            peak_floor_clearance = max(peak_floor_clearance, floor_clearance)
            # Evaluator witness: lifted means the whole object clears the floor
            # while both actual grip pads still touch it for at least 0.2 s.
            touching = set()
            if stage in ("settle", "lift", "carry", "lower", "verify"):
                mujoco.mj_collision(model, measured)
                for contact in measured.contact:
                    if contact.dist > .001:
                        continue
                    a, b = (int(g) for g in contact.geom)
                    if model.geom_bodyid[a] == object_body:
                        touching.add(model.geom(b).name)
                    elif model.geom_bodyid[b] == object_body:
                        touching.add(model.geom(a).name)
                pinched = {f"robot/{prefix}_palm_grip_insert_link_meshcol",
                           f"robot/{prefix}_finger_grip_insert_link_meshcol"} <= touching
                if stage == "settle" and any(name.startswith("robot/") for name in touching):
                    initial_robot_clear = False
                clear_grasp_steps = clear_grasp_steps + 1 if pinched and floor_clearance > .015 else 0
                picked = picked or clear_grasp_steps * dt >= .2
            goal = None
            desired_arm = None
            closing = stage in ("close", "lift", "carry", "lower") and not close_open
            if stage == "settle" and now >= 1.:
                if not initial_robot_clear:
                    reason = "object touched the robot before the approach"
                    transition("failed", now)
                else:
                    transition("raise", now)
            missing = ((stage in ("raise", "pregrasp", "reach", "close", "lift", "carry", "lower")
                        and obj is None) or
                       (stage in ("carry", "lower") and bin_pos is None))
            if missing:
                # Stop advancing all targets, retaining grip if already carrying.
                stale_steps += 1
                if now - entered > 3.:
                    reason = "target estimate unavailable or expired"
                    transition("failed", now)
            elif stage == "raise":
                # Raise the stowed arm before swinging the long fixed jaw over
                # the object. A joint interpolation straight to pregrasp can
                # sweep that jaw sideways through the object at floor height.
                desired_arm = raised_arm
                if np.abs(robot.data.joint_pos[0, arm].numpy() - raised_arm).max() < .04:
                    transition("pregrasp", now)
            elif stage == "pregrasp":
                goal = obj + GRASP_OFFSET * mirror + (-.05, 0, .035)
                if np.linalg.norm(goal - grip_point) < .010:
                    transition("reach", now)
            elif stage == "reach":
                # Grip the upper part of the upright soap; its centre is lower
                # than the opening. The vertical approach avoids sweeping a
                # finger into it while changing the arm's orientation.
                goal = obj + GRASP_OFFSET * mirror
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
                    if finger_sign * float(robot.data.joint_pos[0, finger[0]]) > GRIPPER_CLOSED - .025:
                        reason = "finger closed empty"
                        transition("failed", now)
                    else:
                        held_goal = grip_point.copy() + (-.035, 0, .055)
                        transition("lift", now)
            elif stage == "lift":
                goal = held_goal
                if now - entered >= 2.:
                    if obj[2] - observed_floor_z < .035:
                        reason = "object did not follow the lift"
                        transition("failed", now)
                    else:
                        transition("carry", now)
            elif stage == "carry":
                # Centre of the small bench tray, above its rim.
                # Even the soap's 62 mm bounding radius clears the 24 mm rim
                # here, while the short arm can still reach across the tray.
                object_goal = bin_pos + np.array([.02, -.005, .095]) * mirror
                goal = grip_point + object_goal - obj
                if np.linalg.norm(object_goal - obj) < .012:
                    held_goal = goal.copy()
                    held_goal[2] -= .025
                    transition("lower", now)
            elif stage == "lower":
                object_goal = bin_pos + np.array([.02, -.005, .070]) * mirror
                goal = grip_point + object_goal - obj
                if (np.linalg.norm((object_goal - obj)[:2]) < .012 and
                        obj[2] - bin_pos[2] < .075 and now - entered >= .5):
                    # Open and withdraw vertically together. Holding the sloped
                    # fixed pad beside the released object, or dragging it back,
                    # pushes the soap toward the near wall as it tips down.
                    held_goal = grip_point.copy() + (0, 0, .035)
                    transition("release", now)
            elif stage == "release":
                goal = held_goal
                if now - entered >= 1.:
                    transition("verify", now)
            elif stage == "verify":
                goal = held_goal
                # Evaluator-only truth: the complete physical footprint is inside,
                # resting near the tray floor, released and still for half a second.
                bin_rotation = measured.xmat[tray_body].reshape(3, 3)
                points_in_bin = (world_points - bin_truth) @ bin_rotation
                inside = bool((np.abs(points_in_bin[:, :2]) < TRAY_INNER_HALF).all() and
                              -.004 < points_in_bin[:, 2].min() < BIN_WALL_T + .008)
                released = (finger_sign * float(robot.data.joint_pos[0, finger[0]]) < BENCH_OPEN + .15 and
                            not any(name.startswith("robot/") for name in touching))
                final_inside, final_released = inside, released
                if inside and released and picked:
                    settle_window.append(world_points.copy())
                else:
                    settle_window.clear()
                # Bound translation and rotation together: no physical vertex
                # may move more than 3 mm throughout the half-second window.
                settling_drift = (max(float(np.linalg.norm(p - settle_window[0], axis=1).max())
                                      for p in settle_window) if settle_window else None)
                still = (len(settle_window) == settle_window.maxlen and settling_drift < .003)
                if still:
                    reason = "released object settled inside bin"
                    transition("done", now)
                elif now - entered > 4.:
                    reason = "placement verification failed"
                    transition("failed", now)

            if goal is not None and stage not in ("failed", "done"):
                if stage in ("carry", "lower"):
                    # Continue through nearby IK poses instead of asking the
                    # short arm to jump to a distant inverse-kinematics branch.
                    if cartesian_reference is None:
                        cartesian_reference = grip_point.copy()
                    delta = goal - cartesian_reference
                    length = float(np.linalg.norm(delta))
                    travel = (.03 if stage == "carry" else .02) * dt
                    if length > travel:
                        goal = cartesian_reference + delta * (travel / length)
                # Retain the commanded reference under load. Advancing only a
                # fraction ahead of the measured point tracks gravitational sag
                # instead of generating the position error that supports weight.
                cartesian_reference = goal.copy()
                seed = measured.qpos.copy()
                # Continue the same solution branch despite the servo lag and
                # noisy observations. This seed never writes simulator state.
                seed[ik.qadr] = last_ik_joints
                orientation = ({"axis_plane_normal": (0, 0, 1)}
                               if stage in ("carry", "lower", "release", "verify")
                               else {"axis_target": np.array([0, -1, 0]) * mirror})
                result = ik.solve(seed, goal, **orientation)
                last_ik_joints = result.joints.copy()
                last_error = result.error_m
                max_error = max(max_error, result.error_m)
                max_axis_error = max(max_axis_error, result.axis_error_rad)
                if result.error_m > .015:
                    reason = f"unreachable target (IK residual {result.error_m:.4f} m)"
                    transition("failed", now)
                else:
                    desired_arm = result.joints
            if desired_arm is not None and stage not in ("failed", "done"):
                change = np.clip(desired_arm - target_arm.numpy(), -.45 * dt, .45 * dt)
                max_joint_step = max(max_joint_step, float(np.abs(change).max()))
                target_arm += torch.tensor(change, dtype=target_arm.dtype)
            if stage not in ("done", "failed", "settle") and now - entered > 8.:
                reason = f"{stage} timed out"
                transition("failed", now)
            robot.set_joint_position_target(target_arm[None, :], joint_ids=arm)
            angle = robot.data.joint_pos[:, finger]
            speed = robot.data.joint_vel[:, finger]
            closing = (stage in ("close", "lift", "carry", "lower") or
                       (stage == "failed" and closing)) and not close_open
            target = torch.full_like(angle, GRIPPER_CLOSED if closing else BENCH_OPEN)
            robot.set_joint_position_target(finger_sign * squeeze_limited(
                target, finger_sign * angle, speed=finger_sign * speed),
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
            "side": side,
            "object_offset_mm": list(object_offset_mm),
            "bin_offset_mm": list(bin_offset_mm),
            "world_object_offset_mm": [object_offset_mm[0], object_offset_mm[1] * mirror[1]],
            "world_bin_offset_mm": [bin_offset_mm[0], bin_offset_mm[1] * mirror[1]],
            "initial_object_position_m": initial_object_position,
            "initial_bin_position_m": initial_bin_position,
            "arm_joints": ARM_JOINTS, "finger_joint": FINGER_JOINT,
            "required_grip_contacts": [f"robot/{prefix}_{part}_grip_insert_link_meshcol"
                                       for part in ("palm", "finger")],
            "success": stage == "done", "picked_up": picked, "reason": reason,
            "initially_outside_bin": initially_outside_bin,
            "initial_robot_clear": initial_robot_clear,
            "fully_inside_bin": final_inside, "released_clear_of_robot": final_released,
            "settling_max_vertex_drift_m": settling_drift,
            "source": "simulated position estimates; NOT camera perception",
            "controller": "scripted bounded arm IK; zero-action standing, NOT trained cleanup",
            "negative_control": close_open, "sensor_dropout": args.sensor_dropout,
            "seed": args.seed, "noise_mm": args.noise_mm,
            "transitions": transitions, "max_object_rise_m": peak_z - initial_z,
            "max_object_floor_clearance_m": peak_floor_clearance,
            "max_ik_error_m": max_error, "last_ik_error_m": last_error,
            "max_arm_target_step_rad": max_joint_step, "stale_steps": stale_steps,
            "max_closing_axis_error_deg": math.degrees(max_axis_error),
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
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--object-offset-mm", nargs=2, type=float, default=(0, 0),
                        metavar=("X", "Y"), help="bench-frame XY offset, each within +/-50 mm")
    parser.add_argument("--bin-offset-mm", nargs=2, type=float, default=(0, 0),
                        metavar=("X", "Y"), help="bench-frame XY offset, reflected in Y on right")
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
    try:
        fixture_positions(args.side, object_offset_mm=args.object_offset_mm,
                          bin_offset_mm=args.bin_offset_mm)
    except ValueError as error:
        parser.error(str(error))
    report = run(args)
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
