"""Side-specific bench configuration; never changes the training configuration.

The deployed front-arm mirror keeps shoulder pitch and negates the other four
joints. Rear/middle legs are NOT mirrored here: their calibrated HOME stays put.
Right jaw hulls are cut from the actual right visual meshes, not copied from LF.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np


def right_bench(cfg):
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.sensor import ContactMatch, ObjRef
    from tasks.jumper.common.constants import HOME
    from tasks.jumper.five_foot.claw import (
        ARM_JOINTS, FINGER_JOINT, FIVE_FOOT_LEGS, LF_GRASP, foot_geoms_for,
    )
    from tasks.jumper.five_foot.jaws import (
        GRIP_FRICTION, GRIP_GEOMS, JAW_COLLISION, JAW_PIECES,
    )

    def right(name):
        return name.replace("LF_", "RF_")

    signs = (-1, 1, -1, -1, -1)
    carried = tuple(right(n) for n in ARM_JOINTS) + (right(FINGER_JOINT),)
    action = cfg.actions["joint_pos"]
    action.actuator_names = [n.replace("RF_", "LF_") for n in action.actuator_names]
    action.scale = {n.replace("RF_", "LF_"): v for n, v in action.scale.items()}
    robot = cfg.scene.entities["robot"]
    robot.init_state.joint_pos = dict(HOME) | {
        n: s * LF_GRASP[n.replace("RF_", "LF_")]
        for n, s in zip(carried, signs, strict=True)
    }
    cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=action.actuator_names)
    hold = cfg.events["hold_carried_arm"].params
    hold["asset_cfg"] = SceneEntityCfg("robot", joint_names=carried, preserve_order=True)
    hold["ranges"] = tuple(tuple(sorted(s * x for x in bounds))
                           for s, bounds in zip(signs, hold["ranges"], strict=True))
    legs = tuple("LF" if n == "RF" else n for n in FIVE_FOOT_LEGS)
    for sensor in cfg.scene.sensors:
        if sensor.name == "foot_height_scan":
            sensor.frame = tuple(ObjRef(type="site", name=n, entity="robot") for n in legs)
        elif sensor.name == "feet_ground_contact":
            sensor.primary = ContactMatch(mode="geom", pattern=foot_geoms_for(legs),
                                          entity="robot")
    base_spec = robot.spec_fn

    def spec_fn():
        import trimesh
        from tasks.jumper.five_foot.tools.jaw_decompose import decompose

        spec = base_spec()
        for left_link, count in JAW_PIECES.items():
            link = right(left_link)
            mesh = spec.mesh(link)
            source = Path(spec.modelfiledir) / spec.meshdir / mesh.file
            parts = decompose(trimesh.load(source, process=False), count, .15, 48)
            hull = spec.geom(f"{link}_meshcol")
            for i, part in enumerate(parts):
                name = f"{link}_bench_col{i}"
                spec.add_mesh(name=name, uservert=np.asarray(part.vertices).ravel().tolist(),
                              userface=np.asarray(part.faces).ravel().tolist())
                geom = hull if i == 0 else spec.body(link).add_geom(default=hull.classname)
                if i:
                    geom.name = f"{link}_meshcol{i}"
                    geom.type = hull.type
                    geom.rgba = hull.rgba
                geom.meshname = name
        return spec

    robot.spec_fn = spec_fn
    # Tip-pad labels swap under the actual front-claw reflection: LF's inner
    # pad_f matches RF pad_b exactly (CPU vertex probe, 2026-10-13). RF pad_b
    # otherwise inherits the walking-foot friction (1.2, .01, .01).
    right_grip = tuple("RF_palm_pad_b_link_meshcol"
                       if n == "LF_palm_pad_f_link_meshcol" else right(n)
                       for n in GRIP_GEOMS)
    grip = "|".join(f"^{n}$" for n in right_grip)

    def first(table, value):
        return {grip: value} | table

    robot.collisions = (replace(
        JAW_COLLISION,
        geom_names_expr=JAW_COLLISION.geom_names_expr + right_grip,
        condim=first(JAW_COLLISION.condim, 3),
        priority=first(JAW_COLLISION.priority, 1),
        friction=first(JAW_COLLISION.friction, GRIP_FRICTION),
    ),)
