"""Behaviour at the non-actuating Rust/Python boundary, built from this checkout."""

import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def extension():
    if shutil.which("cargo") is None:
        pytest.skip("Rust toolchain unavailable")
    root = Path(__file__).resolve().parents[1]
    manifest = root / "deploy/fsm/Cargo.toml"
    result = subprocess.run(
        ["cargo", "build", "--manifest-path", str(manifest), "--no-default-features",
         "--features", "py", "--lib", "--locked", "--message-format=json"],
        check=True, capture_output=True, text=True, timeout=300,
    )
    artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    libraries = [Path(path) for item in artifacts if item.get("reason") == "compiler-artifact"
                 and item["target"]["name"] == "mjrl_fsm" for path in item["filenames"]
                 if Path(path).suffix in {".so", ".dylib", ".dll", ".pyd"}]
    assert len(libraries) == 1
    loader = importlib.machinery.ExtensionFileLoader("mjrl_fsm", str(libraries[0]))
    spec = importlib.util.spec_from_file_location("mjrl_fsm", libraries[0], loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor():
    return {
        "joints": [{"name": "hinge", "origin": {
            "translation": [0, 0, 0], "rotation_wxyz": [1, 0, 0, 0]},
            "anchor": [0, 0, 0], "axis": [0, 0, 1], "reference_rad": 0,
            "limits_rad": [-2, 2]}],
        "tool_point": [1, 0, 0], "tool_axis": [1, 0, 0],
    }


def test_given_named_joint_when_queried_then_return_world_point_and_direction(extension):
    arm = extension.SerialArm(json.dumps(descriptor()))
    point, axis = arm.forward([2, 3, 4], [1, 0, 0, 0], {"hinge": 0})
    assert point == pytest.approx([3, 3, 4])
    assert axis == pytest.approx([1, 0, 0])


@pytest.mark.parametrize("joints", [{}, {"unknown": 0}, {"hinge": float("nan")},
                                    {"hinge": 3}, {"hinge": 0, "extra": 0}])
def test_rejection_crosses_boundary_without_poisoning_next_query(extension, joints):
    arm = extension.SerialArm(json.dumps(descriptor()))
    with pytest.raises(ValueError):
        arm.forward([0, 0, 0], [1, 0, 0, 0], joints)
    assert arm.forward([0, 0, 0], [1, 0, 0, 0], {"hinge": 0})[0] == [1, 0, 0]


def test_malformed_descriptor_cannot_construct_adapter(extension):
    with pytest.raises(ValueError):
        extension.SerialArm("{}")


def test_wrong_pose_array_lengths_are_rejected(extension):
    arm = extension.SerialArm(json.dumps(descriptor()))
    with pytest.raises(ValueError):
        arm.forward([0, 0], [1, 0, 0, 0], {"hinge": 0})


def test_motion_checks_candidate_before_return_and_latches_rejection(extension):
    motion = extension.JointMotion([[1., .5]], [-2., -2.], [2., 2.], .45, 2.)
    seen = []

    def clear(candidate):
        seen.append(candidate)
        return True

    result = motion.step([0., 0.], [0., 0.], .02, clear)
    assert result == pytest.approx([.009, .0045])
    assert seen == [result]
    assert not motion.arrived
    with pytest.raises(ValueError):
        motion.step(result, result, .02, lambda _: False)
    with pytest.raises(ValueError):
        motion.step(result, result, .02, clear)
    assert len(seen) == 1


@pytest.mark.parametrize("verdict", [None, 1, "yes"])
def test_motion_guard_requires_a_real_boolean(extension, verdict):
    motion = extension.JointMotion([[1.]], [-2.], [2.], .45, 2.)
    with pytest.raises(ValueError):
        motion.step([0.], [0.], .02, lambda _: verdict)
    with pytest.raises(ValueError):
        motion.step([0.], [0.], .02, lambda _: True)


@pytest.mark.parametrize("positions", [[.2, -.1], [.7, -.6], [-1.1, .8]])
def test_compiled_mujoco_chain_matches_rust_with_anchors_and_references(extension, positions):
    """Independent FK oracle; asymmetry catches reversed order and dropped origins."""
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    model = mujoco.MjModel.from_xml_string('''
      <mujoco><compiler angle="radian"/><worldbody>
        <body name="base" pos=".2 -.3 .4" quat=".8 .2 -.3 .4">
          <body name="first" pos=".07 .03 -.02" quat=".9 .1 .2 -.3">
            <joint name="z_first" axis="0 0 1" pos=".02 -.01 .03"
                   range="-2 2" ref=".2"/>
            <geom size=".01" mass=".1"/>
            <body name="second" pos=".03 -.08 .01" quat=".7 -.4 .1 .2">
              <joint name="a_second" axis="0 1 0" pos="-.01 .02 .04"
                     range="-2 2" ref="-.1"/>
              <geom size=".01" mass=".1"/>
            </body>
          </body>
        </body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    names = ["z_first", "a_second"]
    joints = []
    for name in names:
        j = model.joint(name).id
        b = model.jnt_bodyid[j]
        joints.append(dict(name=name, origin=dict(
            translation=model.body_pos[b].tolist(), rotation_wxyz=model.body_quat[b].tolist()),
            anchor=model.jnt_pos[j].tolist(), axis=model.jnt_axis[j].tolist(),
            reference_rad=float(model.qpos0[model.jnt_qposadr[j]]),
            limits_rad=model.jnt_range[j].tolist()))
    point, axis = np.array([.03, -.02, .07]), np.array([0., 0., 1.])
    arm = extension.SerialArm(json.dumps(dict(joints=joints, tool_point=point.tolist(),
                                            tool_axis=axis.tolist())))
    for name, q in zip(names, positions, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = q
    mujoco.mj_forward(model, data)
    base, end = model.body("base").id, model.body("second").id
    actual_point, actual_axis = arm.forward(data.xpos[base].tolist(), data.xquat[base].tolist(),
                                           dict(zip(names, positions, strict=True)))
    rotation = data.xmat[end].reshape(3, 3)
    assert actual_point == pytest.approx(data.xpos[end] + rotation @ point, abs=1e-12, rel=0)
    assert actual_axis == pytest.approx(rotation @ axis, abs=1e-12, rel=0)
