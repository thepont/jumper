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


def tracking(extension, planner):
    """Adapter contract: durations share the Rust monotonic clock, never wall time."""
    def duration(ns):
        return {"secs": ns // 1_000_000_000, "nanos": ns % 1_000_000_000}
    policy = dict(lower=[-1.], upper=[1.], rate=.45, timeout=duration(1_000_000_000),
                  freshness=duration(250_000_000), position_tolerance=.005,
                  axis_tolerance=.17453292519943295, settle=duration(0))
    return extension.Tracking(json.dumps(policy), 0, planner)


def test_rust_tracking_adapter_reacquires_and_follows_a_moving_target(extension):
    requests = []

    def planner(request):
        requests.append(request)
        return None if request["position"][0] > .1 else [[request["position"][0] / .1]]

    session = tracking(extension, planner)
    axis = [1., 0., 0.]
    def feedback(ms, q):
        return ([q], [q], [q * .1, 0., 0.], axis, ms * 1_000_000, .02)

    def packet(ms, x):
        return ([x, 0., 0.], axis, 0, ms * 1_000_000)
    assert session.step(feedback(20, 0.), packet(20, .2), lambda q: True) == ([0.], "holding")
    joints, status = session.step(feedback(40, 0.), packet(40, .03), lambda q: True)
    assert joints[0] == pytest.approx(.009)
    assert status == "following"
    joints, status = session.step(feedback(60, .009), packet(60, .0012), lambda q: True)
    assert joints[0] > .009
    assert status == "aligned"  # target moved; stationarity is not required
    assert requests[-1]["remaining_ns"] < requests[0]["remaining_ns"]


@pytest.mark.parametrize("failure", ["malformed", "guard", "planner"])
def test_rust_tracking_adapter_failures_latch_without_unchecked_resume(extension, failure):
    def planner(request):
        if failure == "planner":
            raise RuntimeError("injected planner failure")
        return [[.3]]

    session = tracking(extension, planner)
    feedback = ([0.], [0.], [0., 0., 0.], [1., 0., 0.], 20_000_000, .02)
    observation = ([.03, 0., 0.], [1., 0., 0.], 0, 20_000_000)
    with pytest.raises((ValueError, TypeError)):
        session.step(("bad",) if failure == "malformed" else feedback,
                     observation, lambda q: 1 if failure == "guard" else True)
    with pytest.raises(ValueError):
        session.step(feedback, observation, lambda q: True)


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


def scene_motion(extension):
    return extension.SceneMotion([[1., .5]], [-2., -2.], [2., 2.], .45, 2., 7, 250_000_000)


def test_fresh_scene_binding_preserves_motion_and_checked_candidate(extension):
    motion = scene_motion(extension)
    seen = []
    q = motion.step([0., 0.], [0., 0.], .02, (7, 100_000_000), 350_000_000,
                    lambda candidate: seen.append(candidate) or True)
    assert q == pytest.approx([.009, .0045])
    assert seen == [q]
    assert not motion.arrived


@pytest.mark.parametrize("scene,now", [
    ((8, 100_000_000), 100_000_000),  # target/obstacle revision changed
    ((7, 100_000_000), 350_000_001),  # coverage expired
    ((7, 100_000_001), 100_000_000),  # future packet
    (None, 100_000_000), ((7,), 100_000_000),
    ((True, 100_000_000), 100_000_000), ((7, -1), 100_000_000),
    ((7, 100_000_000), float("nan")), ((7, 100_000_000), True),
])
def test_rejected_scene_packet_cannot_resume_or_reach_geometry(extension, scene, now):
    motion = scene_motion(extension)
    seen = []
    for packet, clock in [(scene, now), ((7, 120_000_000), 120_000_000)]:
        with pytest.raises((ValueError, TypeError, OverflowError)):
            motion.step([0., 0.], [0., 0.], .02, packet, clock,
                        lambda candidate: seen.append(candidate) or True)
    assert seen == []
    assert not motion.arrived


@pytest.mark.parametrize("verdict", [None, 1, False, "exception"])
def test_scene_geometry_failure_cannot_resume(extension, verdict):
    motion = scene_motion(extension)

    def guard(_):
        if verdict == "exception":
            raise RuntimeError("geometry unavailable")
        return verdict

    with pytest.raises(ValueError):
        motion.step([0., 0.], [0., 0.], .02, (7, 100), 100, guard)
    with pytest.raises(ValueError):
        motion.step([0., 0.], [0., 0.], .02, (7, 120), 120, lambda _: True)


def test_malformed_joint_packet_latches_scene_binding_stop(extension):
    motion = scene_motion(extension)
    with pytest.raises((ValueError, TypeError)):
        motion.step(None, [0., 0.], .02, (7, 100), 100, lambda _: True)
    with pytest.raises(ValueError):
        motion.step([0., 0.], [0., 0.], .02, (7, 120), 120, lambda _: True)


def test_scene_hold_checks_current_target_without_advancing_and_consumes_budget(extension):
    motion = extension.SceneMotion([[.2]], [-1.], [1.], .45, .04, 7, 250_000_000)
    seen = []
    held = motion.hold([0.], [.01], .02, (7, 0), 20_000_000,
                       lambda candidate: seen.append(candidate) or True)
    assert held == [0.]
    assert seen == [[0.]]
    assert not motion.arrived
    assert motion.step([0.], [.01], .02, (7, 0), 40_000_000,
                       lambda _: True) == pytest.approx([.009])
    with pytest.raises(ValueError):
        motion.hold([.009], [.01], .02, (7, 0), 60_000_000, lambda _: True)
    with pytest.raises(ValueError):
        motion.step([.009], [.01], .02, (7, 0), 80_000_000, lambda _: True)


@pytest.mark.parametrize("packet,clock,guard", [
    ((8, 0), 20_000_000, True), ((7, 0), 250_000_001, True),
    ((7, 0), 20_000_000, False), (None, 20_000_000, True),
])
def test_scene_hold_failure_latches_stop(extension, packet, clock, guard):
    motion = scene_motion(extension)
    with pytest.raises((ValueError, TypeError)):
        motion.hold([0., 0.], [0., 0.], .02, packet, clock, lambda _: guard)
    with pytest.raises(ValueError):
        motion.step([0., 0.], [0., 0.], .02, (7, 120), 120, lambda _: True)


@pytest.mark.parametrize("event", ["target_moved", "obstacle_entered", "coverage_lost"])
def test_scene_event_prevents_next_simulated_actuator_dispatch(extension, event):
    """Injected perception events, real MuJoCo actuator; not a camera detector test."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string('''
      <mujoco><option timestep=".002" gravity="0 0 0" integrator="implicitfast"/>
        <worldbody>
          <body><joint name="hinge" type="hinge" axis="0 0 1"/>
            <geom name="arm" type="capsule" fromto="0 0 0 .2 0 0" size=".01"/>
          </body>
          <body name="obstacle" mocap="true" pos="1 1 1">
            <geom name="obstacle_geom" type="sphere" size=".03"/>
          </body>
        </worldbody>
        <actuator><position joint="hinge" kp="20" kv="2"/></actuator>
      </mujoco>''')
    data, query = mujoco.MjData(model), mujoco.MjData(model)
    motion = extension.SceneMotion([[.3]], [-1.], [1.], .45, 2., 7, 250_000_000)
    checks = []

    def clear(candidate):
        checks.append(candidate)
        query.qpos[:] = candidate
        query.mocap_pos[:] = data.mocap_pos
        mujoco.mj_forward(model, query)
        return mujoco.mj_geomDistance(model, query, model.geom("arm").id,
                                     model.geom("obstacle_geom").id, 2., None) > .001

    # Positive control: accepted proposals reach the actuator and produce motion.
    for tick in range(10):
        now = tick * 20_000_000
        data.ctrl[:] = motion.step(data.ctrl.tolist(), data.qpos.tolist(), .02,
                                   (7, now), now, clear)
        for _ in range(10):
            mujoco.mj_step(model, data)
    assert data.qpos[0] > 0
    previous = data.ctrl.copy()
    assert len(checks) == 10
    if event == "obstacle_entered":
        data.mocap_pos[0] = [.15, .02, 0]
        mujoco.mj_forward(model, data)
    scene = (7, 0) if event == "coverage_lost" else (8, 300_000_000)
    # Assignment occurs only after successful proposal; rejection leaves the
    # prior target held. This is not evidence of zero mechanical stopping distance.
    with pytest.raises(ValueError):
        data.ctrl[:] = motion.step(data.ctrl.tolist(), data.qpos.tolist(), .02,
                                   scene, 300_000_000, clear)
    assert data.ctrl == pytest.approx(previous)
    assert len(checks) == 10
    with pytest.raises(ValueError):
        motion.step(data.ctrl.tolist(), data.qpos.tolist(), .02,
                    (7, 320_000_000), 320_000_000, clear)


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
