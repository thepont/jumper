"""Planning must not confuse a disabled collision pair with positive clearance."""
import mujoco
import numpy as np
import pytest


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="a"><freejoint/><geom name="a" type="sphere" size=".01"
        contype="0" conaffinity="0"/></body>
      <body name="b" pos=".05 0 0"><freejoint/><geom name="b" type="sphere"
        size=".01" contype="0" conaffinity="0"/></body>
      </worldbody></mujoco>''')


@pytest.mark.parametrize('separation, expected', [(.05, True), (.0205, False),
                                                (.02, False), (.015, False)])
def test_mask_excluded_pair_is_checked_without_live_state_mutation(model, separation, expected):
    from mjrl.clearance import GeometryClearance
    live = mujoco.MjData(model)
    qpos = live.qpos.copy()
    qpos[7] = separation
    before = qpos.copy()
    check = GeometryClearance(model, [('a', 'b')], margin_m=.001)
    result = check.check(qpos)
    assert result.passed is expected
    np.testing.assert_array_equal(qpos, before)
    np.testing.assert_array_equal(live.qpos, model.qpos0)
    assert model.geom_contype.sum() == 0
    assert result.distances[0][0:2] == ('a', 'b')


def test_invalid_pairs_margin_and_pose_fail_at_boundary(model):
    from mjrl.clearance import GeometryClearance
    for pairs in ([], [('a', 'missing')], [('a', 'a')], [('a', 'b'), ('b', 'a')]):
        with pytest.raises(ValueError):
            GeometryClearance(model, pairs, margin_m=.001)
    for margin in (True, 0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            GeometryClearance(model, [('a', 'b')], margin_m=margin)
    check = GeometryClearance(model, [('a', 'b')], margin_m=.001)
    for pose in ([0.], np.full(model.nq, np.nan)):
        with pytest.raises(ValueError):
            check.check(pose)


def test_nonfinite_distance_is_rejected_not_reported_clear(model, monkeypatch):
    from mjrl.clearance import GeometryClearance
    check = GeometryClearance(model, [('a', 'b')], margin_m=.001)
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: float('nan'))
    with pytest.raises(RuntimeError):
        check.check(model.qpos0)


@pytest.mark.parametrize('separation,expected', [(.05,True),(.0205,False),(.02,False),(.015,False)])
def test_zero_mesh_distance_needs_an_independent_separation_certificate(monkeypatch,separation,expected):
    from mjrl.clearance import GeometryClearance
    model=mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="cube" vertex="
      -.01 -.01 -.01  .01 -.01 -.01  -.01 .01 -.01  .01 .01 -.01
      -.01 -.01 .01   .01 -.01 .01   -.01 .01 .01   .01 .01 .01"/></asset><worldbody>
      <body name="a"><freejoint/><geom name="a" type="mesh" mesh="cube" contype="0" conaffinity="0"/></body>
      <body name="b" pos=".05 0 0"><freejoint/><geom name="b" type="mesh" mesh="cube" contype="0" conaffinity="0"/></body>
      </worldbody></mujoco>''')
    pose=model.qpos0.copy()
    pose[7]=separation
    monkeypatch.setattr(mujoco,'mj_geomDistance',lambda *args:0.)
    result=GeometryClearance(model,[('a','b')],margin_m=.001).check(pose)
    assert result.passed is expected
    if expected:
        assert result.certified_pairs == (('a','b'),)
        assert result.distances[0][2] >= .001


def test_certificate_uses_geom_frames_after_a_shared_world_rotation(monkeypatch):
    from mjrl.clearance import GeometryClearance
    model=mujoco.MjModel.from_xml_string('''<mujoco><asset><mesh name="cube" vertex="
      -.01 -.01 -.01  .01 -.01 -.01  -.01 .01 -.01  .01 .01 -.01
      -.01 -.01 .01   .01 -.01 .01   -.01 .01 .01   .01 .01 .01"/></asset><worldbody>
      <body name="a"><freejoint/><geom name="a" type="mesh" mesh="cube"/></body>
      <body name="b" pos=".05 0 0"><freejoint/><geom name="b" type="mesh" mesh="cube"/></body>
      </worldbody></mujoco>''')
    pose=model.qpos0.copy()
    pose[:3]=[2.,3.,4.]
    pose[7:10]=[2.,3.05,4.]
    pose[3:7]=pose[10:14]=[np.sqrt(.5),0,0,np.sqrt(.5)]
    monkeypatch.setattr(mujoco,'mj_geomDistance',lambda *args:0.)
    result=GeometryClearance(model,[('a','b')],margin_m=.001).check(pose)
    assert result.passed
    np.testing.assert_allclose(result.distances[0][2],.03,atol=1e-8)


@pytest.mark.parametrize('offset, margin, expected', [(1., .001, True),
                                                     (1., .01, False), (0., .001, False)])
@pytest.mark.parametrize('transformed', [False, True])
def test_edge_separation_is_certified_without_admitting_overlapping_hulls(
        monkeypatch, offset, margin, expected, transformed):
    """Skew rods can be separated on edge-cross axes but not face normals."""
    from itertools import product
    from scipy.spatial.transform import Rotation
    from mjrl.clearance import GeometryClearance
    vertices = np.array(list(product([-1, 1], repeat=3))) * [.03, .006, .004]
    text = ' '.join(str(x) for x in vertices.ravel())
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <mesh name="rod" vertex="{text}"/></asset><worldbody>
      <body><freejoint/><geom name="a" type="mesh" mesh="rod"/></body>
      <body><freejoint/><geom name="b" type="mesh" mesh="rod"/></body>
      </worldbody></mujoco>''')
    rng = np.random.default_rng(53)
    pose = model.qpos0.copy()
    pose[3:7] = Rotation.random(random_state=rng).as_quat(scalar_first=True)
    pose[10:14] = Rotation.random(random_state=rng).as_quat(scalar_first=True)
    pose[7:10] = offset * np.array([.029897714081332676, .016127470174822384,
                                   -.012490355810275966])
    if transformed:
        rotation = Rotation.from_rotvec([.3, -.8, .2])
        for start in (0, 7):
            pose[start:start+3] = rotation.apply(pose[start:start+3]) + [2., -3., 1.]
            pose[start+3:start+7] = (rotation * Rotation.from_quat(
                pose[start+3:start+7], scalar_first=True)).as_quat(scalar_first=True)
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: 0.)
    result = GeometryClearance(model, [('a', 'b')], margin_m=margin).check(pose)
    assert result.passed is expected
    if expected:
        assert result.certified_pairs == (('a', 'b'),)
        assert result.distances[0][2] > .008
    else:
        assert result.certified_pairs == ()
