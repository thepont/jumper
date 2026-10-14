"""Executing a checked joint segment must preserve its direction."""
import numpy as np
import pytest


def test_unequal_joint_moves_remain_on_the_checked_segment():
    from mjrl.manipulation import joint_segment_step
    current=np.array([0., 1., -2.])
    goal=np.array([1., .5, -1.9])
    result=joint_segment_step(current,goal,max_delta=.1)
    np.testing.assert_allclose(result, [.1,.95,-1.99])
    # Independent clipping leaves this segment and is the rejected control.
    assert not np.allclose(result,current+np.clip(goal-current,-.1,.1))
    np.testing.assert_array_equal(current,[0.,1.,-2.])


def test_arrival_and_stationary_commands_do_not_overshoot_or_alias_inputs():
    from mjrl.manipulation import joint_segment_step
    for goal in [np.zeros(2),np.array([.01,-.02])]:
        result=joint_segment_step(np.zeros(2),goal,max_delta=.1)
        np.testing.assert_array_equal(result,goal)
        assert not np.shares_memory(result,goal)


@pytest.mark.parametrize('current,goal,step', [([],[],.1),([0.],[1.,2.],.1),
    ([np.nan],[1.],.1),([0.],[np.inf],.1),([0.],[1.],0),([0.],[1.],True)])
def test_malformed_segment_is_rejected(current,goal,step):
    from mjrl.manipulation import joint_segment_step
    with pytest.raises(ValueError):
        joint_segment_step(current,goal,max_delta=step)
