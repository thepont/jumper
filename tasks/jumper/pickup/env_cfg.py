"""Separate, five-action pickup PPO task using the physically checked bench.

Reward weights and randomisation widths below are initial choices, not tuned
claims. Reuse bench physics and the existing servo squeeze limiter; there is no
scripted controller, force assistance, weld or pose-writing step event.
"""
from dataclasses import replace
from functools import partial
import math
from pathlib import Path

import numpy as np
from mjlab.envs import mdp as env_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ..common.assets import JUMPER_XML
from ..five_foot.claw import GRIPPER_CLOSED
from ..five_foot.mdp.events import hold_carried_arm
from ..five_foot.objects import BIN_WALL_T, SOLIDS, _prop, _solid_spec
from ..five_foot.tools.pickup_check import (
    BENCH_OPEN, FIXED_CONTACT_PALM, GRASP_OFFSET, TRAY_INNER_HALF, config, fixture_positions,
)
from .mdp.actions import ClawActionCfg
from .mdp import terms


def env_cfg(asset: Path | None = None, play=False, *, claw_side='left',
            position_jitter_mm=10., detection_dropout=.05, object_size='mixed'):
    if asset is not None and Path(asset).resolve() != JUMPER_XML.resolve():
        raise ValueError('pickup geometry is calibrated only for the registered jumper asset')
    if claw_side not in ('left', 'right') or object_size not in ('mixed', 'small', 'standard', 'large'):
        raise ValueError('invalid claw side or object size mode')
    for value, low, high, name in ((position_jitter_mm, 0, 30, 'position-jitter-mm'),
                                  (detection_dropout, 0, 1, 'detection-dropout')):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f'{name} must be finite and in [{low}, {high}]')
    cfg = config(side=claw_side)
    cfg.scene.num_envs = 32 if not play else 1
    cfg.episode_length_s = 30.
    cfg.commands = {}
    cfg.curriculum = {}
    robot = cfg.scene.entities['robot']
    stance_joints = tuple(cfg.actions['joint_pos'].actuator_names)
    stance = tuple((robot.init_state.joint_pos[n],) * 2 for n in stance_joints)
    cfg.events['hold_stance'] = EventTermCfg(
        func=hold_carried_arm, mode='reset', params={
            'ranges': stance, 'asset_cfg': SceneEntityCfg('robot', joint_names=stance_joints,
                                                         preserve_order=True)})
    solid = next(s for s in SOLIDS if s.name == 'soap')
    names, scales = ('soap_small', 'soap', 'soap_large'), (.8, 1., 1.2)
    for name, scale in zip(names, scales, strict=True):
        if name == 'soap':
            continue
        scaled = replace(solid, name=name, size=tuple(x * scale for x in solid.size),
                         mass=solid.mass * scale**3,
                         hull=tuple(tuple(x * scale for x in point) for point in solid.hull))
        cfg.scene.entities[name] = _prop(partial(_solid_spec, scaled),
                                         (2. + names.index(name) * .3, 0., scaled.size[2] + .002))
    for name in ('reset_prop_soap', 'reset_prop_dropbin'):
        cfg.events.pop(name)
    object_origin, bin_origin = fixture_positions('left')
    cfg.events['reset_pickup_layout'] = EventTermCfg(
        func=terms.reset_layout, mode='reset', params={
            'jitter_m': position_jitter_mm / 1000, 'side': claw_side,
            'object_names': names, 'scales': scales, 'object_size': solid.size,
            'object_xy': object_origin[:2], 'bin_xy': bin_origin[:2], 'tray_half': TRAY_INNER_HALF,
            'wall_thickness': BIN_WALL_T, 'object_size_mode': object_size})
    carried = tuple(cfg.events['hold_carried_arm'].params['asset_cfg'].joint_names)
    local_sign = np.array([-1, -1, 1]) if claw_side == 'right' else np.ones(3)
    cfg.actions = {'claw': ClawActionCfg(
        entity_name='robot', joint_names=carried, side=claw_side, arm_speed=.45,
        open_angle=BENCH_OPEN, closed_angle=GRIPPER_CLOSED, noise_m=.002,
        dropout=detection_dropout, local_point=tuple(FIXED_CONTACT_PALM * local_sign),
        tray_half=TRAY_INNER_HALF, floor_thickness=BIN_WALL_T, hull=solid.hull,
        object_names=names, object_scales=scales, object_size=solid.size)}
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    prefix = 'LF' if claw_side == 'left' else 'RF'
    sensors = []
    for i, name in enumerate(names):
        body = 'soap' if name == 'soap' else name
        secondary = ContactMatch(mode='body', pattern=body, entity=name)
        sensors.extend((
            ContactSensorCfg(name=f'pickup_grip_{i}',
                             primary=ContactMatch(mode='geom', pattern=tuple(
                                 f'{prefix}_{part}_grip_insert_link_meshcol' for part in ('palm', 'finger')),
                                 entity='robot'), secondary=secondary, fields=('found',)),
            ContactSensorCfg(name=f'pickup_robot_contact_{i}',
                             primary=ContactMatch(mode='subtree', pattern='base_link', entity='robot'),
                             secondary=secondary, fields=('found',)),
            ContactSensorCfg(name=f'pickup_object_contact_{i}',
                             primary=secondary, fields=('found',)),
            ContactSensorCfg(name=f'pickup_claw_contact_{i}',
                             primary=ContactMatch(mode='subtree', pattern=f'{prefix}_palm_link', entity='robot'),
                             secondary=secondary, fields=('found',)),
        ))
    cfg.scene.sensors = tuple(sensors)
    actor_terms = {'proprioception': ObservationTermCfg(func=terms.proprioception),
                   'simulated_detections': ObservationTermCfg(func=terms.detections)}
    cfg.observations = {
        'actor': ObservationGroupCfg(terms=actor_terms, history_length=3, nan_policy='error'),
        'critic': ObservationGroupCfg(terms=actor_terms | {
            'privileged_state': ObservationTermCfg(func=terms.privileged_state)}, nan_policy='error')}
    mirror = np.array([1, -1 if claw_side == 'right' else 1, 1])
    offset = tuple(GRASP_OFFSET * mirror)
    from .rl_cfg import DISCOUNT
    # Initial learning choices, owned here rather than hidden in MDP machinery.
    shaping = {'reach_std_m': .06, 'carry_std_m': .08,
               'lift_target_m': .05, 'carry_height_m': .095, 'grasp_offset': offset}
    cfg.rewards = {kind: RewardTermCfg(func=terms.PotentialReward, weight=weight,
                                      params=shaping | {'kind': kind, 'discount': DISCOUNT})
                   for kind, weight in (('reach', 1.), ('grip', 2.), ('lift', 4.),
                                        ('carry', 6.), ('release', 8.))}
    cfg.rewards['placed'] = RewardTermCfg(func=terms.reward, weight=1000.,
                                        params=shaping | {'kind': 'placed'})
    cfg.rewards['action_rate'] = RewardTermCfg(func=env_mdp.action_rate_l2, weight=-.005)
    cfg.terminations['placed'] = TerminationTermCfg(func=terms.outcome, params={'kind': 'success'})
    cfg.terminations['invalid_fixture'] = TerminationTermCfg(func=terms.outcome, params={'kind': 'invalid'})
    cfg.metrics = {name: MetricsTermCfg(func=terms.metric, params={'kind': kind}, reduce='max')
                   for name, kind in (('picked_up', 'picked'), ('placed', 'success'),
                                      ('invalid_fixture', 'invalid'), ('grip_contact', 'pinched'))}
    cfg.metrics['reach_distance_m'] = MetricsTermCfg(func=terms.metric,
                                                    params={'kind': 'reach_distance'})
    cfg.metrics['size_scale'] = MetricsTermCfg(func=terms.metric, params={'kind': 'size_scale'})
    cfg.metrics['floor_clearance_m'] = MetricsTermCfg(func=terms.metric,
                                                     params={'kind': 'clearance'}, reduce='max')
    return cfg
