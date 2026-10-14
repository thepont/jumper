"""Pickup-only chassis contact; walking retains its original collision policy."""
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations

import mujoco
from mjlab.utils.spec_config import CollisionCfg

from ..common.constants import CHASSIS_LINKS


CHASSIS_GEOMS = tuple(f'{link}_meshcol' for link in CHASSIS_LINKS)
CHASSIS_CONTACT_BIT = 1 << 7  # Ground and the six leg channels occupy bits 0..6.
ARM_GEOMS = r'^(LF|RF)_(?!shoulder).*_meshcol'


def nonadjacent_arm_pairs(model):
    """Conservative planning scope, independent of contact masks.

    Same rigid assemblies and directly adjacent bodies are excluded, not
    certified clear. Joint-interface geometry needs a separate audit; this
    scope must never be advertised as complete no-scrape verification.
    """
    geoms = [g for g in range(model.ngeom) if
             (model.geom(g).name or '').startswith('robot/') and
             '_meshcol' in model.geom(g).name]
    pairs = []
    for a, b in combinations(geoms, 2):
        names = (model.geom(a).name, model.geom(b).name)
        if not any(n.startswith(('robot/LF_', 'robot/RF_')) for n in names):
            continue
        ba, bb = (int(model.body_weldid[model.geom_bodyid[g]]) for g in (a, b))
        if ba == bb or model.body_parentid[ba] == bb or model.body_parentid[bb] == ba:
            continue
        pairs.append(names)
    if not pairs:
        raise ValueError('pickup clearance requires named front-arm collision geometry')
    return tuple(pairs)


def non_insert_object_pairs(model, object_name):
    """Teacher approach must not push an object with the palm/finger bodies."""
    if not isinstance(object_name, str) or not object_name:
        raise ValueError('object geometry scope needs a nonempty entity name')
    objects = [model.geom(g).name for g in range(model.ngeom)
               if (model.geom(g).name or '').startswith(object_name + '/')]
    arm = [model.geom(g).name for g in range(model.ngeom)
           if (model.geom(g).name or '').startswith('robot/LF_') and
           '_meshcol' in model.geom(g).name and '_grip_insert_' not in model.geom(g).name]
    if not objects or not arm:
        raise ValueError('teacher clearance requires object and left-arm geometry')
    return tuple((a, b) for a in arm for b in objects)


def non_claw_object_pairs(model, object_name):
    """Grasp permits palm-subtree contact, never support from other robot parts."""
    if not isinstance(object_name, str) or not object_name:
        raise ValueError('object geometry scope needs a nonempty entity name')
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot/LF_palm_link')
    if palm < 0:
        raise ValueError('left-claw geometry is required')
    objects = [model.geom(g).name for g in range(model.ngeom)
               if (model.geom(g).name or '').startswith(object_name + '/')]
    other = []
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ''
        if not name.startswith('robot/') or '_meshcol' not in name:
            continue
        body = int(model.geom_bodyid[geom])
        while body and body != palm:
            body = int(model.body_parentid[body])
        if body != palm:
            other.append(name)
    if not objects or not other:
        raise ValueError('grasp clearance requires object and non-claw robot geometry')
    return tuple((a, b) for a in other for b in objects)


@dataclass(frozen=True)
class ChassisSpec:
    """Decorate the bench spec factory without copying its jaw/side logic.

    Full CAD hulls conservatively keep an arm outside each chassis component;
    they do not claim to model free space inside hollow shells. Do not decimate
    them: the walking hull's inward approximation leaves parts of the visible
    robot unprotected. Keep explicit CAD inertias and all visual geoms intact.
    """

    source: Callable[[], mujoco.MjSpec]

    def __call__(self) -> mujoco.MjSpec:
        spec = self.source()
        for link in CHASSIS_LINKS:
            geom = spec.geom(f'{link}_meshcol')
            if geom is None or spec.mesh(link) is None:
                raise ValueError(f'pickup chassis requires the CAD mesh and collider for {link}')
            geom.meshname = link
        return spec


def with_chassis_contacts(base: CollisionCfg) -> CollisionCfg:
    """Extend, never replace, the bench's ground, cross-leg and grip settings.

    Only distal front-arm geoms emit the new channel. Chassis geoms receive it;
    stance hips and adjacent shoulder mounts retain their existing exclusions.
    First-match regex tables must put these entries before leg/catch-all rules.
    """
    if not isinstance(base.contype, dict) or not isinstance(base.conaffinity, dict):
        raise ValueError('pickup requires the bench per-leg collision mask tables')
    if any(f'^{side}_' not in base.contype for side in ('LF', 'RF')):
        raise ValueError('pickup requires both front-arm collision channels')
    return replace(
        base,
        geom_names_expr=base.geom_names_expr + (ARM_GEOMS,),
        contype={rf'^{side}_(?!shoulder).*_meshcol': base.contype[f'^{side}_'] | CHASSIS_CONTACT_BIT
                 for side in ('LF', 'RF')} | base.contype,
        conaffinity={'|'.join(f'^{name}$' for name in CHASSIS_GEOMS): CHASSIS_CONTACT_BIT}
                    | base.conaffinity,
    )
