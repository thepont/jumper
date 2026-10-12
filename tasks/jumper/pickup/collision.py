"""Pickup-only chassis contact; walking retains its original collision policy."""
from collections.abc import Callable
from dataclasses import dataclass, replace

import mujoco
from mjlab.utils.spec_config import CollisionCfg

from ..common.constants import CHASSIS_LINKS


CHASSIS_GEOMS = tuple(f'{link}_meshcol' for link in CHASSIS_LINKS)
CHASSIS_CONTACT_BIT = 1 << 7  # Ground and the six leg channels occupy bits 0..6.
ARM_GEOMS = r'^(LF|RF)_(?!shoulder).*_meshcol'


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
