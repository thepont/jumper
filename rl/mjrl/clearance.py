"""Mask-independent, explicit-pair clearance queries in private kinematic state.

Distances refer to simulator collision geometry, not necessarily CAD surfaces.
This is a sampled planning primitive, not a continuous or hardware safety proof.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class ClearanceResult:
    passed: bool
    distances: tuple[tuple[str, str, float], ...]


class GeometryClearance:
    """Adapter that queries specified pairs even when contact masks exclude them.

    Callers own pair coverage, justified interface exclusions and uncertainty
    margins. No automatic parent/same-limb exemptions are made here. Distances
    at the search cutoff are lower bounds; zero/ambiguous distances fail closed.
    One instance owns one scratch state and is not shared across worker threads.
    """

    def __init__(self, model, pairs, *, margin_m):
        if (isinstance(margin_m, (bool, np.bool_)) or
                not isinstance(margin_m, (int, float, np.integer, np.floating)) or
                not math.isfinite(margin_m) or not 0 < margin_m <= .1):
            raise ValueError('clearance margin must be finite in (0, .1] metres')
        if not isinstance(pairs, (list, tuple)) or not 1 <= len(pairs) <= 10000:
            raise ValueError('provide 1..10000 explicit geometry pairs')
        resolved, seen = [], set()
        for pair in pairs:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
                    any(not isinstance(n, str) or not n for n in pair)):
                raise ValueError('each pair needs two geometry names')
            ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in pair)
            key = tuple(sorted(ids))
            if min(ids) < 0 or ids[0] == ids[1] or key in seen:
                raise ValueError('geometry pairs must exist, be distinct and unique')
            seen.add(key)
            resolved.append((tuple(pair), ids))
        self.model, self.data = model, mujoco.MjData(model)
        self.pairs, self.margin_m = tuple(resolved), float(margin_m)

    def check(self, qpos):
        pose = np.asarray(qpos, dtype=float)
        if pose.shape != (self.model.nq,) or not np.isfinite(pose).all():
            raise ValueError('clearance needs a complete finite model qpos')
        self.data.qpos[:] = pose
        mujoco.mj_forward(self.model, self.data)
        distances = []
        for names, ids in self.pairs:
            distance = float(mujoco.mj_geomDistance(
                self.model, self.data, *ids, 2 * self.margin_m, None))
            if not math.isfinite(distance):
                raise RuntimeError('geometry query returned nonfinite clearance')
            distances.append((*names, distance))
        return ClearanceResult(all(d >= self.margin_m for _, _, d in distances), tuple(distances))
