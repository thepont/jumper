"""Mask-independent, explicit-pair clearance queries in private kinematic state.

Distances refer to simulator collision geometry, not necessarily CAD surfaces.
This is a sampled planning primitive, not a continuous or hardware safety proof.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class ClearanceResult:
    passed: bool
    distances: tuple[tuple[str, str, float], ...]
    certified_pairs: tuple[tuple[str, str], ...] = ()


class GeometryClearance:
    """Adapter that queries specified pairs even when contact masks exclude them.

    Callers own pair coverage, justified interface exclusions and uncertainty
    margins. No automatic parent/same-limb exemptions are made here. Distances
    at the search cutoff are lower bounds. Zero mesh distances can pass only
    with an independent separating-plane certificate; other ambiguity fails
    closed. Certified values are lower bounds, not exact surface distances.
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
        self._meshes = {}
        self._mesh_edges = {}

    def _mesh(self, geom):
        if geom not in self._meshes:
            if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                self._meshes[geom] = None
            else:
                mesh = int(self.model.geom_dataid[geom])
                start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
                vertices = self.model.mesh_vert[start:start + count].astype(float)
                try:
                    hull = ConvexHull(vertices)
                    normals = hull.equations[:, :3]
                    triangles = hull.simplices
                    indices = np.unique(np.sort(np.concatenate([
                        triangles[:, [0, 1]], triangles[:, [1, 2]],
                        triangles[:, [2, 0]]]), axis=1), axis=0)
                    edges = vertices[indices[:, 1]] - vertices[indices[:, 0]]
                    lengths = np.linalg.norm(edges, axis=1)
                    self._mesh_edges[geom] = edges[lengths > 1e-12] / lengths[lengths > 1e-12, None]
                    self._meshes[geom] = (vertices, normals)
                except QhullError:
                    self._meshes[geom] = None
        return self._meshes[geom]

    def _separation_bound(self, ids):
        meshes = [self._mesh(g) for g in ids]
        if any(mesh is None for mesh in meshes):
            return None
        points, axes = [], []
        for geom, (vertices, normals) in zip(ids, meshes, strict=True):
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            points.append(vertices @ rotation.T + self.data.geom_xpos[geom])
            axes.append(normals @ rotation.T)
        directions = np.concatenate(axes)
        projections = [p @ directions.T for p in points]
        a, b = projections
        gap = float(np.maximum(b.min(axis=0) - a.max(axis=0),
                               a.min(axis=0) - b.max(axis=0)).max())
        if gap - 1e-8 < self.margin_m:
            # Face normals alone miss skew edge/edge separation. Any additional
            # unit axis still gives a conservative hull-distance lower bound.
            # Bound work and fail closed if the candidate budget is exceeded;
            # this remains a certificate search, not an intersection verdict.
            edges = [self._mesh_edges[g] @ self.data.geom_xmat[g].reshape(3, 3).T
                     for g in ids]
            if len(edges[0]) * len(edges[1]) <= 250000:
                for edge in edges[0]:
                    cross = np.cross(edge, edges[1])
                    lengths = np.linalg.norm(cross, axis=1)
                    directions = cross[lengths > 1e-10] / lengths[lengths > 1e-10, None]
                    for start in range(0, len(directions), 256):
                        axes = directions[start:start+256]
                        a, b = [p @ axes.T for p in points]
                        candidate = float(np.maximum(b.min(axis=0) - a.max(axis=0),
                                                     a.min(axis=0) - b.max(axis=0)).max())
                        gap = max(gap, candidate)
                        if gap - 1e-8 >= self.margin_m:
                            return gap - 1e-8
        # Any exhibited separating plane proves a lower bound for the convex
        # hulls. A nonpositive result proves nothing; retain fail-closed zero.
        return gap - 1e-8 if math.isfinite(gap) and gap > 1e-8 else None

    def check(self, qpos):
        pose = np.asarray(qpos, dtype=float)
        if pose.shape != (self.model.nq,) or not np.isfinite(pose).all():
            raise ValueError('clearance needs a complete finite model qpos')
        self.data.qpos[:] = pose
        mujoco.mj_forward(self.model, self.data)
        distances, certified = [], []
        for names, ids in self.pairs:
            distance = float(mujoco.mj_geomDistance(
                self.model, self.data, *ids, 2 * self.margin_m, None))
            if not math.isfinite(distance):
                raise RuntimeError('geometry query returned nonfinite clearance')
            if distance == 0:
                bound = self._separation_bound(ids)
                if bound is not None and bound >= self.margin_m:
                    distance = bound
                    certified.append(names)
            distances.append((*names, distance))
        return ClearanceResult(all(d >= self.margin_m for _, _, d in distances),
                               tuple(distances), tuple(certified))

    def check_contact_depth(self, qpos, *, max_penetration_m):
        """Opt-in EXPECTED-contact check; never use for forbidden robot contact.

        Reuse the same pose, pair and native-query authority as positive clearance.
        A zero query needs either a separating plane or the compiled collision
        manifold's signed depth. Missing manifold evidence stays unresolved,
        including mask-disabled pairs. This proves neither grip load nor capture.
        """
        if (isinstance(max_penetration_m, (bool, np.bool_)) or
                not isinstance(max_penetration_m, (int, float, np.integer, np.floating)) or
                not math.isfinite(max_penetration_m) or not 0 <= max_penetration_m <= .1):
            raise ValueError('expected-contact depth must be finite within [0, .1] metres')
        baseline = self.check(qpos)
        distances, verdicts = [], []
        for (a, b, distance), (_, ids) in zip(baseline.distances, self.pairs, strict=True):
            manifold = [float(c.dist) for c in self.data.contact[:self.data.ncon]
                        if {int(c.geom1), int(c.geom2)} == set(ids)]
            if any(not math.isfinite(d) for d in manifold):
                raise RuntimeError('contact manifold returned nonfinite depth')
            resolved = distance != 0.
            if manifold:
                distance = min(manifold) if distance == 0. else min(distance, *manifold)
                resolved = True
            elif not resolved:
                bound = self._separation_bound(ids)
                if bound is not None:
                    distance, resolved = bound, True
            distances.append((a, b, distance))
            verdicts.append(resolved and distance >= -max_penetration_m)
        return ClearanceResult(all(verdicts), tuple(distances), baseline.certified_pairs)
