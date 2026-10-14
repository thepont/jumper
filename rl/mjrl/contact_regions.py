"""Finite planar pad/box-face overlap for candidate generation, not grasp evidence.

All coordinates are in the object's frame, in metres. Region area is projected
onto the selected box face. It does not describe pressure, actual contact,
friction or force closure; callers still need collision and physical checks.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class FaceWindow:
    """Validated face and signed-gap band; owns values rather than caller arrays."""

    half_extents_m: tuple[float, float, float]
    normal_axis: int
    sign: int
    gap_m: float

    def __post_init__(self):
        half = np.asarray(self.half_extents_m, dtype=float)
        if (half.shape != (3,) or not np.isfinite(half).all() or (half <= 0).any()
                or type(self.normal_axis) is not int or self.normal_axis not in (0, 1, 2)
                or type(self.sign) is not int or self.sign not in (-1, 1)
                or type(self.gap_m) not in (float, int)
                or not math.isfinite(self.gap_m) or self.gap_m <= 0):
            raise ValueError('face needs positive finite half extents, scalar axis/sign and gap')
        object.__setattr__(self, 'half_extents_m', tuple(float(v) for v in half))
        object.__setattr__(self, 'gap_m', float(self.gap_m))


@dataclass(frozen=True)
class ContactRegion:
    """Geometric overlap only. Not a physical contact-centroid observation."""

    area_m2: float
    centroid_m: tuple[float, float, float]
    central_fraction: float


def _clip(vertices, axis, bound, sign):
    if not len(vertices):
        return vertices
    result = []
    previous = vertices[-1]
    before = sign * previous[axis] - bound
    for current in vertices:
        after = sign * current[axis] - bound
        if (before <= 0) != (after <= 0):
            result.append(previous + (current - previous) * (before / (before - after)))
        if after <= 0:
            result.append(current)
        previous, before = current, after
    return np.asarray(result).reshape(-1, 3)


def contact_region(vertices_m, window: FaceWindow) -> ContactRegion | None:
    """Clip a planar convex pad facet to a box face and its +/- gap band.

    Vertex order is irrelevant. A zero-area intersection returns None. The
    caller chooses the actual pad facet; unrelated mesh vertices are invalid.
    """
    vertices = np.asarray(vertices_m, dtype=float)
    if (not isinstance(window, FaceWindow) or vertices.ndim != 2
            or vertices.shape[1:] != (3,) or not 3 <= len(vertices) <= 256
            or not np.isfinite(vertices).all()):
        raise ValueError('region needs a face window and 3..256 finite planar 3D vertices')
    centred = vertices - vertices.mean(axis=0)
    _, singular, vh = np.linalg.svd(centred, full_matrices=False)
    if singular[1] < 1e-12 or np.max(np.abs(centred @ vh[-1])) > 1e-7:
        raise ValueError('pad facet must be nondegenerate and planar')
    tangents = [axis for axis in (0, 1, 2) if axis != window.normal_axis]
    projected = vertices[:, tangents]
    if np.linalg.matrix_rank(projected - projected.mean(axis=0), tol=1e-12) < 2:
        return None
    polygon = vertices[ConvexHull(projected).vertices]
    for axis in tangents:
        polygon = _clip(polygon, axis, window.half_extents_m[axis], 1)
        polygon = _clip(polygon, axis, window.half_extents_m[axis], -1)
    face = window.sign * window.half_extents_m[window.normal_axis]
    polygon = _clip(polygon, window.normal_axis, face + window.gap_m, 1)
    polygon = _clip(polygon, window.normal_axis, -face + window.gap_m, -1)
    if len(polygon) < 3:
        return None
    triangles = np.array([[polygon[0], polygon[i], polygon[i + 1]]
                          for i in range(1, len(polygon) - 1)])
    a = triangles[:, 1, tangents] - triangles[:, 0, tangents]
    b = triangles[:, 2, tangents] - triangles[:, 0, tangents]
    areas = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    area = float(areas.sum())
    if area <= 1e-16:
        return None
    centroid = np.sum(triangles.mean(axis=1) * areas[:, None], axis=0) / area
    fraction = float(np.max(np.abs(centroid[tangents]) /
                            np.asarray(window.half_extents_m)[tangents]))
    return ContactRegion(area, tuple(float(v) for v in centroid), fraction)
