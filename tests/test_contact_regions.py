"""Flat-face candidate geometry must not inherit an arbitrary closest corner."""
import numpy as np
import pytest

from mjrl.contact_regions import FaceWindow, contact_region


def patch(y=.0185):
    return np.array([[-.026, y, -.020], [.026, y, -.020],
                     [.026, y, .020], [-.026, y, .020]])


def test_given_parallel_faces_when_region_is_measured_then_corner_choice_does_not_reject_it():
    window = FaceWindow((.03, .018, .047), 1, 1, .002)
    for vertices in (patch(), patch()[::-1], patch()[[2, 0, 3, 1]]):
        region = contact_region(vertices, window)
        assert region is not None
        assert region.area_m2 == pytest.approx(.052 * .04)
        assert region.centroid_m == pytest.approx((0., .0185, 0.))
        assert region.central_fraction == pytest.approx(0.)
    # The former corner proxy would reject the same parallel near-contact patch.
    assert np.max(np.abs(patch()[0, [0, 2]]) / [.03, .047]) > .35


def test_given_offset_patch_when_clipped_then_only_the_object_face_overlap_counts():
    region = contact_region(patch() + [.02, 0., 0.], FaceWindow((.03, .018, .047), 1, 1, .002))
    assert region.area_m2 == pytest.approx(.036 * .04)
    assert region.centroid_m == pytest.approx((.012, .0185, 0.))
    assert region.central_fraction == pytest.approx(.4)


def test_far_and_edge_only_regions_are_not_positive_area_candidates():
    window = FaceWindow((.03, .018, .047), 1, 1, .002)
    assert contact_region(patch(y=.023), window) is None
    assert contact_region(patch() + [.056, 0., 0.], window) is None


def test_tilted_patch_is_clipped_to_the_gap_band_not_flattened():
    vertices = patch()
    vertices[:, 1] = .018 + vertices[:, 0] * .2
    region = contact_region(vertices, FaceWindow((.03, .018, .047), 1, 1, .002))
    assert region.area_m2 == pytest.approx(.020 * .04)
    assert region.centroid_m == pytest.approx((0., .018, 0.))


def test_reflected_opposing_face_has_the_same_central_region():
    vertices = patch()
    vertices[:, 1] *= -1
    region = contact_region(vertices, FaceWindow((.03, .018, .047), 1, -1, .002))
    assert region.centroid_m == pytest.approx((0., -.0185, 0.))


def test_face_axis_permutation_preserves_units_area_and_centroid():
    permutation = [1, 2, 0]
    vertices = patch()[:, permutation]
    half = np.array([.03, .018, .047])[permutation]
    region = contact_region(vertices, FaceWindow(tuple(half), 0, 1, .002))
    assert region.area_m2 == pytest.approx(.052 * .04)
    assert region.centroid_m == pytest.approx((.0185, 0., 0.))


def test_analysis_owns_the_face_contract_and_leaves_input_buffers_unchanged():
    half = [.03, .018, .047]
    window = FaceWindow(half, 1, 1, .002)
    half[0] = 99.
    vertices = patch()
    before = vertices.copy()
    region = contact_region(vertices, window)
    assert np.array_equal(vertices, before)
    assert region.area_m2 == pytest.approx(.052 * .04)


@pytest.mark.parametrize('half,axis,sign,gap', [
    ((.03, -.018, .047), 1, 1, .002), ((.03, .018, .047), True, 1, .002),
    ((.03, .018, .047), 1, 0, .002), ((.03, .018, .047), 1, 1, float('nan')),
])
def test_invalid_face_contract_is_rejected_at_the_boundary(half, axis, sign, gap):
    with pytest.raises(ValueError):
        FaceWindow(half, axis, sign, gap)


def test_nonplanar_and_nonfinite_input_never_becomes_contact_evidence():
    vertices = patch()
    vertices[0, 1] += .001
    window = FaceWindow((.03, .018, .047), 1, 1, .002)
    with pytest.raises(ValueError):
        contact_region(vertices, window)
    with pytest.raises(ValueError):
        contact_region([[float('nan'), 0., 0.]] * 4, window)
