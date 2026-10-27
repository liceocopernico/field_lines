import numpy as np
import pytest

from field_lines import sampling
from field_lines.sampling import Exclusion
from field_lines.sources import PointCharge, Superposition
from field_lines.units import CM, K, NC


def single(q=2 * NC):
    charge = PointCharge(q, [0, 0, 0])
    return Superposition([charge]), [Exclusion(charge.position, 0.5 * CM)]


def test_levels_of_a_symmetric_dipole_include_zero():
    model = Superposition([PointCharge(2 * NC, [-0.03, 0, 0]), PointCharge(-2 * NC, [0.03, 0, 0])])
    grid = sampling.view_grid(41, planar=False)
    V = model.potential(np.asarray(grid.points))
    levels = sampling.potential_levels(V, 9)
    assert len(levels) == 9
    assert levels[4] == pytest.approx(0.0, abs=1e-6 * np.abs(levels).max())


def test_equipotentials_of_a_point_charge_are_spheres():
    model, exclusions = single()
    mesh, levels = sampling.equipotentials(model, planar=False, count=3, exclusions=exclusions)
    r = np.linalg.norm(mesh.points, axis=1)
    V = model.potential(np.asarray(mesh.points))
    for level in levels:
        on_level = np.abs(V - level) < 0.02 * level
        radius = K * 2 * NC / level
        assert on_level.any()
        np.testing.assert_allclose(r[on_level], radius, rtol=0.05)


def test_equipotentials_in_2d_are_lines_in_the_plane():
    model, exclusions = single()
    mesh, _ = sampling.equipotentials(model, planar=True, count=5, exclusions=exclusions)
    assert mesh.n_lines > 0 and mesh.n_faces == 0
    assert np.abs(mesh.points[:, 2]).max() == 0


def test_arrows_point_along_the_field_and_skip_the_charge():
    model, exclusions = single()
    mesh = sampling.arrows(model, planar=True, exclusions=exclusions)
    assert mesh.n_points > 0
    assert "E" in mesh.point_data
    # the centre of the 17 x 17 grid is the charge: it gets no arrow
    centres = mesh.cell_centers().points
    assert np.linalg.norm(centres, axis=1).min() > 0.5 * CM


def test_slice_carries_V_and_E_of_the_model():
    model, _ = single()
    plane = sampling.slice_plane(model, "x", 0.02, planar=False)
    assert np.allclose(plane.points[:, 0], 0.02)
    p = np.asarray(plane.points)
    np.testing.assert_allclose(plane["V"], model.potential(p))
    np.testing.assert_allclose(plane["E"], np.linalg.norm(model.field(p), axis=1))


def test_slice_in_2d_is_the_plane_just_below_the_lines():
    model, _ = single()
    plane = sampling.slice_plane(model, "x", 0.05, planar=True)
    assert np.allclose(plane.points[:, 2], -sampling.PLANE_BELOW)
