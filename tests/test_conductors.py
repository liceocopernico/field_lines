import math

import numpy as np
import pytest
from scipy.sparse.linalg import cg

from field_lines import conductors, presets
from field_lines.conductors import ConductorSolution
from field_lines.scene import Scene
from field_lines.units import EPS0, K

L = 0.10  # plate side


@pytest.fixture(scope="module")
def grid():
    """10 cm square plates, 3 cm apart."""
    return conductors.solve(L, L, 0.03)


def test_plates_are_at_plus_and_minus_half_the_voltage(grid):
    solution = ConductorSolution(grid, 500.0, "z")
    points = np.array([[0, 0, 0.015], [0.03, -0.04, 0.015], [0, 0, -0.015], [-0.02, 0.01, -0.015], [0, 0, 0]])
    np.testing.assert_allclose(solution.potential(points), [250, 250, -250, -250, 0], atol=1e-9)


def test_field_between_the_centres_is_voltage_over_gap(grid):
    solution = ConductorSolution(grid, 500.0, "z")
    E = solution.field(np.zeros((1, 3)))[0]
    assert E[2] == pytest.approx(-500 / 0.03, rel=0.01)  # from the + plate (top) down to the - plate
    assert abs(E[0]) < 1e-9 and abs(E[1]) < 1e-9


def test_capacitance_with_fringing_at_d_over_l_one_tenth():
    d = 0.1 * L
    C = conductors.solve(L, L, d).capacitance
    reference = conductors.solve(L, L, d, refine=1.25).capacitance
    ideal = EPS0 * L * L / d
    assert C == pytest.approx(reference, rel=0.05)
    assert C > ideal
    # Kirchhoff's first-order fringing for thin discs, for a square of the same area,
    # with the edge term scaled by the perimeter ratio 4 / (2√π)
    a = L / math.sqrt(math.pi)
    kirchhoff = ideal + EPS0 * a * (math.log(16 * math.pi * a / d) - 1) * 4 / (2 * math.sqrt(math.pi))
    assert C == pytest.approx(kirchhoff, rel=0.05)


def test_charge_piles_up_at_the_edges(grid):
    centre = grid.sigma[0, 0]
    edge = grid.sigma[-1, 0]
    corner = grid.sigma[-1, -1]
    assert centre > 0
    assert edge > 2 * centre
    assert corner > edge


def gauss_flux(solution, center, half, n=60):
    """ε₀ × the flux of E out of a box around center: the charge inside."""
    t = (np.arange(n) + 0.5) / n * 2 - 1
    a, b = np.meshgrid(t, t, indexing="ij")
    total = 0.0
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        area = (2 * half[others[0]] / n) * (2 * half[others[1]] / n)
        for side in (-1, 1):
            points = np.empty((n * n, 3))
            points[:, axis] = center[axis] + side * half[axis]
            points[:, others[0]] = center[others[0]] + a.ravel() * half[others[0]]
            points[:, others[1]] = center[others[1]] + b.ravel() * half[others[1]]
            total += side * solution.field(points)[:, axis].sum() * area
    return EPS0 * total


def test_gauss_law_gives_equal_and_opposite_plate_charges(grid):
    solution = ConductorSolution(grid, 100.0, "z")
    half = np.array([0.065, 0.065, 0.008])
    plus = gauss_flux(solution, np.array([0, 0, 0.015]), half)
    minus = gauss_flux(solution, np.array([0, 0, -0.015]), half)
    assert plus == pytest.approx(solution.charge, rel=0.05)
    assert minus == pytest.approx(-solution.charge, rel=0.05)


def test_octant_matches_a_solve_of_the_whole_grid():
    width, height, d = 0.06, 0.04, 0.02
    octant = conductors.solve(width, height, d, refine=0.5)
    axes = [np.concatenate((-a[:0:-1], a)) for a in octant.axes]  # mirror every axis
    shape = tuple(len(a) for a in axes)
    A = conductors._laplacian(axes)
    fixed = np.zeros(shape, dtype=bool)
    fixed[[0, -1], :, :] = fixed[:, [0, -1], :] = fixed[:, :, [0, -1]] = True
    u, v, w = axes
    # the box at the potentials the octant solve gave it, mirrored (odd across the midplane)
    nodes = np.stack(np.meshgrid(u, v, w, indexing="ij"), axis=-1)[fixed]
    V = np.zeros(shape)
    V[fixed] = np.sign(nodes[:, 2]) * octant.interpolator(np.abs(nodes))[:, 0]
    on_plate = (np.abs(u) <= width / 2 + 1e-12)[:, None] & (np.abs(v) <= height / 2 + 1e-12)[None, :]
    for sign in (1, -1):
        k = int(np.argmin(np.abs(w - sign * d / 2)))
        fixed[:, :, k] |= on_plate
        V[:, :, k][on_plate] = sign * 0.5
    free = ~fixed.ravel()
    x, info = cg(A[free][:, free], -(A[free][:, ~free] @ V.ravel()[~free]), rtol=1e-10, maxiter=5000)
    assert info == 0
    V.ravel()[free] = x
    solution = ConductorSolution(octant, 1.0, "z")
    grid_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    np.testing.assert_allclose(solution.potential(grid_points), V.ravel(), atol=1e-6)


def test_turned_capacitor_is_the_same_field_with_permuted_axes(grid):
    along_z = ConductorSolution(grid, 100.0, "z")
    along_x = ConductorSolution(grid, 100.0, "x")
    p = np.random.default_rng(3).uniform(-0.08, 0.08, (30, 3))
    # normal z: (u, v, w) = (x, y, z); normal x: (u, v, w) = (y, z, x)
    p_x = p[:, [2, 0, 1]]
    np.testing.assert_allclose(along_x.potential(p_x), along_z.potential(p))
    np.testing.assert_allclose(along_x.field(p_x), along_z.field(p)[:, [2, 0, 1]])


def test_min_separation_fits_the_budget():
    d = conductors.min_separation(L, L)
    assert conductors.unknowns(L, L, d) <= conductors.MAX_UNKNOWNS
    assert conductors.unknowns(L, L, 0.95 * d) > conductors.MAX_UNKNOWNS
    with pytest.raises(ValueError):
        conductors.solve(L, L, 0.5 * d)


def test_conductor_capacitor_in_2d_keeps_lines_in_the_plane():
    scene = Scene()
    scene.planar = True
    presets.load(scene, "capacitor")
    cap = scene.capacitor
    cap.model = "conductor"
    scene.conductor = ConductorSolution(conductors.solve(cap.width, cap.height, cap.separation), cap.voltage, cap.normal)
    lines = scene.lines()
    assert lines
    assert all(np.abs(line.points[:, 2]).max() < 1e-12 for line in lines)
    assert all(line.end in ("plate", "bounds") for line in lines)
    readouts = scene.capacitor_readouts()
    assert readouts["dV"] == pytest.approx(cap.voltage)
    assert readouts["C"] > readouts["C_ideal"]


def test_field_next_to_the_plates_points_the_right_way(grid):
    """A central difference across the one-node plate would make E point back at it from outside."""
    solution = ConductorSolution(grid, 500.0, "z")
    across = np.linspace(-0.049, 0.049, 40)
    for z, expected_sign in ((0.0155, 1), (0.0145, -1), (-0.0145, -1), (-0.0155, 1)):
        # just outside the + plate E points up, away from it; between the plates down; outside - up into it
        points = np.column_stack((across, np.zeros(40), np.full(40, z)))
        Ez = solution.field(points)[:, 2]
        assert np.all(np.sign(Ez) == expected_sign), z
    centre = solution.field(np.array([[0.0, 0.0, 0.0145]]))[0, 2]
    assert centre == pytest.approx(-500 / 0.03, rel=0.01)  # the full gap field, right up to the plate


def test_lines_run_from_plate_surface_to_plate_surface():
    scene = Scene()
    scene.planar = True
    presets.load(scene, "capacitor")
    cap = scene.capacitor
    cap.model = "conductor"
    scene.conductor = ConductorSolution(conductors.solve(cap.width, cap.height, cap.separation), cap.voltage, cap.normal)
    lines = scene.lines()
    for line in lines:
        assert line.points[0, 0] == pytest.approx(cap.separation / 2)  # on the + plate (normal x)
        if line.end == "plate":
            assert line.points[-1, 0] == pytest.approx(-cap.separation / 2)
    inner = [line for line in lines if line.points[1, 0] < cap.separation / 2]
    # near the edges the gap lines bulge outwards (fringing) instead of running straight across
    outermost = max(inner, key=lambda line: abs(line.points[0, 1]))
    assert np.abs(outermost.points[:, 1]).max() > cap.width / 2


def test_the_field_carries_on_smoothly_outside_the_box(grid):
    """Outside the grid's box the field is that of the plates' charges, which also set the box's potential."""
    solution = ConductorSolution(grid, 1.0, "z")
    end = grid.axes[0][-1]
    inside = np.array([[0.999 * end, 0.02, 0.05], [0.03, 0.04, 0.999 * end]])
    outside = inside * np.array([[1.002 / 0.999, 1, 1], [1, 1, 1.002 / 0.999]])
    np.testing.assert_allclose(solution.potential(outside), solution.potential(inside), rtol=0.03)
    np.testing.assert_allclose(solution.field(outside), solution.field(inside), rtol=0.1, atol=2e-3)
    # far away: the field of a dipole, Q d along the axis
    far = solution.field(np.array([[0.0, 0.0, 5.0]]))[0]
    dipole = 2 * K * solution.charge * grid.separation / 5.0**3
    assert far[2] == pytest.approx(dipole, rel=0.05)


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("model", ["sigma", "conductor"])
def test_every_line_from_the_positive_plate_ends_on_the_negative_plate(planar, model):
    scene = Scene()
    scene.planar = planar
    presets.load(scene, "capacitor")
    cap = scene.capacitor
    cap.model = model
    if model == "conductor":
        grid = conductors.solve(cap.width, cap.height, cap.separation)
        scene.conductor = ConductorSolution(grid, cap.voltage, cap.normal)
    lines = scene.lines()
    assert lines
    assert all(line.end == "plate" for line in lines)
