import numpy as np
import pytest

from field_lines.sources import Plate, PointCharge, Superposition
from field_lines.units import EPS0, K, NC, format_quantity

RNG = np.random.default_rng(1)


def brute_force(plate: Plate, points: np.ndarray, n: int = 200):
    """E and V of a plate as a Gauss-Legendre sum over n x n point charges."""
    t, wt = np.polynomial.legendre.leggauss(n)
    u = t * plate.width / 2
    v = t * plate.height / 2
    uu, vv = np.meshgrid(u, v, indexing="ij")
    weights = np.outer(wt, wt).ravel() * plate.width * plate.height / 4
    local = np.column_stack((uu.ravel(), vv.ravel(), np.zeros(uu.size)))
    sources = plate.from_local(local) + plate.center
    r = points[:, None, :] - sources[None, :, :]
    d = np.linalg.norm(r, axis=2)
    dq = plate.sigma * weights
    E = K * np.einsum("mn,mnk->mk", dq / d**3, r)
    V = K * (dq / d).sum(axis=1)
    return E, V


def test_point_charge_matches_coulomb():
    q = PointCharge(2 * NC, [0.01, 0.02, -0.03])
    p = np.array([[0.05, 0.02, -0.03], [0.01, -0.02, 0.0]])
    r = p - q.position
    d = np.linalg.norm(r, axis=1)
    np.testing.assert_allclose(q.field(p), K * q.q * r / d[:, None] ** 3)
    np.testing.assert_allclose(q.potential(p), K * q.q / d)


def test_point_charge_is_zero_on_itself():
    q = PointCharge(NC, [0.0, 0.0, 0.0])
    p = np.zeros((1, 3))
    assert np.all(q.field(p) == 0) and q.potential(p)[0] == 0


@pytest.mark.parametrize("normal", ["x", "y", "z"])
def test_plate_matches_brute_force(normal):
    plate = Plate(3e-7, [0.01, -0.02, 0.005], width=0.10, height=0.06, normal=normal)
    local = np.column_stack(
        (RNG.uniform(-0.12, 0.12, 40), RNG.uniform(-0.08, 0.08, 40), RNG.choice([-1, 1], 40) * RNG.uniform(0.01, 0.1, 40))
    )
    points = plate.from_local(local) + plate.center
    E_ref, V_ref = brute_force(plate, points)
    np.testing.assert_allclose(plate.field(points), E_ref, rtol=1e-4, atol=1e-4 * np.abs(E_ref).max())
    np.testing.assert_allclose(plate.potential(points), V_ref, rtol=1e-5)


def test_plate_in_its_own_plane_outside_and_on_edge_lines():
    plate = Plate(1e-7, [0, 0, 0], width=0.1, height=0.1)
    points = np.array([[0.08, 0.0, 0.0], [0.05, 0.09, 0.0], [0.07, 0.06, 0.0], [-0.05, -0.2, 0.0]])
    E, V = plate.field(points), plate.potential(points)
    assert np.all(np.isfinite(E)) and np.all(np.isfinite(V))
    np.testing.assert_array_equal(E[:, 2], 0.0)
    E_ref, V_ref = brute_force(plate, points, n=400)
    np.testing.assert_allclose(E, E_ref, rtol=2e-3, atol=1e-3 * np.abs(E_ref).max())
    np.testing.assert_allclose(V, V_ref, rtol=1e-4)


def test_plate_far_away_looks_like_a_point_charge():
    plate = Plate(2e-7, [0, 0, 0], width=0.1, height=0.05)
    point = PointCharge(plate.charge, [0, 0, 0])
    p = np.array([[3.0, 1.0, 2.0], [-2.0, 0.5, -4.0]])
    np.testing.assert_allclose(plate.field(p), point.field(p), rtol=1e-3)
    np.testing.assert_allclose(plate.potential(p), point.potential(p), rtol=1e-3)


def test_large_plate_gives_sigma_over_two_eps0():
    plate = Plate(1e-6, [0, 0, 0], width=10.0, height=10.0)
    E = plate.field(np.array([[0.0, 0.0, 0.001], [0.0, 0.0, -0.001]]))
    np.testing.assert_allclose(E[:, 2], [plate.field_ideal(), -plate.field_ideal()], rtol=1e-3)


def test_minus_gradient_of_potential_is_the_field():
    model = Superposition([Plate(1e-7, [0, 0, 0.01], 0.08, 0.08), PointCharge(-NC, [0.02, 0.01, -0.02])])
    p = RNG.uniform(-0.06, 0.06, (20, 3))
    h = 1e-6
    grad = np.column_stack(
        [(model.potential(p + h * e) - model.potential(p - h * e)) / (2 * h) for e in np.eye(3)]
    )
    np.testing.assert_allclose(-grad, model.field(p), rtol=1e-5, atol=1e-6 * np.abs(grad).max())


def test_large_capacitor_textbook_values():
    sigma, d = 1e-6, 0.01
    capacitor = Superposition(
        [Plate(sigma, [0, 0, d / 2], 2.0, 2.0), Plate(-sigma, [0, 0, -d / 2], 2.0, 2.0)]
    )
    inside = capacitor.field(np.array([[0.0, 0.0, 0.0]]))[0]
    outside = capacitor.field(np.array([[0.0, 0.0, 2 * d]]))[0]
    assert inside[2] == pytest.approx(-sigma / EPS0, rel=1e-2)  # from + (top) to - (bottom)
    assert abs(outside[2]) < 0.01 * sigma / EPS0
    dV = capacitor.potential(np.array([[0, 0, d / 2], [0, 0, -d / 2]]))
    assert dV[0] - dV[1] == pytest.approx(sigma * d / EPS0, rel=1e-2)


def test_format_quantity():
    assert format_quantity(22600.0, "V/m") == "22.6 kV/m"
    assert format_quantity(2e-9, "C") == "2.00 nC"
    assert format_quantity(-0.5, "V") == "-500 mV"
    assert format_quantity(0.0, "V") == "0 V"


def test_ring_distance_and_nearest_surface_point():
    from field_lines.sources import Ring

    tube = Ring(0.0, [0.0, 0.0, 0.0], 0.02, 0.022, 0.05)  # 2 mm wall, 10 cm long
    points = np.array([[0.01, 0, 0], [0.021, 0, 0], [0.0, 0.021, 0.06], [0.03, 0, 0.0]])
    np.testing.assert_allclose(tube.distances(points), [0.01, 0.0, 0.01, 0.008], atol=1e-12)
    # from inside the wall, out through the nearer cylinder; from the hole, onto the inner one
    np.testing.assert_allclose(tube.surface_point(np.array([0.0205, 0, 0])), [0.02, 0, 0], atol=1e-12)
    np.testing.assert_allclose(tube.surface_point(np.array([0, 0.0215, 0.01])), [0, 0.022, 0.01], atol=1e-12)
    np.testing.assert_allclose(tube.surface_point(np.array([0.01, 0, 0.0])), [0.02, 0, 0], atol=1e-12)
    disc = Ring(0.0, [0.0, 0.0, 0.0], 0.0, 0.05, 0.001, gap="w-")
    np.testing.assert_allclose(disc.surface_point(np.array([0.0, 0.0, 0.0005])), [0, 0, 0.001], atol=1e-12)
    assert disc.area == pytest.approx(np.pi * 0.05**2)
    assert tube.area == pytest.approx(np.pi * (0.022**2 - 0.02**2))  # gap "w+" by default: a flat face
    tube.gap = "out"
    assert tube.area == pytest.approx(2 * np.pi * 0.022 * 0.1)
    tube.gap = "in"
    assert tube.area == pytest.approx(2 * np.pi * 0.02 * 0.1)


def test_ring_entry_catches_steps_across_a_thin_wall():
    from field_lines.sources import Ring

    tube = Ring(0.0, [0.0, 0.0, 0.0], 0.02, 0.0205, 0.05)  # a 0.5 mm wall
    p = np.array([[0.015, 0, 0], [0.0, 0.03, 0.0], [0.0, 0.0, -0.08], [0.019, 0, 0], [0.0202, 0, 0], [0.03, 0, 0.049]])
    q = np.array([[0.03, 0, 0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.08], [0.0195, 0, 0], [0.03, 0, 0], [0.03, 0, 0.06]])
    t = tube.entry(p, q)
    assert t[0] == pytest.approx(5 / 15)  # out through the wall, from the hole: enters at ρ = 2 cm
    assert t[1] == pytest.approx(0.95 / 2)  # in from outside: enters at ρ = 2.05 cm
    assert np.isnan(t[2]) and np.isnan(t[3])  # along the axis, and within the hole: never in the wall
    assert np.isnan(t[4])  # starting inside the wall
    assert np.isnan(t[5])  # outside, past the end
    disc = Ring(0.0, [0.0, 0.0, 0.0], 0.0, 0.05, 0.001)
    t = disc.entry(np.array([[0.01, 0, 0.01], [0.06, 0, 0.0]]), np.array([[0.01, 0, -0.01], [0.04, 0, 0.0]]))
    assert t[0] == pytest.approx(0.45) and t[1] == pytest.approx(0.5)  # the top face, the rim
