import math

import numpy as np
import pytest

from field_lines import conductors, presets
from field_lines.conductors import Solver, Sphere, evaluate, touching
from field_lines.scene import Capacitor, Scene
from field_lines.units import EPS0, K

L = 0.10  # plate side


@pytest.fixture(scope="module")
def capacitor():
    """10 cm square plates, 3 cm apart, at ±250 V."""
    cap = Capacitor()
    return cap, Solver(cap.plates(), []).solve([250.0, -250.0])


def conductor_scene(key: str, planar: bool = False) -> Scene:
    scene = Scene()
    scene.planar = planar
    scene.kind = "conductors"
    presets.load(scene, key)
    scene.conductor = Solver(scene.conductor_plates(), scene.spheres).solve(scene.voltages())
    return scene


def test_isolated_sphere_charge_and_density():
    R, V = 0.02, 1000.0
    solution = Solver([], [Sphere([0.01, -0.02, 0.03], R, V)]).solve([V])
    assert solution.charge(0) == pytest.approx(4 * math.pi * EPS0 * R * V, rel=5e-3)
    np.testing.assert_allclose(solution.sphere_sigma(0), EPS0 * V / R, rtol=0.08)


def test_isolated_sphere_looks_like_a_point_charge_outside():
    center = np.array([0.01, -0.02, 0.03])
    solution = Solver([], [Sphere(center, 0.02, 1000.0)]).solve([1000.0])
    Q = solution.charge(0)
    points = center + np.array([[0.05, 0, 0], [0, 0.03, 0.04], [0.0, 0.0, 0.025], [1.0, 2.0, -2.0]])
    r = points - center
    d = np.linalg.norm(r, axis=1)
    np.testing.assert_allclose(solution.potential(points), K * Q / d, rtol=2e-3)
    exact = K * Q * r / d[:, None] ** 3
    np.testing.assert_allclose(solution.field(points), exact, rtol=1e-2, atol=1e-4 * np.abs(exact).max())
    assert np.linalg.norm(solution.field(center[None])[0]) < 0.02 * K * Q / 0.02**2  # no field inside a conductor


def test_plates_are_at_their_potentials(capacitor):
    _, solution = capacitor
    panels = solution.panels
    target = np.where(panels.owner == 0, 250.0, -250.0)
    np.testing.assert_allclose(solution.potential(panels.centers), target, atol=0.5)  # clusters: within 0.2%
    between = np.array([[0.03, -0.04, 0.015], [-0.02, 0.01, -0.015]])
    np.testing.assert_allclose(solution.potential(between), [250, -250], rtol=0.01)


def test_capacitor_gap_field_and_charges(capacitor):
    _, solution = capacitor
    E = solution.field(np.zeros((1, 3)))[0]
    assert E[2] == pytest.approx(-500 / 0.03, rel=0.01)  # from the + plate (top) to the - plate
    assert abs(E[0]) < 1 and abs(E[1]) < 1
    assert solution.charge(0) == pytest.approx(-solution.charge(1), rel=1e-9)


def test_capacitance_converges_and_matches_kirchhoff(monkeypatch):
    d = 0.1 * L
    cap = Capacitor(separation=d)
    C = Solver(cap.plates(), []).solve([0.5, -0.5]).charge(0)
    monkeypatch.setattr(conductors, "PLATE_PANELS", 35)
    finer = Solver(cap.plates(), []).solve([0.5, -0.5]).charge(0)
    ideal = EPS0 * L * L / d
    assert C == pytest.approx(finer, rel=0.02)
    assert C > ideal
    # Kirchhoff's first-order fringing for thin discs, for a square of the same area,
    # with the edge term scaled by the perimeter ratio 4 / (2√π)
    a = L / math.sqrt(math.pi)
    kirchhoff = ideal + EPS0 * a * (math.log(16 * math.pi * a / d) - 1) * 4 / (2 * math.sqrt(math.pi))
    assert C == pytest.approx(kirchhoff, rel=0.05)


def test_charge_density_on_each_face(capacitor):
    _, solution = capacitor
    inner, outer = solution.faces[0]
    c = conductors.PLATE_PANELS // 2
    assert inner[c, c] == pytest.approx(EPS0 * 500 / 0.03, rel=0.01)  # σ = ε₀ E = ε₀ ΔV/d in the middle
    assert 0 < outer[c, c] < 0.5 * inner[c, c]  # little charge on the outer face...
    assert outer[0, c] > 3 * outer[c, c]  # ...most of it near the edges
    assert inner[0, c] > inner[c, c]
    mine = solution.panels.owner == 0
    np.testing.assert_allclose((inner + outer).ravel(), solution.sigma[mine], rtol=1e-9)


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


def test_gauss_law_around_each_plate(capacitor):
    _, solution = capacitor
    half = np.array([0.065, 0.065, 0.008])
    assert gauss_flux(solution, np.array([0, 0, 0.015]), half) == pytest.approx(solution.charge(0), rel=0.03)
    assert gauss_flux(solution, np.array([0, 0, -0.015]), half) == pytest.approx(solution.charge(1), rel=0.03)


def test_far_away_the_capacitor_is_a_dipole(capacitor):
    _, solution = capacitor
    far = solution.field(np.array([[0.0, 0.0, 5.0]]))[0]
    assert far[2] == pytest.approx(2 * K * solution.charge(0) * 0.03 / 5.0**3, rel=0.02)


def test_grounded_sphere_between_the_plates_is_neutral_but_charged_on_each_side():
    scene = conductor_scene("sphere_in_capacitor")
    plate_charge = scene.conductor.charge(0)
    assert abs(scene.sphere_charge(0)) < 1e-3 * plate_charge
    sigma = scene.conductor.sphere_sigma(0)
    assert sigma.min() < -0.5 * sigma.max() < 0  # negative towards the + plate, positive towards the -
    side = scene.conductor.panels.centers[scene.conductor.panels.owner == 2][:, 2]  # the plates are along z
    assert sigma[side > 0.005].mean() < 0 < sigma[side < -0.005].mean()


def test_clusters_match_the_panel_by_panel_sum(capacitor):
    _, solution = capacitor
    points = np.random.default_rng(2).uniform(-0.1, 0.1, (300, 3))
    V, E = evaluate(solution.panels, solution.sigma, points, clusters=solution.solver.clusters)
    V0, E0 = evaluate(solution.panels, solution.sigma, points)
    assert np.abs(V - V0).max() < 3e-3 * np.abs(V0).max()
    error = np.linalg.norm(E - E0, axis=1) / np.linalg.norm(E0, axis=1)
    assert np.median(error) < 0.01


def test_touching_spheres_and_plates():
    plates = Capacitor().plates()
    assert touching(plates, [Sphere([0, 0, 0.0], 0.0145)])  # reaches the plates at ±1.5 cm
    assert not touching(plates, [Sphere([0, 0, 0.0], 0.01)])
    assert touching([], [Sphere([0, 0, 0], 0.02), Sphere([0.03, 0, 0], 0.015)])
    assert not touching(plates, [Sphere([0.09, 0, 0], 0.02)])


@pytest.mark.parametrize("planar", [True, False])
@pytest.mark.parametrize("key", ["capacitor", "sphere_in_capacitor", "two_spheres"])
def test_every_line_ends_on_a_conductor(key, planar):
    scene = conductor_scene(key, planar)
    lines = scene.lines()
    assert lines
    ends = [line.end for line in lines]
    assert sum(end in ("plate", "sphere") for end in ends) >= len(ends) - 1  # at most one escapes past 100 m
    for line in lines:  # and every line starts on a conductor's surface
        start = line.points[0]
        on_plate = any(abs(p.to_local(start[None])[0, 2]) < 1e-9 for p in scene.conductor_plates())
        on_sphere = any(abs(np.linalg.norm(start - s.center) - s.radius) < 1e-9 for s in scene.spheres)
        assert on_plate or on_sphere
    if planar:
        assert all(np.abs(line.points[:, 2]).max() == 0.0 for line in lines)


def test_lines_bulge_out_at_the_plate_edges():
    scene = conductor_scene("capacitor", planar=True)
    cap = scene.capacitor
    inner = [line for line in scene.lines() if abs(line.points[1, 0]) < cap.separation / 2]
    outermost = max(inner, key=lambda line: abs(line.points[0, 1]))
    assert np.abs(outermost.points[:, 1]).max() > cap.width / 2


def test_plate_and_sphere_densities_for_drawing():
    scene = conductor_scene("sphere_in_capacitor")
    plus, minus = scene.plate_densities()
    assert plus.u[0] == pytest.approx(-scene.capacitor.width / 2)
    assert plus.u[-1] == pytest.approx(scene.capacitor.width / 2)
    assert plus.inner.shape == (len(plus.u), len(plus.v))
    np.testing.assert_allclose(minus.inner, -plus.inner, rtol=1e-4)
    ((points, quads, sigma),) = scene.sphere_densities()
    assert len(quads) == len(sigma) == 6 * conductors.SPHERE_PANELS**2
    scene.conductor = None  # being solved again: the plates plain, the spheres without σ (drawn grey)
    assert scene.plate_densities() is None
    ((_, quads, sigma),) = scene.sphere_densities()
    assert sigma is None and len(quads) == 6 * conductors.SPHERE_PANELS**2
