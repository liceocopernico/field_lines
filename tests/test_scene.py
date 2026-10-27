import numpy as np
import pytest

from field_lines import presets
from field_lines.scene import Capacitor, Scene
from field_lines.sources import Superposition
from field_lines.strings import LANGUAGES, STRINGS
from field_lines.units import CM


def test_presets_load_and_trace():
    from field_lines.conductors import Solver

    scene = Scene()
    for key in presets.PRESETS:
        presets.load(scene, key)
        if scene.uses_conductor:
            scene.conductor = Solver(scene.conductor_plates(), scene.spheres).solve(scene.voltages())
        lines = scene.lines()
        assert (len(lines) > 0) == (key != "empty"), key


def test_presets_choose_their_model():
    scene = Scene()
    presets.load(scene, "sphere")
    assert scene.kind == "conductors" and len(scene.spheres) == 1
    presets.load(scene, "capacitor")  # keeps the model that is on
    assert scene.kind == "conductors" and scene.spheres == [] and scene.capacitor is not None
    presets.load(scene, "dipole")
    assert scene.kind == "charges" and scene.capacitor is None


def test_going_2d_flattens_charges_and_turns_the_capacitor():
    scene = Scene()
    presets.load(scene, "quadrupole")
    scene.charges[0].position[2] = 0.04
    scene.capacitor = Capacitor(normal="z")
    scene.set_planar(True)
    assert all(c.position[2] == 0 for c in scene.charges)
    assert scene.capacitor.normal == "x"
    scene.set_planar(False)
    assert scene.capacitor.normal == "x"  # what was seen in the plane stays as it was


def test_click_seed_traces_both_ways_through_the_point():
    scene = Scene()
    scene.planar = True
    presets.load(scene, "dipole")
    before = len(scene.lines())
    point = np.array([0.0, 0.04, 0.0])
    scene.click_seeds.append(point)
    lines = scene.lines()
    assert len(lines) == before + 2
    halves = lines[-2:]
    for half in halves:
        np.testing.assert_allclose(half.points[0], point)
    # the two halves end near the two charges
    ends = sorted(half.points[-1][0] for half in halves)
    assert ends[0] < -0.02 and ends[1] > 0.02


def test_capacitor_plates_follow_the_axis():
    cap = Capacitor(separation=2 * CM, normal="x")
    plus, minus = cap.plates()
    assert plus.center[0] == 0.01 and minus.center[0] == -0.01
    assert plus.charge == -minus.charge > 0


def test_sigma_model_is_superposition_of_charges_and_plates():
    scene = Scene()
    presets.load(scene, "dipole")
    scene.capacitor = Capacitor()
    model = scene.model()
    assert isinstance(model, Superposition)
    assert len(model.sources) == 4


def test_every_string_has_every_language():
    for key, texts in STRINGS.items():
        assert set(texts) == set(LANGUAGES), key
        assert all(texts.values()), key


def test_lines_of_neutral_setups_close_and_of_charged_ones_go_to_infinity():
    scene = Scene()
    for key, end in (("dipole", "sink"), ("quadrupole", "sink"), ("single", "bounds")):
        presets.load(scene, key)
        assert {line.end for line in scene.lines()} == {end}, key


def test_off_screen_parts_of_a_line_take_no_growth_time():
    from field_lines.tracing import FieldLine
    from field_lines.view import shown_length

    points = np.array([[0.0, 0, 0], [0.05, 0, 0], [3.0, 0, 0], [3.0, 1.0, 0], [0.0, 0.05, 0], [0.0, 0.01, 0]])
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    shown = shown_length(FieldLine(points, s, np.ones(len(points)), "plate"))
    np.testing.assert_allclose(shown, [0, 0.05, 0.05, 0.05, 0.05, 0.09])


def test_plate_densities_of_the_sigma_model_are_uniform():
    scene = Scene()
    presets.load(scene, "capacitor")
    plus, minus = scene.plate_densities()
    assert plus.at(0.01, -0.02) == (scene.capacitor.sigma, scene.capacitor.sigma)
    assert minus.at(0.0, 0.0) == (-scene.capacitor.sigma, -scene.capacitor.sigma)
    presets.load(scene, "dipole")
    assert scene.plate_densities() is None


def test_capacitor_shapes():
    import math

    from field_lines.scene import capacitor_axis
    from field_lines.sources import Ring

    disc = Capacitor(shape="disc", radius=5 * CM, separation=3 * CM, thickness=0.002)
    assert disc.area == math.pi * (5 * CM) ** 2
    plus, minus = disc.bodies()
    assert isinstance(plus, Ring) and plus.r_in == 0 and plus.half_length == 0.001
    assert plus.center[2] == 1.5 * CM + 0.001 and plus.gap == "w-" and minus.gap == "w+"
    assert plus.charge == -minus.charge == disc.sigma * disc.area
    cylinder = Capacitor(shape="cylinder", inner_radius=2 * CM, outer_radius=4 * CM, length=10 * CM, thickness=0.002)
    inner, outer = cylinder.bodies()
    np.testing.assert_allclose([inner.r_in, inner.r_out, outer.r_in, outer.r_out], [0.018, 0.02, 0.04, 0.042])
    assert inner.charge == pytest.approx(-outer.charge)  # equal and opposite: σ a = -σ' b
    rod, _ = Capacitor(shape="cylinder", inner_radius=0.6 * CM, thickness=0.008).bodies()
    assert rod.r_in == 0.0  # thicker than its radius: a rod
    center, a, b = cylinder.probes()
    assert np.linalg.norm(center) == pytest.approx(3 * CM) and np.linalg.norm(b) == pytest.approx(4 * CM)
    E_ideal, C_ideal = cylinder.ideal(1e-9)
    assert C_ideal == pytest.approx(2 * math.pi * 8.8541878128e-12 * 0.1 / math.log(2), rel=1e-6)
    # in 2D the plates cross the plane; a cylinder is cut across its axis or along it
    assert capacitor_axis(Capacitor(shape="disc"), planar=True) == "x"
    assert capacitor_axis(Capacitor(shape="cylinder"), planar=True) == "z"
    assert capacitor_axis(Capacitor(shape="cylinder", cut="along"), planar=True) == "x"
    assert capacitor_axis(Capacitor(shape="cylinder", normal="x"), planar=False) == "x"


def test_cylinder_in_the_charges_model():
    scene = Scene()
    presets.load(scene, "cylindrical_capacitor")
    assert scene.kind == "charges" and scene.capacitor.shape == "cylinder"
    readouts = scene.capacitor_readouts()
    cap = scene.capacitor
    # halfway across the gap, in the middle of the tubes' length: nearly the field of infinite tubes
    assert readouts["E_center"] == pytest.approx(readouts["E_ideal"], rel=0.06)
    assert readouts["C"] == pytest.approx(readouts["C_ideal"], rel=0.06)
    assert readouts["Q"] == pytest.approx(cap.sigma * cap.area)
    scene.set_planar(True)
    assert scene.capacitor.normal == "z"  # cut across: two rings
    lines = scene.lines()
    assert lines and all(line.end == "plate" for line in lines)
    assert all(np.abs(line.points[:, 2]).max() == 0.0 for line in lines)
