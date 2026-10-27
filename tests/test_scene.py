import numpy as np

from field_lines import presets
from field_lines.scene import Capacitor, Scene
from field_lines.sources import Superposition
from field_lines.strings import LANGUAGES, STRINGS
from field_lines.units import CM


def test_presets_load_and_trace():
    scene = Scene()
    for key in presets.PRESETS:
        presets.load(scene, key)
        lines = scene.lines()
        assert (len(lines) > 0) == (key != "empty")


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
