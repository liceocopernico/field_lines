import numpy as np
import pytest
import pyvista as pv

from field_lines import presets
from field_lines.scene import Capacitor, Scene
from field_lines.view import SceneView, truncated_lines

pytestmark = pytest.mark.render


def coloured_pixels(pl):
    """Number of clearly coloured (not grey, not white) pixels in the rendered image."""
    pl.render()
    image = pl.screenshot(return_img=True).astype(int)
    spread = image.max(axis=2) - image.min(axis=2)
    return int((spread > 60).sum())


def test_lines_appear_and_grow(plotter):
    scene = Scene()
    scene.planar = True
    presets.load(scene, "dipole")
    view = SceneView(plotter)
    view.set_planar(True)
    view.set_lines(scene.lines())
    view.show_lines(0.0)
    nothing = coloured_pixels(plotter)
    view.show_lines(0.02)
    some = coloured_pixels(plotter)
    view.show_lines()
    everything = coloured_pixels(plotter)
    assert nothing < some < everything


def test_plates_are_drawn_red_and_blue(plotter):
    view = SceneView(plotter)
    view.set_planar(True)
    view.set_plates(Capacitor(normal="x").slabs())
    plotter.render()
    image = plotter.screenshot(return_img=True).astype(int)
    h, w = image.shape[:2]
    # slabs from x = ±1.5 to ±1.7 cm; the view spans ±10.8 cm
    right = image[h // 2, int(w / 2 + 1.6 / 10.8 * w / 2)]
    left = image[h // 2, int(w / 2 - 1.6 / 10.8 * w / 2)]
    assert right[0] > right[2] + 60  # positive: red
    assert left[2] > left[0] + 60  # negative: blue


def test_truncated_lines_end_exactly_at_the_limit():
    scene = Scene()
    presets.load(scene, "single")
    lines = scene.lines()
    mesh = truncated_lines(lines, np.full(len(lines), 0.03))
    starts = np.array([line.points[0] for line in lines])
    distances = np.linalg.norm(mesh.points[:, None, :] - starts[None], axis=2).min(axis=1)
    assert distances.max() == pytest.approx(0.03, rel=1e-3)  # radial lines: arc length = distance


def test_translucent_equipotential_surfaces_render_in_3d(plotter):
    from field_lines import sampling
    from field_lines.sampling import Exclusion

    scene = Scene()
    presets.load(scene, "dipole")
    view = SceneView(plotter)
    view.set_planar(False)
    exclusions = [Exclusion(c.position, 0.005) for c in scene.charges]
    mesh, levels = sampling.equipotentials(scene.model(), planar=False, count=5, exclusions=exclusions)
    before = plotter.screenshot(return_img=True).astype(int)
    view.set_equipotentials(mesh, planar=False)
    plotter.render()
    after = plotter.screenshot(return_img=True).astype(int)
    assert "equipotentials" in plotter.actors
    assert (np.abs(after - before).sum(axis=2) > 30).sum() > 1000  # the surfaces tint the view


def test_plates_coloured_by_density_in_2d(plotter):
    from field_lines.strings import tr

    scene = Scene()
    scene.planar = True
    presets.load(scene, "capacitor")
    view = SceneView(plotter)
    view.set_planar(True)
    view.set_plates(scene.drawn_plates(), scene.plate_densities(), planar=True)
    assert len(view.plate_actors) == 8  # each slab's cut: an inner and an outer half, two ends
    assert tr("bar_sigma") in plotter.scalar_bars
    plotter.render()
    image = plotter.screenshot(return_img=True).astype(int)
    h, w = image.shape[:2]
    right = image[h // 2, int(w / 2 + 1.6 / 10.8 * w / 2)]  # the middle of each slab, 1.5 to 1.7 cm out
    left = image[h // 2, int(w / 2 - 1.6 / 10.8 * w / 2)]
    assert right[0] > right[2] + 60 and left[2] > left[0] + 60  # + red, - blue
    view.set_plates(scene.drawn_plates(), None, planar=True)  # back to plain plates: the colour bar goes too
    assert len(view.plate_actors) == 2
    assert tr("bar_sigma") not in plotter.scalar_bars


def test_plates_coloured_by_density_in_3d(plotter):
    from field_lines.view import slab_faces

    scene = Scene()
    presets.load(scene, "capacitor")
    plus = scene.plate_densities()[0]
    inner, outer, *rims = slab_faces(plus)
    # the + plate is above z = +1.5 cm: its inner face is the lower one, a thickness below the outer one
    assert inner.points[:, 2] == pytest.approx(0.015)
    assert outer.points[:, 2] == pytest.approx(0.017)
    assert len(rims) == 4
    for rim in rims:  # the rims close the slab
        assert rim.points[:, 2].min() == pytest.approx(0.015) and rim.points[:, 2].max() == pytest.approx(0.017)
    surface = pv.merge([inner, outer, *rims]).extract_surface(algorithm="dataset_surface").clean()
    assert surface.n_open_edges == 0
    for face in (inner, outer, *rims):
        assert np.allclose(face["sigma"], 0.2)  # μC/m²


def test_lines_are_drawn_only_inside_the_drawing_box():
    from field_lines.view import DRAW_HALF, _inside_box

    # out of the box along +x, around, and back in along -x
    points = np.array([[0.0, 0, 0], [0.3, 0, 0], [5.0, 0, 0], [5.0, 4.0, 0], [-5.0, 4.0, 0], [-0.2, 0.1, 0], [-0.1, 0, 0]])
    values = np.arange(len(points), dtype=float)
    pieces = _inside_box(points, values, DRAW_HALF)
    assert len(pieces) == 2
    (out_piece, out_values), (back_piece, _) = pieces
    np.testing.assert_allclose(out_piece[-1], [DRAW_HALF, 0, 0])  # leaves exactly on the box
    assert out_values[-1] == pytest.approx(1 + (DRAW_HALF - 0.3) / 4.7)
    assert np.abs(back_piece[0]).max() == pytest.approx(DRAW_HALF)  # comes back in on the box
    np.testing.assert_allclose(back_piece[-1], [-0.1, 0, 0])
    inside = np.array([[0.0, 0, 0], [0.1, 0.1, 0.1]])
    assert len(_inside_box(inside, np.zeros(2), DRAW_HALF)) == 1


def test_sigma_colour_scales():
    from field_lines.view import sigma_colours

    values = np.concatenate((np.linspace(0.95, 1.05, 98), [-3.0, 4.0]))  # μC/m²: small differences, two spikes
    symmetric = sigma_colours(values, "symmetric")
    assert symmetric.clim[0] == -symmetric.clim[1] > -4.0  # centred on 0, spikes left out
    contrast = sigma_colours(values, "contrast")
    assert 0.9 < contrast.clim[0] < 1.0 and 1.0 < contrast.clim[1] < 1.1  # only the bulk: small differences show
    log = sigma_colours(values, "log")
    assert 0 < log.clim[0] < log.clim[1] == 4.0
    uniform = sigma_colours(np.full(10, 0.2), "contrast")  # one value: still a valid range around it
    assert uniform.clim[0] < 0.2 < uniform.clim[1]
    almost = sigma_colours(0.4466 + 0.0009 * np.random.default_rng(0).random(100), "contrast")
    assert almost.clim[1] - almost.clim[0] >= 0.02 * 0.4466  # finer than the solver's error: not stretched


def test_contrast_scale_on_a_sphere_between_plates(plotter):
    from field_lines.conductors import Solver
    from field_lines.strings import tr

    scene = Scene()
    scene.kind = "conductors"
    presets.load(scene, "sphere_in_capacitor")
    scene.conductor = Solver(scene.conductor_plates(), scene.spheres).solve(scene.voltages())
    view = SceneView(plotter)
    for scale, title in (("symmetric", "bar_sigma"), ("contrast", "bar_sigma"), ("log", "bar_sigma_abs")):
        spheres = scene.sphere_densities()
        view.set_plates(scene.drawn_plates(), scene.plate_densities(), False, spheres, scale, "viridis")
        assert list(plotter.scalar_bars.keys()) == [tr(title)]
        lo, hi = view.sphere_actors[0][0].mapper.scalar_range
        if scale == "log":
            assert 0 < lo < hi
        else:
            assert lo < 0 < hi  # the grounded sphere is charged both ways


def test_sigma_colour_bar_labels_are_readable():
    from field_lines.view import SigmaColours

    assert SigmaColours("contrast", "viridis", (-0.91, 0.91)).label_format == "%.2f"  # 0 prints as 0.00
    assert SigmaColours("contrast", "viridis", (0.4462, 0.4483)).label_format == "%.5f"  # tiny ranges keep digits
    assert SigmaColours("log", "viridis", (0.03, 2.3)).label_format == "%.2g"
