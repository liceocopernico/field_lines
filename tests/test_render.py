import numpy as np
import pytest

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
    view.set_plates(Capacitor(normal="x").plates())
    plotter.render()
    image = plotter.screenshot(return_img=True).astype(int)
    h, w = image.shape[:2]
    # plates at x = ±1.5 cm; the view spans ±10.8 cm
    right = image[h // 2, int(w / 2 + 1.5 / 10.8 * w / 2)]
    left = image[h // 2, int(w / 2 - 1.5 / 10.8 * w / 2)]
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
