"""The PyVista side: actors for lines and plates, and the 2D/3D camera."""

from collections.abc import Sequence

import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.colors import ListedColormap

from field_lines.scene import VIEW_HALF
from field_lines.sources import Plate
from field_lines.strings import tr
from field_lines.tracing import FieldLine

BACKGROUND = "white"
POSITIVE_COLOR = (0.85, 0.12, 0.12)
NEGATIVE_COLOR = (0.12, 0.25, 0.85)
OUTLINE_COLOR = (0.8, 0.8, 0.8)
PLATE_THICKNESS = 0.0015  # m, only for drawing
# plasma without its palest yellow, which vanishes on white
LINE_CMAP = ListedColormap(colormaps["plasma"](np.linspace(0.0, 0.85, 256)))


def sign_color(value: float) -> tuple[float, float, float]:
    return POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR


def truncated_lines(lines: Sequence[FieldLine], limits: np.ndarray) -> pv.PolyData:
    """All lines in one PolyData, each cut at arc length limits[i] (inf: whole line)."""
    points, strengths, cells = [], [], []
    offset = 0
    for line, limit in zip(lines, limits):
        k = int(np.searchsorted(line.s, limit, side="right"))
        pts, mags = line.points[:k], line.strength[:k]
        if 0 < k < len(line.s):  # end exactly at the limit, between two samples
            t = (limit - line.s[k - 1]) / (line.s[k] - line.s[k - 1])
            pts = np.vstack((pts, line.points[k - 1] + t * (line.points[k] - line.points[k - 1])))
            mags = np.append(mags, line.strength[k - 1] + t * (line.strength[k] - line.strength[k - 1]))
        if len(pts) < 2:
            continue
        points.append(pts)
        strengths.append(mags)
        cells.append(np.concatenate(([len(pts)], np.arange(offset, offset + len(pts)))))
        offset += len(pts)
    if not points:
        return pv.PolyData()
    mesh = pv.PolyData(np.concatenate(points), lines=np.concatenate(cells))
    mesh.point_data["E"] = np.maximum(np.concatenate(strengths), 1e-30)
    return mesh


class SceneView:
    def __init__(self, plotter: pv.Plotter):
        self.pl = plotter
        self.lines: list[FieldLine] = []
        self.line_mesh = pv.PolyData()
        self.line_actor = None
        self.plate_actors: list = []
        self.lines_visible = True
        self.pl.set_background(BACKGROUND)
        box = pv.Box((-VIEW_HALF, VIEW_HALF) * 3)
        self.pl.add_mesh(box.outline(), color=OUTLINE_COLOR, line_width=1, pickable=False, reset_camera=False)
        self.pl.add_axes(interactive=False)

    @property
    def max_length(self) -> float:
        return max((line.length for line in self.lines), default=0.0)

    def set_lines(self, lines: list[FieldLine]) -> None:
        self.lines = lines

    def show_lines(self, limits: np.ndarray | float = np.inf) -> None:
        """Draw the lines, each up to its arc length limit."""
        limits = np.broadcast_to(np.asarray(limits, dtype=float), (len(self.lines),))
        mesh = truncated_lines(self.lines, limits)
        if mesh.n_points == 0:
            if self.line_actor is not None:
                self.line_actor.visibility = False
            return
        clim = self._clim()
        if self.line_actor is None:
            self.line_mesh = mesh
            self.line_actor = self.pl.add_mesh(
                self.line_mesh,
                scalars="E",
                cmap=LINE_CMAP,
                log_scale=True,
                clim=clim,
                render_lines_as_tubes=True,
                line_width=4,
                pickable=False,
                reset_camera=False,
                scalar_bar_args=dict(title=tr("bar_E"), vertical=True, position_x=0.02, position_y=0.25, fmt="%.0e"),
            )
        else:
            self.line_mesh.copy_from(mesh)
            self.line_actor.mapper.scalar_range = clim
        self.line_actor.visibility = self.lines_visible

    def set_lines_visible(self, visible: bool) -> None:
        self.lines_visible = visible
        if self.line_actor is not None:
            self.line_actor.visibility = visible and self.line_mesh.n_points > 0

    def _clim(self) -> tuple[float, float]:
        mags = np.concatenate([line.strength for line in self.lines]) if self.lines else np.ones(1)
        mags = mags[mags > 0]
        if mags.size == 0:
            return (1.0, 10.0)
        lo, hi = np.percentile(mags, [5, 95])
        return (lo, max(hi, 1.01 * lo))

    def set_plates(self, plates: Sequence[Plate]) -> None:
        for actor in self.plate_actors:
            self.pl.remove_actor(actor, render=False)
        self.plate_actors = []
        for plate in plates:
            half = np.array([plate.width / 2, plate.height / 2, PLATE_THICKNESS / 2])
            lo = plate.from_local(-half[None])[0] + plate.center
            hi = plate.from_local(half[None])[0] + plate.center
            bounds = np.column_stack((lo, hi)).ravel()
            actor = self.pl.add_mesh(
                pv.Box(bounds), color=sign_color(plate.sigma), opacity=0.6, pickable=False, reset_camera=False
            )
            self.plate_actors.append(actor)

    def set_planar(self, planar: bool) -> None:
        """2D: look down the z axis with a parallel camera that pans instead of rotating."""
        if planar:
            self.pl.enable_parallel_projection()
            self.pl.view_xy()
            self.pl.camera.parallel_scale = 1.08 * VIEW_HALF
            self.pl.enable_custom_trackball_style(
                left="pan", shift_left="pan", control_left="pan", middle="pan", right="dolly"
            )
        else:
            self.pl.disable_parallel_projection()
            self.pl.enable_trackball_style()
            self.pl.view_isometric()
            self.pl.reset_camera(bounds=(-VIEW_HALF, VIEW_HALF) * 3)
        self.pl.render()
