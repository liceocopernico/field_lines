"""The PyVista side: actors for lines and plates, and the 2D/3D camera."""

from collections.abc import Sequence

import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.colors import ListedColormap

from field_lines.scene import VIEW_HALF
from field_lines.sources import Plate
from field_lines.strings import LANGUAGES, STRINGS, tr
from field_lines.tracing import FieldLine

BACKGROUND = "white"
POSITIVE_COLOR = (0.85, 0.12, 0.12)
NEGATIVE_COLOR = (0.12, 0.25, 0.85)
OUTLINE_COLOR = (0.8, 0.8, 0.8)
EQUIPOTENTIAL_COLOR = (0.1, 0.55, 0.25)
V_CMAP = "RdBu_r"  # red for positive potentials, blue for negative, like the charges
PLATE_THICKNESS = 0.0015  # m, only for drawing
# plasma without its palest yellow, which vanishes on white
LINE_CMAP = ListedColormap(colormaps["plasma"](np.linspace(0.0, 0.85, 256)))


def sign_color(value: float) -> tuple[float, float, float]:
    return POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR


def _percentile_range(values: np.ndarray) -> tuple[float, float]:
    """5th to 95th percentile of the positive values, for a log colour scale."""
    values = values[values > 0]
    if values.size == 0:
        return (1.0, 10.0)
    lo, hi = np.percentile(values, [5, 95])
    return (float(lo), float(max(hi, 1.01 * lo)))


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
        self.slice_actor = None
        self.lines_visible = True
        self.pl.set_background(BACKGROUND)
        self.pl.enable_depth_peeling()  # translucent equipotentials drawn in the right order
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
                scalar_bar_args=dict(title=tr("bar_E"), vertical=True, position_x=0.02, position_y=0.25, fmt="%.2g"),
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
        if not self.lines:
            return (1.0, 10.0)
        return _percentile_range(np.concatenate([line.strength for line in self.lines]))

    def set_equipotentials(self, mesh: pv.PolyData | None, planar: bool) -> None:
        """Equipotential lines (2D) or translucent surfaces (3D); None removes them."""
        self.pl.remove_actor("equipotentials", render=False)
        if mesh is None or mesh.n_points == 0:
            return
        style = dict(line_width=2) if planar else dict(opacity=0.15, smooth_shading=True)
        self.pl.add_mesh(
            mesh, name="equipotentials", color=EQUIPOTENTIAL_COLOR, pickable=False, reset_camera=False, **style
        )

    def set_arrows(self, mesh: pv.PolyData | None) -> None:
        """Arrows coloured by |E| on the same scale as the lines; None removes them."""
        self.pl.remove_actor("arrows", render=False)
        if mesh is None or mesh.n_points == 0:
            return
        self.pl.add_mesh(
            mesh,
            name="arrows",
            scalars="E",
            cmap=LINE_CMAP,
            log_scale=True,
            clim=self._clim() if self.lines else _percentile_range(mesh["E"]),
            show_scalar_bar=False,
            pickable=False,
            reset_camera=False,
        )

    def set_slice(self, mesh: pv.ImageData | None, quantity: str) -> None:
        """The coloured plane, by V (red +, blue -) or by |E| (log scale); None removes it."""
        self.pl.remove_actor("slice", render=False)
        self.slice_actor = None
        for key in ("bar_V", "bar_slice_E"):  # removing the actor leaves its colour bar behind
            for language in LANGUAGES:
                title = STRINGS[key][language]
                if title in self.pl.scalar_bars:
                    self.pl.remove_scalar_bar(title, render=False)
        if mesh is None:
            return
        bar = dict(vertical=True, position_x=0.88, position_y=0.25, fmt="%.2g")
        if quantity == "V":
            limit = float(np.percentile(np.abs(mesh["V"]), 95)) or 1.0
            style = dict(scalars="V", cmap=V_CMAP, clim=(-limit, limit), scalar_bar_args=bar | dict(title=tr("bar_V")))
        else:
            style = dict(
                scalars="E",
                cmap="viridis",
                log_scale=True,
                clim=_percentile_range(mesh["E"]),
                scalar_bar_args=bar | dict(title=tr("bar_slice_E")),
            )
        self.slice_actor = self.pl.add_mesh(
            mesh, name="slice", lighting=False, pickable=True, reset_camera=False, **style
        )

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
