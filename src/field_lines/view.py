"""The PyVista side: actors for lines and plates, and the 2D/3D camera."""

from collections.abc import Sequence

import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.colors import ListedColormap

from field_lines.scene import VIEW_HALF, PlateDensity
from field_lines.sources import AXES, Plate
from field_lines.strings import LANGUAGES, STRINGS, tr
from field_lines.tracing import FieldLine
from field_lines.units import UC_PER_M2

BACKGROUND = "white"
POSITIVE_COLOR = (0.85, 0.12, 0.12)
NEGATIVE_COLOR = (0.12, 0.25, 0.85)
OUTLINE_COLOR = (0.8, 0.8, 0.8)
EQUIPOTENTIAL_COLOR = (0.1, 0.55, 0.25)
V_CMAP = "RdBu_r"  # red for positive potentials, blue for negative, like the charges
PLATE_THICKNESS = 0.0015  # m, only for drawing
FACE_STRIP = 0.002  # m: in 2D each face of a cut plate is drawn as a strip this wide
SIGMA_BAR = dict(vertical=False, position_x=0.3, position_y=0.02, width=0.4, height=0.1, fmt="%.2g")
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


SHOWN_HALF = 1.5 * VIEW_HALF  # line pieces beyond this are off screen


def shown_length(line: FieldLine) -> np.ndarray:
    """Arc length along the line counting only what lies near the view (so off-screen loops take no time)."""
    middles = (line.points[1:] + line.points[:-1]) / 2
    near = np.all(np.abs(middles) <= SHOWN_HALF, axis=1)
    return np.concatenate(([0.0], np.cumsum(np.diff(line.s) * near)))


def truncated_lines(lines: Sequence[FieldLine], limits: np.ndarray, lengths: Sequence[np.ndarray] | None = None):
    """All lines in one PolyData, each cut at limits[i] (inf: whole line) along lengths[i] (default: s)."""
    points, strengths, cells = [], [], []
    offset = 0
    lengths = [line.s for line in lines] if lengths is None else lengths
    for line, limit, s in zip(lines, limits, lengths):
        k = int(np.searchsorted(s, limit, side="right"))
        pts, mags = line.points[:k], line.strength[:k]
        if 0 < k < len(s):  # end exactly at the limit, between two samples
            t = (limit - s[k - 1]) / (s[k] - s[k - 1])
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


def _inner_side(plate: Plate) -> float:
    """-1 if the plate's inner face (towards the other plate) is on its local -w side, else +1."""
    return -1.0 if plate.center[AXES.index(plate.normal)] > 0 else 1.0


def _sheet(plate: Plate, local: np.ndarray, sigma: np.ndarray) -> pv.StructuredGrid:
    """A surface through local points of shape (nu, nv, 3), with σ (C/m²) shown in μC/m²."""
    points = plate.from_local(local.reshape(-1, 3)) + plate.center
    x, y, z = (points[:, k].reshape(local.shape[:2]) for k in range(3))
    mesh = pv.StructuredGrid(x, y, z)
    mesh.point_data["sigma"] = sigma.ravel(order="F") / UC_PER_M2
    return mesh


def face_sheets(density: PlateDensity) -> list[pv.StructuredGrid]:
    """3D: the plate's two faces, a little apart, each coloured by its own σ."""
    uu, vv = np.meshgrid(density.u, density.v, indexing="ij")
    inner = _inner_side(density.plate)
    sheets = []
    for side, sigma in ((inner, density.inner), (-inner, density.outer)):
        local = np.stack((uu, vv, np.full(uu.shape, side * PLATE_THICKNESS / 2)), axis=-1)
        sheets.append(_sheet(density.plate, local, sigma))
    return sheets


def face_strips(density: PlateDensity) -> list[pv.StructuredGrid]:
    """2D: where the plate crosses the z = 0 plane, a strip on each side of it coloured by that face's σ."""
    plate = density.plate
    z_axis = plate.local_axes.index(2)
    if z_axis == 2:
        return []  # a plate lying in the plane is not cut by it
    cut = -plate.center[2]
    if z_axis == 1:  # the plate runs along u in the plane
        along = density.u
        j = int(np.argmin(np.abs(density.v - cut)))
        faces = density.inner[:, j], density.outer[:, j]
    else:
        along = density.v
        i = int(np.argmin(np.abs(density.u - cut)))
        faces = density.inner[i, :], density.outer[i, :]
    inner = _inner_side(plate)
    strips = []
    for side, sigma in ((inner, faces[0]), (-inner, faces[1])):
        across = np.array([0.0, side * FACE_STRIP])
        a, w = np.meshgrid(along, across, indexing="ij")
        c = np.full(a.shape, cut)
        local = np.stack((a, c, w) if z_axis == 1 else (c, a, w), axis=-1)
        strips.append(_sheet(plate, local, np.repeat(sigma[:, None], 2, axis=1)))
    return strips


class SceneView:
    def __init__(self, plotter: pv.Plotter):
        self.pl = plotter
        self.lines: list[FieldLine] = []
        self.lengths: list[np.ndarray] = []  # shown_length of each line, which the growth follows
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
        """How far the growth has to go for every line to be complete."""
        return max((s[-1] for s in self.lengths), default=0.0)

    def set_lines(self, lines: list[FieldLine]) -> None:
        self.lines = lines
        self.lengths = [shown_length(line) for line in lines]

    def show_lines(self, limits: np.ndarray | float = np.inf) -> None:
        """Draw the lines, each up to its arc length limit."""
        limits = np.broadcast_to(np.asarray(limits, dtype=float), (len(self.lines),))
        mesh = truncated_lines(self.lines, limits, self.lengths)
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
        """The colour range of |E| over the parts of the lines that can be seen."""
        if not self.lines:
            return (1.0, 10.0)
        strengths = np.concatenate([line.strength for line in self.lines])
        points = np.concatenate([line.points for line in self.lines])
        near = np.all(np.abs(points) <= SHOWN_HALF, axis=1)
        return _percentile_range(strengths[near] if near.any() else strengths)

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
        bar = dict(vertical=True, position_x=0.82, position_y=0.25, fmt="%.2g")
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

    def set_plates(
        self, plates: Sequence[Plate], densities: Sequence[PlateDensity] | None = None, planar: bool = False
    ) -> None:
        """The plates: plain red and blue, or each face coloured by its surface charge density."""
        for actor in self.plate_actors:
            self.pl.remove_actor(actor, render=False)
        self.plate_actors = []
        for language in LANGUAGES:  # removing the actors leaves their colour bar behind
            if STRINGS["bar_sigma"][language] in self.pl.scalar_bars:
                self.pl.remove_scalar_bar(STRINGS["bar_sigma"][language], render=False)
        if densities:
            self._coloured_plates(densities, planar)
            return
        for plate in plates:
            half = np.array([plate.width / 2, plate.height / 2, PLATE_THICKNESS / 2])
            lo = plate.from_local(-half[None])[0] + plate.center
            hi = plate.from_local(half[None])[0] + plate.center
            bounds = np.column_stack((lo, hi)).ravel()
            actor = self.pl.add_mesh(
                pv.Box(bounds), color=sign_color(plate.sigma), opacity=0.6, pickable=False, reset_camera=False
            )
            self.plate_actors.append(actor)

    def _coloured_plates(self, densities: Sequence[PlateDensity], planar: bool) -> None:
        values = np.concatenate([np.concatenate((d.inner.ravel(), d.outer.ravel())) for d in densities])
        limit = float(np.percentile(np.abs(values), 95)) / UC_PER_M2 or 1.0  # the edges may saturate
        for density in densities:
            meshes = face_strips(density) if planar else face_sheets(density)
            for mesh in meshes:
                actor = self.pl.add_mesh(
                    mesh,
                    scalars="sigma",
                    cmap=V_CMAP,
                    clim=(-limit, limit),
                    lighting=not planar,
                    pickable=False,
                    reset_camera=False,
                    scalar_bar_args=SIGMA_BAR | dict(title=tr("bar_sigma")),
                )
                self.plate_actors.append(actor)
        # the labels below the bar, so that they do not run into the title above it
        self.pl.scalar_bars[tr("bar_sigma")].SetTextPositionToPrecedeScalarBar()

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
        # mark the camera as placed: otherwise PyVista resets it the next time plotter.camera is read
        self.pl.renderer.camera.is_set = True
        self.pl.render()
