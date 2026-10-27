"""The PyVista side: actors for lines and plates, and the 2D/3D camera."""

from collections.abc import Sequence
from dataclasses import dataclass

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
FACE_STRIP = 0.002  # m: in 2D each face of a cut plate is drawn as a strip this wide, a sphere's rim as a ring
PLANE_BELOW = 0.0003  # m: in 2D a sphere's inside is drawn just below the lines
SPHERE_INSIDE = (0.85, 0.85, 0.85)
SIGMA_BAR = dict(vertical=False, position_x=0.3, position_y=0.02, width=0.4, height=0.1)
SIGMA_SCALES = ("symmetric", "contrast", "log")
SIGMA_PALETTES = ("RdBu_r", "viridis", "plasma", "turbo", "cividis", "Greys")
# The contrast scale never spreads its colours over less than this fraction of the typical |σ|: the
# conductors are solved to about 0.2-0.5%, and stretching finer differences would show the panels.
CONTRAST_FLOOR = 0.02
# plasma without its palest yellow, which vanishes on white
LINE_CMAP = ListedColormap(colormaps["plasma"](np.linspace(0.0, 0.85, 256)))


@dataclass
class SigmaColours:
    """How σ (in μC/m²) maps to colours: the scale, the palette and the range."""

    scale: str  # "symmetric": centred on 0, shows the sign; "contrast": the values' own range; "log": |σ|
    palette: str
    clim: tuple[float, float]

    @property
    def title(self) -> str:
        return tr("bar_sigma_abs" if self.scale == "log" else "bar_sigma")

    @property
    def label_format(self) -> str:
        """Fixed decimals enough to tell the labels apart (a %g format would print 0 as -2.7e-15)."""
        if self.scale == "log":
            return "%.2g"
        span = self.clim[1] - self.clim[0]
        decimals = int(np.clip(2 - np.floor(np.log10(span)), 0, 8)) if span > 0 else 2
        return f"%.{decimals}f"


def sigma_colours(values: np.ndarray, scale: str = "symmetric", palette: str = "RdBu_r") -> SigmaColours:
    """The colour range for these σ values (μC/m²).

    symmetric: ±(95th percentile of |σ|), so a few sharp edges do not wash out the rest;
    contrast: 2nd to 98th percentile, so that small differences spread over the whole palette
    (but at least CONTRAST_FLOOR of the typical |σ|);
    log: |σ| from its 2nd percentile to its largest value.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        values = np.ones(1)
    if scale == "log":
        magnitude = np.abs(values[values != 0]) if np.any(values != 0) else np.ones(1)
        lo, hi = float(np.percentile(magnitude, 2)), float(magnitude.max())
        return SigmaColours(scale, palette, (lo, max(hi, 1.01 * lo)))
    if scale == "contrast":
        lo, hi = (float(v) for v in np.percentile(values, [2, 98]))
        floor = CONTRAST_FLOOR * (float(np.median(np.abs(values))) or 1e-6)
        if hi - lo < floor:  # (nearly) one value everywhere: a range just wider than the solver's error
            middle = (lo + hi) / 2
            lo, hi = middle - floor / 2, middle + floor / 2
        return SigmaColours(scale, palette, (lo, hi))
    limit = float(np.percentile(np.abs(values), 95)) or 1.0
    return SigmaColours(scale, palette, (-limit, limit))


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
# Lines are traced up to 100 m away, but drawn only within this box: VTK sizes the camera's depth
# range to everything drawn, and with lines 100 m long its picking tolerance grows so large that a
# metal sphere's handle catches every click and the camera no longer rotates.
DRAW_HALF = 0.5  # m


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
        for piece, piece_mags in _inside_box(pts, mags, DRAW_HALF):
            points.append(piece)
            strengths.append(piece_mags)
            cells.append(np.concatenate(([len(piece)], np.arange(offset, offset + len(piece)))))
            offset += len(piece)
    if not points:
        return pv.PolyData()
    mesh = pv.PolyData(np.concatenate(points), lines=np.concatenate(cells))
    mesh.point_data["E"] = np.maximum(np.concatenate(strengths), 1e-30)
    return mesh


def _inside_box(points: np.ndarray, values: np.ndarray, half: float):
    """The pieces of a polyline (and of values along it) inside the box |x|, |y|, |z| <= half.

    A piece that leaves or enters the box ends exactly on its surface.
    """
    if len(points) < 2:
        return []
    inside = np.all(np.abs(points) <= half, axis=1)
    if inside.all():
        return [(points, values)]
    pieces = []
    start = None  # where the current piece begins: (index, boundary point or None, value)
    for i in range(len(points)):
        if inside[i] and start is None:
            start = (i, None, None) if i == 0 or inside[i - 1] else (i, *_crossing(points, values, i, i - 1, half))
        if start is not None and (i == len(points) - 1 or not inside[i + 1]):
            first, entry, entry_value = start
            piece, piece_values = [points[first : i + 1]], [values[first : i + 1]]
            if entry is not None:
                piece.insert(0, entry[None])
                piece_values.insert(0, [entry_value])
            if i < len(points) - 1:
                exit_point, exit_value = _crossing(points, values, i, i + 1, half)
                piece.append(exit_point[None])
                piece_values.append([exit_value])
            piece = np.concatenate(piece)
            if len(piece) >= 2:
                pieces.append((piece, np.concatenate(piece_values)))
            start = None
    return pieces


def _crossing(points: np.ndarray, values: np.ndarray, inner: int, outer: int, half: float):
    """Where the segment from points[inner] (inside the box) to points[outer] (outside) meets its surface."""
    a, b = points[inner], points[outer]
    d = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d != 0, (np.sign(d) * half - a) / d, np.inf)
    t = float(np.clip(t[t >= 0].min(initial=1.0), 0.0, 1.0))
    return a + t * d, values[inner] + t * (values[outer] - values[inner])


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


def sphere_ring(points: np.ndarray, quads: np.ndarray, sigma: np.ndarray, samples: int = 144) -> pv.StructuredGrid:
    """2D: the sphere's rim where it crosses its centre's z plane, a ring just inside it coloured by σ there."""
    center = points.mean(axis=0)
    radius = np.linalg.norm(points[0] - center)
    angle = np.linspace(0, 2 * np.pi, samples + 1)
    rim = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(samples + 1)))
    panel_dirs = points[quads].mean(axis=1) - center
    panel_dirs /= np.linalg.norm(panel_dirs, axis=1, keepdims=True)
    nearest = np.argmax(rim @ panel_dirs.T, axis=1)  # the panel each rim point lies on
    r = np.array([radius - FACE_STRIP, radius])
    x = center[0] + np.outer(rim[:, 0], r)
    y = center[1] + np.outer(rim[:, 1], r)
    ring = pv.StructuredGrid(x, y, np.full(x.shape, center[2]))
    ring.point_data["sigma"] = np.repeat(sigma[nearest][:, None], 2, axis=1).ravel(order="F") / UC_PER_M2
    return ring


class SceneView:
    def __init__(self, plotter: pv.Plotter):
        self.pl = plotter
        self.lines: list[FieldLine] = []
        self.lengths: list[np.ndarray] = []  # shown_length of each line, which the growth follows
        self.line_mesh = pv.PolyData()
        self.line_actor = None
        self.plate_actors: list = []
        self.sphere_actors: list[list] = []  # the actors of each metallic sphere
        self.sphere_centers: list[np.ndarray] = []  # where each sphere was drawn
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
        self,
        plates: Sequence[Plate],
        densities: Sequence[PlateDensity] | None = None,
        planar: bool = False,
        spheres: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        scale: str = "symmetric",
        palette: str = "RdBu_r",
    ) -> None:
        """The plates (plain red and blue, or each face coloured by its σ) and the metallic spheres (by σ).

        spheres: for each sphere its corner points, quads and σ per quad, as Scene.sphere_densities gives.
        scale, palette: how σ maps to colours (see sigma_colours), one range for all of them.
        """
        for actor in self.plate_actors + [a for actors in self.sphere_actors for a in actors]:
            self.pl.remove_actor(actor, render=False)
        self.plate_actors, self.sphere_actors, self.sphere_centers = [], [], []
        for key in ("bar_sigma", "bar_sigma_abs"):  # removing the actors leaves their colour bar behind
            for language in LANGUAGES:
                if STRINGS[key][language] in self.pl.scalar_bars:
                    self.pl.remove_scalar_bar(STRINGS[key][language], render=False)
        values = [np.concatenate((d.inner.ravel(), d.outer.ravel())) for d in densities or []]
        values += [sigma for _, _, sigma in spheres or [] if sigma is not None]
        colours = sigma_colours(np.concatenate(values) / UC_PER_M2 if values else np.zeros(0), scale, palette)
        if densities:
            self._coloured_plates(densities, planar, colours)
        for points, quads, sigma in spheres or []:
            self._coloured_sphere(points, quads, sigma, planar, colours)
        if colours.title in self.pl.scalar_bars:
            self.pl.scalar_bars[colours.title].SetTextPositionToPrecedeScalarBar()  # labels clear of the title
        if densities:
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

    def _sigma_mesh(self, mesh, colours: SigmaColours, planar: bool, **style):
        scalars = "sigma"
        if colours.scale == "log":
            mesh.point_data["abs_sigma"] = np.maximum(np.abs(mesh["sigma"]), 1e-3 * colours.clim[0])
            scalars = "abs_sigma"
        return self.pl.add_mesh(
            mesh,
            scalars=scalars,
            cmap=colours.palette,
            clim=colours.clim,
            log_scale=colours.scale == "log",
            lighting=not planar,
            pickable=False,
            reset_camera=False,
            scalar_bar_args=SIGMA_BAR | dict(title=colours.title, fmt=colours.label_format),
            **style,
        )

    def _coloured_plates(self, densities: Sequence[PlateDensity], planar: bool, colours: SigmaColours) -> None:
        for density in densities:
            for mesh in face_strips(density) if planar else face_sheets(density):
                self.plate_actors.append(self._sigma_mesh(mesh, colours, planar))

    def _coloured_sphere(self, points, quads, sigma: np.ndarray | None, planar: bool, colours: SigmaColours):
        """A metallic sphere coloured by σ, or plain grey while σ is not known yet."""
        center = points.mean(axis=0)
        if planar:
            radius = np.linalg.norm(points[0] - center)
            disc = pv.Disc(center=center - [0, 0, PLANE_BELOW], inner=0.0, outer=radius, c_res=72)
            actors = [self.pl.add_mesh(disc, color=SPHERE_INSIDE, lighting=False, pickable=False, reset_camera=False)]
            if sigma is not None:
                actors.append(self._sigma_mesh(sphere_ring(points, quads, sigma), colours, planar))
        else:
            mesh = pv.PolyData(points, np.hstack((np.full((len(quads), 1), 4), quads)).ravel())
            if sigma is None:
                actors = [
                    self.pl.add_mesh(
                        mesh.clean(), color=SPHERE_INSIDE, smooth_shading=True, pickable=False, reset_camera=False
                    )
                ]
            else:
                mesh.cell_data["sigma"] = sigma / UC_PER_M2
                mesh = mesh.clean().cell_data_to_point_data()  # smooth colours across the panels
                actors = [self._sigma_mesh(mesh, colours, planar, smooth_shading=True)]
        self.sphere_actors.append(actors)
        self.sphere_centers.append(center)

    def move_sphere(self, index: int, center: np.ndarray) -> None:
        """Draw sphere index at a new centre (while it is dragged, before its charges are solved again)."""
        if index < len(self.sphere_actors):
            for actor in self.sphere_actors[index]:
                actor.position = np.asarray(center) - self.sphere_centers[index]

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
