"""The field sampled on grids: equipotentials, arrows and the coloured slice plane.

These functions only compute meshes (no actors), so they can run in a worker thread.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from field_lines.scene import VIEW_HALF
from field_lines.sources import AXES, FieldModel

EQUIPOTENTIAL_RES_3D = 72  # grid points per side
EQUIPOTENTIAL_RES_CONDUCTORS = 48  # in 3D in the conductors model, whose field is slower to evaluate (~1.5 s)
EQUIPOTENTIAL_RES_2D = 241
ARROWS_3D = 9
ARROWS_2D = 17
SLICE_RES = 161
PLANE_BELOW = 0.0005  # m: in 2D the coloured plane sits just below the lines
CHUNK = 200_000  # points per field evaluation, to bound memory


@dataclass
class Exclusion:
    """A ball around a point charge where the field is too strong to sample meaningfully."""

    center: np.ndarray
    radius: float


def view_grid(n: int, planar: bool) -> pv.ImageData:
    spacing = 2 * VIEW_HALF / (n - 1)
    dims = (n, n, 1) if planar else (n, n, n)
    origin = (-VIEW_HALF, -VIEW_HALF, 0.0 if planar else -VIEW_HALF)
    return pv.ImageData(dimensions=dims, spacing=(spacing,) * 3, origin=origin)


def evaluate(model: FieldModel, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V and E at many points, in chunks."""
    V = np.empty(len(points))
    E = np.empty((len(points), 3))
    for start in range(0, len(points), CHUNK):
        part = points[start : start + CHUNK]
        V[start : start + CHUNK] = model.potential(part)
        E[start : start + CHUNK] = model.field(part)
    return V, E


def evaluate_potential(model: FieldModel, points: np.ndarray) -> np.ndarray:
    V = np.empty(len(points))
    for start in range(0, len(points), CHUNK):
        V[start : start + CHUNK] = model.potential(points[start : start + CHUNK])
    return V


def outside(points: np.ndarray, exclusions: Sequence[Exclusion], factor: float = 1.0) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for ex in exclusions:
        keep &= np.linalg.norm(points - ex.center, axis=1) > factor * ex.radius
    return keep


def potential_levels(values: np.ndarray, count: int) -> np.ndarray:
    """count equally spaced potentials covering the bulk of the values (5th to 95th percentile).

    Equal steps in V: the surfaces crowd where the field is strong. With an odd
    count and a symmetric setup (dipole, capacitor) V = 0 is one of the levels.
    """
    lo, hi = np.percentile(values, [5, 95])
    if count == 1:
        return np.array([(lo + hi) / 2])
    return np.linspace(lo, hi, count)


def equipotentials(
    model: FieldModel, planar: bool, count: int, exclusions: Sequence[Exclusion], resolution: int | None = None
) -> tuple[pv.PolyData, np.ndarray]:
    """Equipotential surfaces (lines in 2D) and their potentials; resolution: grid points per side."""
    resolution = resolution or (EQUIPOTENTIAL_RES_2D if planar else EQUIPOTENTIAL_RES_3D)
    grid = view_grid(resolution, planar)
    points = np.asarray(grid.points)
    V = evaluate_potential(model, points)
    grid.point_data["V"] = V
    levels = potential_levels(V[outside(points, exclusions, 2.0)], count)
    return grid.contour(isosurfaces=levels, scalars="V"), levels


def arrows(model: FieldModel, planar: bool, exclusions: Sequence[Exclusion]) -> pv.PolyData:
    """One fixed-length arrow per grid point along E, carrying |E| as "E" for colouring."""
    n = ARROWS_2D if planar else ARROWS_3D
    grid = view_grid(n, planar)
    spacing = grid.spacing[0]
    points = np.asarray(grid.points)
    points = points[outside(points, exclusions, 1.5)]
    E = model.field(points)
    mag = np.linalg.norm(E, axis=1)
    keep = mag > 0
    cloud = pv.PolyData(points[keep])
    cloud.point_data["dir"] = E[keep] / mag[keep, None]
    cloud.point_data["E"] = mag[keep]
    arrow = pv.Arrow(start=(-0.5, 0, 0), tip_length=0.35, tip_radius=0.14, shaft_radius=0.05)
    return cloud.glyph(orient="dir", scale=False, factor=0.75 * spacing, geom=arrow)


def slice_plane(model: FieldModel, axis: str, position: float, planar: bool) -> pv.ImageData:
    """A square plane across the view, perpendicular to axis, with "V" and "E" (=|E|) on it."""
    if planar:
        axis, position = "z", -PLANE_BELOW
    a = AXES.index(axis)
    spacing = 2 * VIEW_HALF / (SLICE_RES - 1)
    dims = [SLICE_RES] * 3
    dims[a] = 1
    origin = [-VIEW_HALF] * 3
    origin[a] = position
    plane = pv.ImageData(dimensions=dims, spacing=(spacing,) * 3, origin=origin)
    V, E = evaluate(model, np.asarray(plane.points))
    plane.point_data["V"] = V
    plane.point_data["E"] = np.maximum(np.linalg.norm(E, axis=1), 1e-30)
    return plane
