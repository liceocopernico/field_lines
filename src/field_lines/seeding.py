"""Where field lines start: the number of lines from a source is proportional to its charge.

Lines start from the sources of the sign with the larger total charge:
forward along E from positive sources, or backwards from negative ones
when the negative charge is larger. Lines that do not end on a source of
the other sign go off to infinity, as Gauss's law requires.
"""

from collections.abc import Sequence

import numpy as np

from field_lines.sources import FieldModel, Plate, PointCharge
from field_lines.units import NC

PLATE_SAMPLES = 48  # candidate seed positions along each side of a plate face
SPHERE_SAMPLES = 800  # candidate seed positions on a sphere (in 2D: around its circle)


def fibonacci_sphere(n: int) -> np.ndarray:
    """n nearly evenly spaced unit vectors."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5**0.5) * k
    r = np.sqrt(1 - z * z)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def circle(n: int) -> np.ndarray:
    """n evenly spaced unit vectors in the xy plane, the first along +x."""
    angle = 2 * np.pi * np.arange(n) / n
    return np.column_stack((np.cos(angle), np.sin(angle), np.zeros(n)))


def line_count(charge: float, lines_per_nc: float) -> int:
    if charge == 0:
        return 0
    return max(1, round(lines_per_nc * abs(charge) / NC))


def seed_sign(charges: Sequence[PointCharge], plates: Sequence[Plate]) -> int:
    """+1 if the positive charge is at least as large as the negative one, else -1."""
    values = [c.q for c in charges] + [p.charge for p in plates]
    positive = sum(v for v in values if v > 0)
    negative = -sum(v for v in values if v < 0)
    return 1 if positive >= negative else -1


def charge_seeds(charge: PointCharge, n: int, radius: float, planar: bool) -> np.ndarray:
    directions = circle(n) if planar else fibonacci_sphere(n)
    return charge.position + radius * directions


def plate_seeds(model: FieldModel, plate: Plate, n: int, offset: float, sign: int, planar: bool) -> np.ndarray:
    """n points just off the plate's faces, spread in proportion to the flux leaving each face.

    planar: only points on the z = 0 plane (the plate's normal must lie in that plane).
    """
    a, b = plate.width / 2, plate.height / 2
    cells_u = np.linspace(-a, a, PLATE_SAMPLES + 1)
    cells_v = np.linspace(-b, b, PLATE_SAMPLES + 1)
    u = (cells_u[:-1] + cells_u[1:]) / 2
    v = (cells_v[:-1] + cells_v[1:]) / 2
    if planar:
        z_axis = plate.local_axes.index(2)
        if z_axis == 2:
            return np.empty((0, 3))  # a plate lying in the plane: no field line leaves it within the plane
        fixed = -plate.center[2]
        if z_axis == 0:
            u = np.array([fixed])
        else:
            v = np.array([fixed])
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uu[1::2] = uu[1::2, ::-1]  # serpentine order, so neighbouring candidates are neighbours in space
    vv[1::2] = vv[1::2, ::-1]
    candidates, weights = [], []
    for face in (1, -1):
        local = np.column_stack((uu.ravel(), vv.ravel(), np.full(uu.size, face * offset)))
        points = plate.from_local(local) + plate.center
        normal = plate.from_local(np.array([[0.0, 0.0, face]]))[0]
        flux = sign * model.field(points) @ normal
        candidates.append(points)
        weights.append(np.maximum(flux, 0.0))
    return _spread(np.concatenate(candidates), np.concatenate(weights), n)


def sphere_seeds(model: FieldModel, center: np.ndarray, radius: float, n: int, offset: float, sign: int, planar: bool):
    """n points just off a metallic sphere, spread in proportion to the flux leaving each part of it.

    planar: only points on its circle in the z = 0 plane (its centre must lie in that plane).
    """
    directions = circle(SPHERE_SAMPLES) if planar else fibonacci_sphere(SPHERE_SAMPLES)
    candidates = center + (radius + offset) * directions
    flux = sign * np.einsum("ij,ij->i", model.field(candidates), directions)
    return _spread(candidates, np.maximum(flux, 0.0), n)


def _spread(candidates: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    """n of the candidates, as many from each part as its share of the weight (systematic sampling)."""
    total = weights.sum()
    if n == 0 or total == 0:
        return np.empty((0, 3))
    targets = (np.arange(n) + 0.5) * total / n
    chosen = np.searchsorted(np.cumsum(weights), targets)
    return candidates[np.minimum(chosen, len(candidates) - 1)]


def seeds(
    model: FieldModel,
    charges: Sequence[PointCharge],
    plates: Sequence[Plate],
    *,
    lines_per_nc: float,
    charge_radius: float,
    plate_offset: float,
    planar: bool,
    spheres: Sequence = (),
    conductor=None,
    sphere_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Seed points and directions (+1 along E, -1 against it) for all sources.

    In the conductors model (conductor: a conductors.ConductorSolution, whose
    conductors are the plates, then the spheres) a conductor may be charged
    positive in some parts and negative in others: the lines from a conductor
    are as many as the charge of its parts of the seeding sign.
    """
    if conductor is not None:
        return _conductor_seeds(model, plates, spheres, conductor, lines_per_nc, plate_offset, sphere_offset, planar)
    sign = seed_sign(charges, plates)
    points = [np.empty((0, 3))]
    for charge in charges:
        if charge.q * sign > 0:
            points.append(charge_seeds(charge, line_count(charge.q, lines_per_nc), charge_radius, planar))
    for plate in plates:
        if plate.charge * sign > 0:
            n = line_count(plate.charge, lines_per_nc)
            points.append(plate_seeds(model, plate, n, plate_offset, sign, planar))
    points = np.concatenate(points)
    return points, np.full(len(points), sign)


def _conductor_seeds(model, plates, spheres, conductor, lines_per_nc, plate_offset, sphere_offset, planar):
    count = len(plates) + len(spheres)
    positive = sum(conductor.signed_charge(k, 1) for k in range(count))
    negative = sum(conductor.signed_charge(k, -1) for k in range(count))
    sign = 1 if positive >= negative else -1
    points = [np.empty((0, 3))]
    for k, plate in enumerate(plates):
        n = line_count(conductor.signed_charge(k, sign), lines_per_nc)
        points.append(plate_seeds(model, plate, n, plate_offset, sign, planar))
    for k, sphere in enumerate(spheres):
        n = line_count(conductor.signed_charge(len(plates) + k, sign), lines_per_nc)
        points.append(sphere_seeds(model, sphere.center, sphere.radius, n, sphere_offset, sign, planar))
    points = np.concatenate(points)
    return points, np.full(len(points), sign)
