"""Where field lines start: the number of lines from a source is proportional to its charge.

Lines start from the sources of the sign with the larger total charge:
forward along E from positive sources, or backwards from negative ones
when the negative charge is larger. Lines that do not end on a source of
the other sign go off to infinity, as Gauss's law requires.
"""

from collections.abc import Sequence

import numpy as np

from field_lines.sources import FieldModel, Plate, PointCharge, Ring
from field_lines.units import NC

PLATE_SAMPLES = 48  # candidate seed positions along each side of a plate face
RING_CELLS = 4000  # candidate seed positions all over a disc or a tube
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
    return _larger(positive, negative)


def _larger(positive: float, negative: float) -> int:
    """+1 if the positive charge is at least as large as the negative one (equal ones up to rounding)."""
    return 1 if positive >= negative * (1 - 1e-9) else -1


def charge_seeds(charge: PointCharge, n: int, radius: float, planar: bool) -> np.ndarray:
    directions = circle(n) if planar else fibonacci_sphere(n)
    return charge.position + radius * directions


def plate_seeds(model: FieldModel, plate, n: int, offset: float, sign: int, planar: bool) -> np.ndarray:
    """n points just off the plate's surface, spread in proportion to the flux leaving each part of it.

    A slab has six faces: the two large ones and four rims across its
    thickness; a disc or a tube (a Ring) has two flat faces and one or two
    cylindrical ones. Each face gets its share of the lines.
    planar: only points on the z = 0 plane; a plate lying in that plane has none.
    """
    cells = _ring_cells(plate, offset, planar) if isinstance(plate, Ring) else _surface_cells(plate, offset, planar)
    candidates, weights = [], []
    for local, normals, measure in cells:
        points = plate.from_local(local) + plate.center
        normals = plate.from_local(np.broadcast_to(normals, local.shape).copy())
        flux = sign * np.einsum("ij,ij->i", model.field(points), normals)
        candidates.append(points)
        weights.append(np.maximum(flux, 0.0) * measure)
    if not candidates:
        return np.empty((0, 3))
    shares = _shares([w.sum() for w in weights], n)  # each face its share of the lines
    seeds = [_stratified(c, w, k) for c, w, k in zip(candidates, weights, shares)]
    return np.concatenate(seeds)


def _surface_cells(plate: Plate, offset: float, planar: bool):
    """Cells covering a rectangular plate's faces, offset outwards: (local centres, local normal, cell area).

    planar: the cells along the plate's cut by the z = 0 plane, with their length for the area.
    """
    half = plate.half
    fixed = None
    if planar:  # the local axis along z, and the value of that coordinate on the plane
        fixed = plate.local_axes.index(2), -plate.center[2]
        if fixed[0] == 2:
            return  # a plate lying in the plane: no field line leaves it within the plane
    spacing = 2 * half[:2].max() / PLATE_SAMPLES

    def samples(axis: int) -> tuple[np.ndarray, float]:
        """Cell centres along a local axis of the plate, and the cells' size."""
        if fixed is not None and fixed[0] == axis:
            return np.array([fixed[1]]), 1.0
        count = PLATE_SAMPLES if axis < 2 else max(1, round(2 * half[2] / spacing))
        edges = np.linspace(-half[axis], half[axis], count + 1)
        return (edges[:-1] + edges[1:]) / 2, 2 * half[axis] / count

    for axis in range(3) if half[2] > 0 else (2,):
        if fixed is not None and fixed[0] == axis:
            continue  # the faces across the plane's axis are not cut by the plane
        others = [k for k in range(3) if k != axis]
        (s0, d0), (s1, d1) = samples(others[0]), samples(others[1])
        grid = np.meshgrid(s0, s1, indexing="ij")
        for side in (1, -1):
            local = np.empty((grid[0].size, 3))
            local[:, others[0]], local[:, others[1]] = grid[0].ravel(), grid[1].ravel()
            local[:, axis] = side * (half[axis] + offset)
            normal = np.zeros(3)
            normal[axis] = side
            yield local, normal, d0 * d1


def ring_faces(ring: Ring) -> list[tuple[str, float, int]]:
    """A ring's faces: ("w", w, side) flat ones at w with normal side along w, ("rho", ρ, side) cylinders.

    A sheet's face counts twice, once for each side.
    """
    h, r_in, r_out = ring.half_length, ring.r_in, ring.r_out
    faces = []
    if r_out > r_in:
        faces += [("w", h, 1), ("w", -h, -1)]
    if h > 0:
        faces += [("rho", r_out, 1)] + ([("rho", r_in, -1)] if r_in > 0 else [])
    return faces


def _ring_cells(ring: Ring, offset: float, planar: bool):
    """Cells covering a disc's or a tube's faces, offset outwards (see _surface_cells).

    In 3D, about RING_CELLS cells in all. In 2D, along the ring's cut by the
    z = 0 plane: two circles if its axis is along z, or the outline of its
    cross-section through the axis if the axis lies in the plane.
    """
    h, r_in, r_out = ring.half_length, ring.r_in, ring.r_out
    faces = ring_faces(ring)
    area = sum(np.pi * (r_out**2 - r_in**2) if kind == "w" else 2 * np.pi * at * 2 * h for kind, at, _ in faces)
    spacing = max(r_out, h) / PLATE_SAMPLES if planar else np.sqrt(area / RING_CELLS)

    def cells(lo: float, hi: float) -> tuple[np.ndarray, float]:
        count = max(1, round((hi - lo) / spacing))
        edges = np.linspace(lo, hi, count + 1)
        return (edges[:-1] + edges[1:]) / 2, (hi - lo) / count

    def around(radius: float) -> tuple[np.ndarray, float]:
        count = max(8, round(2 * np.pi * radius / spacing))
        return (np.arange(count) + 0.5) * 2 * np.pi / count, 2 * np.pi / count

    z_axis = ring.local_axes.index(2)
    if planar and z_axis == 2:  # the axis across the plane: the cylinders' circles, where the plane cuts them
        cut = -ring.center[2]
        if abs(cut) > h:
            return
        for kind, radius, side in faces:
            if kind == "rho":
                phi, d_phi = around(radius)
                r = radius + side * offset
                local = np.column_stack((r * np.cos(phi), r * np.sin(phi), np.full(len(phi), cut)))
                yield local, side * np.column_stack((np.cos(phi), np.sin(phi), np.zeros(len(phi)))), radius * d_phi
        return
    if planar:  # the axis in the plane: the two halves of the cross-section through it
        if ring.center[2] != 0:
            return
        angles = np.array([0.0, np.pi]) if z_axis == 1 else np.array([np.pi / 2, 3 * np.pi / 2])
    for kind, at, side in faces:
        if kind == "w":
            s, d_s = cells(r_in, r_out)
        else:
            s, d_s = cells(-h, h)
        if planar:
            phi, measure = np.repeat(angles, len(s)), np.full(2 * len(s), d_s)
            s = np.tile(s, 2)
        else:
            phi, d_phi = around((r_in + r_out) / 2 if kind == "w" else at)
            s, phi = (g.ravel() for g in np.meshgrid(s, phi, indexing="ij"))
            measure = (s if kind == "w" else at) * d_s * d_phi
        radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(len(phi))))
        if kind == "w":  # a flat face: s is ρ
            local = s[:, None] * radial + [0.0, 0.0, at + side * offset]
            normals = np.array([0.0, 0.0, side])
        else:  # a cylinder: s is w
            local = (at + side * offset) * radial + s[:, None] * [0.0, 0.0, 1.0]
            normals = side * radial
        if planar:
            local[:, z_axis] = 0.0  # exactly in the plane (cos and sin of the angles leave a trace)
        yield local, normals, measure


def sphere_seeds(model: FieldModel, center: np.ndarray, radius: float, n: int, offset: float, sign: int, planar: bool):
    """n points just off a metallic sphere, spread in proportion to the flux leaving each part of it.

    planar: only points on its circle in the z = 0 plane (its centre must lie in that plane).
    """
    directions = circle(SPHERE_SAMPLES) if planar else fibonacci_sphere(SPHERE_SAMPLES)
    candidates = center + (radius + offset) * directions
    flux = sign * np.einsum("ij,ij->i", model.field(candidates), directions)
    return _stratified(candidates, np.maximum(flux, 0.0), n)


def _shares(weights: Sequence[float], n: int) -> list[int]:
    """n split into whole numbers in proportion to the weights (largest remainders)."""
    total = sum(weights)
    if n == 0 or total <= 0:
        return [0] * len(weights)
    exact = [n * w / total for w in weights]
    shares = [int(e) for e in exact]
    for i in sorted(range(len(weights)), key=lambda i: exact[i] - shares[i], reverse=True)[: n - sum(shares)]:
        shares[i] += 1
    return shares


def _stratified(candidates: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    """n of the candidates, evenly spread with density proportional to the weight.

    The candidates are cut in two across their longest extent, so that each
    part carries weight in proportion to the lines it gets, and each part is
    cut again, until a part gets one line: it starts at the candidate nearest
    the part's centre of weight. Unlike taking candidates at equal steps of
    weight along a list, this spreads the lines over a surface, not in rows.
    """
    keep = weights > 0
    points, w = candidates[keep], weights[keep]
    seeds: list[np.ndarray] = []

    def split(index: np.ndarray, k: int) -> None:
        if k == 0 or len(index) == 0:
            return
        part = points[index]
        if k == 1 or len(index) == 1:
            centre = (part * w[index, None]).sum(axis=0) / w[index].sum()
            nearest = part[np.argmin(np.linalg.norm(part - centre, axis=1))]
            seeds.extend([nearest] * k)
            return
        axis = int(np.argmax(part.max(axis=0) - part.min(axis=0)))
        order = index[np.argsort(part[:, axis], kind="stable")]
        cumulative = np.cumsum(w[order])
        k_left = k // 2
        cut = int(np.clip(np.searchsorted(cumulative, cumulative[-1] * k_left / k) + 1, 1, len(order) - 1))
        split(order[:cut], k_left)
        split(order[cut:], k - k_left)

    split(np.arange(len(points)), n)
    return np.array(seeds).reshape(-1, 3)


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
    sign = _larger(positive, negative)
    points = [np.empty((0, 3))]
    for k, plate in enumerate(plates):
        n = line_count(conductor.signed_charge(k, sign), lines_per_nc)
        layout = conductor.solver.layouts[k]
        if planar or layout is None:
            points.append(plate_seeds(model, plate, n, plate_offset, sign, planar))
        else:
            faces = [np.arange(start, start + shape[0] * shape[1]) for start, shape in layout.values()]
            def on_plate(q, normals, plate=plate):
                return plate.surface_points(q) + plate_offset * normals

            points.append(_panel_seeds(conductor, faces, n, sign, on_plate))
    for k, sphere in enumerate(spheres):
        n = line_count(conductor.signed_charge(len(plates) + k, sign), lines_per_nc)
        if planar:
            points.append(sphere_seeds(model, sphere.center, sphere.radius, n, sphere_offset, sign, planar))
        else:
            mine = np.flatnonzero(conductor.panels.owner == len(plates) + k)

            def on_sphere(q, normals, sphere=sphere):
                out = (q - sphere.center) / np.linalg.norm(q - sphere.center, axis=1, keepdims=True)
                return sphere.center + (sphere.radius + sphere_offset) * out

            points.append(_panel_seeds(conductor, [mine], n, sign, on_sphere))
    points = np.concatenate(points)
    return points, np.full(len(points), sign)


def _panel_seeds(conductor, faces, n: int, sign: int, place) -> np.ndarray:
    """n points just off a conductor in 3D, from the charges on its panels, without the field.

    On a conductor the flux leaving a panel is its charge over ε₀, so the
    panels' σ give the weights. Each panel offers four candidates, one in
    each of its quarters; place(points, normals) puts them just off the
    surface. faces: the panel indices of each face, which gets its share of
    the lines.
    """
    panels = conductor.panels
    candidates, weights = [], []
    for index in faces:
        quarters = (panels.centers[index, None, :] + panels.corners[index] / 2).reshape(-1, 3)
        candidates.append(place(quarters, np.repeat(panels.normals[index], 4, axis=0)))
        weights.append(np.repeat(np.maximum(sign * conductor.sigma[index], 0.0) * panels.area[index] / 4, 4))
    shares = _shares([w.sum() for w in weights], n)
    return np.concatenate([_stratified(c, w, k) for c, w, k in zip(candidates, weights, shares)])
