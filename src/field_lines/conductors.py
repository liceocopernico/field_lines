"""Conductors held at fixed potentials, alone in empty space: capacitor plates and metallic spheres.

A conductor's charge sits on its surface. Each surface is cut into small flat
panels, each with a uniform surface charge density, and the densities are
chosen so that the centre of every panel is at its conductor's potential (the
method of moments, or boundary elements). That solves Laplace's equation
outside the conductors with V -> 0 far away, without a grid.

- A plate with a thickness is a box: its two large faces are cut into
  PLATE_PANELS x PLATE_PANELS panels, finer towards their edges, where the
  charge piles up, and its four rims into strips of panels. Every face then
  has its own charge. (A plate of zero thickness is one sheet of panels; its
  two faces are told apart afterwards: each carries σ/2 plus or minus ε₀
  times the field of all the other panels across it.)
- A disc-shaped plate or a tube (a Ring, a body of revolution) has a polar
  grid of RING_SEGMENTS panels around: a disc's flat faces DISC_PANELS rings
  of them, finer towards the rim (triangles at the centre), a tube up to
  TUBE_PANELS rows along it, finer towards its ends.
- A sphere is a cube projected onto it: 6 faces of SPHERE_PANELS x
  SPHERE_PANELS panels.

The potential or the field of a panel at a point is that of the uniformly
charged flat polygon it is (closed form, summed over its edges) when the
point is within NEAR panel sizes, and that of a point charge farther away.
The same rule builds the matrix of the solve and evaluates the field
afterwards, so the potentials come out as asked. Panels count with their
true shape: the cube's corners make kite-shaped panels on a sphere, and
counting them as squares would leave σ 7% off there.

The matrix depends only on the geometry; its LU factors are kept in a
Solver, and a new set of potentials is a quick Solver.solve(voltages).

To evaluate the field at many points quickly, neighbouring panels are
grouped into clusters: seen from farther than CLUSTER_NEAR cluster radii, a
cluster is a point charge plus a dipole at its centre; nearer, its panels
count one by one.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from field_lines.sources import Plate, Ring
from field_lines.units import EPS0, K

PLATE_PANELS = 21  # per side of a plate's face; odd, so that a row of panels lies on each centre line
SPHERE_PANELS = 8  # per edge of each of the six faces of the cube
RING_SEGMENTS = 40  # panels around a disc or a tube
DISC_PANELS = 12  # rings of panels on a disc's flat face, from the centre to the rim
TUBE_PANELS = 20  # rows of panels along a tube, at most
SHEET_SIDES = 48  # in the charges model a disc is a polygon, a cylindrical sheet a prism, of this many sides
NEAR = 4.0  # closer than this many panel sizes, a panel is a rectangle, not a point charge
CHUNK = 1000  # points per block of point-panel pairs
PLATE_BLOCK = 3  # a plate face's clusters are blocks of 3 x 3 panels (smaller: fewer panels one by one near a slab)
SPHERE_BLOCK = 4  # a sphere's are blocks of 4 x 4 panels on each face of the cube
CLUSTER_NEAR = 4.0  # farther than this many cluster radii, a cluster is a charge and a dipole


@dataclass
class Sphere:
    center: np.ndarray  # m
    radius: float  # m
    voltage: float = 0.0  # V, with respect to infinity

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)


@dataclass
class Panels:
    centers: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3)
    corners: np.ndarray  # (N, 4, 3) relative to the centres, in each panel's plane, counter-clockwise about its normal
    owner: np.ndarray  # (N,) index of the conductor the panel belongs to

    def __post_init__(self):
        cross = sum(np.cross(self.corners[:, e], self.corners[:, (e + 1) % 4]) for e in range(4))
        self.area = 0.5 * np.einsum("pk,pk->p", cross, self.normals)
        self.size = 2 * np.linalg.norm(self.corners, axis=2).max(axis=1)

    def __len__(self) -> int:
        return len(self.centers)


def _join(parts: Sequence[Panels]) -> Panels:
    fields = ("centers", "normals", "corners", "owner")
    return Panels(*(np.concatenate([getattr(p, f) for p in parts]) for f in fields))


def panel_edges(half: float, n: int) -> np.ndarray:
    """n + 1 edges from -half to half, closer together towards the ends (cosine spacing)."""
    return -half * np.cos(np.linspace(0.0, np.pi, n + 1))


def plate_face_grid(plate: Plate, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The panel centres of a plate along its u and v, as plate_panels lays them out."""
    n = n or PLATE_PANELS
    eu, ev = panel_edges(plate.width / 2, n), panel_edges(plate.height / 2, n)
    return (eu[1:] + eu[:-1]) / 2, (ev[1:] + ev[:-1]) / 2


def _counter_clockwise(corners: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """The corners turned counter-clockwise about the normal (reversed where they are not)."""
    turn = np.einsum("pk,pk->p", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), normal)
    corners = corners.copy()
    corners[turn < 0] = corners[turn < 0, ::-1]
    return corners


def _face_grid(plate: Plate, along: tuple[int, int], edges: tuple[np.ndarray, np.ndarray], at: float, outward: int):
    """Panels of one face of a box plate: a grid over local axes along[0] x along[1] at local[outward axis] = at.

    Returns centres, normals and corners (global, corners relative to the centres), row-major over the grid.
    """
    (i, j), (e1, e2) = along, edges
    k = 3 - i - j
    c1, c2 = (e1[1:] + e1[:-1]) / 2, (e2[1:] + e2[:-1]) / 2
    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
    h1, h2 = (h.ravel() for h in np.meshgrid(np.diff(e1) / 2, np.diff(e2) / 2, indexing="ij"))
    count = g1.size
    local = np.zeros((count, 3))
    local[:, i], local[:, j], local[:, k] = g1.ravel(), g2.ravel(), at
    corner_local = np.zeros((count, 4, 3))
    for c, (s1, s2) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        corner_local[:, c, i], corner_local[:, c, j] = s1 * h1, s2 * h2
    normal_local = np.zeros(3)
    normal_local[k] = 1.0 if at >= 0 else -1.0
    normal = plate.from_local(normal_local[None])[0]
    corners = plate.from_local(corner_local.reshape(-1, 3)).reshape(count, 4, 3)
    normals = np.repeat(normal[None], count, axis=0)
    return plate.from_local(local) + plate.center, normals, _counter_clockwise(corners, normals)


def box_panels(plate: Plate, owner: int, n: int | None = None) -> tuple[Panels, dict]:
    """The six faces of a slab: the large faces at local w = ±t/2, then the rims at u = ±a and v = ±b.

    Returns the panels and the layout: for each face, its first panel and grid shape.
    """
    n = n or PLATE_PANELS
    a, b, half_t = plate.half
    eu, ev = panel_edges(a, n), panel_edges(b, n)
    # panels across the thickness, about as wide as the face panels
    across = max(1, round(2 * half_t / (2 * a / n)))
    ew = np.linspace(-half_t, half_t, across + 1)
    faces = [
        ("w+", (0, 1), (eu, ev), half_t),
        ("w-", (0, 1), (eu, ev), -half_t),
        ("u+", (1, 2), (ev, ew), a),
        ("u-", (1, 2), (ev, ew), -a),
        ("v+", (0, 2), (eu, ew), b),
        ("v-", (0, 2), (eu, ew), -b),
    ]
    parts, layout, first = [], {}, 0
    for name, along, edges, at in faces:
        centers, normals, corners = _face_grid(plate, along, edges, at, 3 - sum(along))
        shape = (len(edges[0]) - 1, len(edges[1]) - 1)
        layout[name] = (first, shape)
        first += shape[0] * shape[1]
        parts.append(Panels(centers, normals, corners, np.full(len(centers), owner)))
    return _join(parts), layout


def plate_panels(plate: Plate, owner: int, n: int | None = None) -> Panels:
    n = n or PLATE_PANELS
    eu, ev = panel_edges(plate.width / 2, n), panel_edges(plate.height / 2, n)
    cu, cv = plate_face_grid(plate, n)
    uu, vv = np.meshgrid(cu, cv, indexing="ij")
    local = np.column_stack((uu.ravel(), vv.ravel(), np.zeros(uu.size)))
    e_u, e_v, normal = plate.from_local(np.eye(3))  # the global directions of local u, v, w
    hu, hv = (h.ravel()[:, None] for h in np.meshgrid(np.diff(eu) / 2, np.diff(ev) / 2, indexing="ij"))
    corners = np.stack((-hu * e_u - hv * e_v, hu * e_u - hv * e_v, hu * e_u + hv * e_v, -hu * e_u + hv * e_v), axis=1)
    normals = np.repeat(normal[None], uu.size, axis=0)
    return Panels(plate.from_local(local) + plate.center, normals, corners, np.full(uu.size, owner))


def _ring_panels_from(ring: Ring, corners: np.ndarray, normals: np.ndarray, owner: int) -> Panels:
    """Panels from corners (N, 4, 3) and normals (N, 3) in the ring's local coordinates.

    Each panel's centre is its centroid; a triangle is a quad with two corners alike.
    """
    a, b, c, d = (corners[:, k] for k in range(4))
    areas = [0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1), 0.5 * np.linalg.norm(np.cross(c - a, d - a), axis=1)]
    centroids = [(a + b + c) / 3, (a + c + d) / 3]
    center = (areas[0][:, None] * centroids[0] + areas[1][:, None] * centroids[1]) / (areas[0] + areas[1])[:, None]
    count = len(corners)
    normals = ring.from_local(normals)
    rel = ring.from_local((corners - center[:, None, :]).reshape(-1, 3)).reshape(count, 4, 3)
    rel = _counter_clockwise(rel, normals)
    return Panels(ring.from_local(center) + ring.center, normals, rel, np.full(count, owner))


def ring_face_grid(ring: Ring, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The panel edges of one face of a ring: along it (ρ on a flat face, w on a cylinder) and around (φ)."""
    h, r_in, r_out = ring.half_length, ring.r_in, ring.r_out
    target = 2 * np.pi * r_out / RING_SEGMENTS  # about the panels' width around the rim
    phi = np.linspace(0.0, 2 * np.pi, RING_SEGMENTS + 1)
    if name in ("w+", "w-"):
        if r_in == 0:  # a disc: finer towards the rim
            return r_out * np.sin(np.linspace(0.0, np.pi / 2, DISC_PANELS + 1)), phi
        count = max(1, round((r_out - r_in) / target))
        return (r_in + r_out) / 2 + panel_edges((r_out - r_in) / 2, count), phi
    count = min(TUBE_PANELS, max(1, round(2 * h / target)))
    return panel_edges(h, count), phi


def ring_panels(ring: Ring, owner: int) -> tuple[Panels, dict]:
    """The faces of a solid ring: its flat faces "w+" and "w-", the cylinders "out" and (a tube's) "in".

    Returns the panels and the layout: for each face, its first panel and grid shape (along, around).
    """
    h, r_in, r_out = ring.half_length, ring.r_in, ring.r_out
    assert h > 0 and r_out > r_in, "a conductor has a thickness"
    names = ["w+", "w-", "out"] + (["in"] if r_in > 0 else [])
    parts, layout, first = [], {}, 0
    for name in names:
        s, phi = ring_face_grid(ring, name)
        ss, pp = np.meshgrid(s, phi, indexing="ij")
        corner_index = [(0, 0), (1, 0), (1, 1), (0, 1)]  # (along, around) offsets, counter-clockwise for "w+"
        cs = np.stack([ss[i : len(s) - 1 + i, j : len(phi) - 1 + j] for i, j in corner_index], axis=-1)
        cp = np.stack([pp[i : len(s) - 1 + i, j : len(phi) - 1 + j] for i, j in corner_index], axis=-1)
        cs, cp = cs.reshape(-1, 4), cp.reshape(-1, 4)
        mid = (phi[1:] + phi[:-1]) / 2
        mid = np.tile(mid, len(s) - 1)
        if name in ("w+", "w-"):
            side = 1.0 if name == "w+" else -1.0
            corners = np.stack((cs * np.cos(cp), cs * np.sin(cp), np.full(cs.shape, side * h)), axis=-1)
            normals = np.tile([0.0, 0.0, side], (len(mid), 1))
        else:
            radius, side = (r_out, 1.0) if name == "out" else (r_in, -1.0)
            corners = np.stack((radius * np.cos(cp), radius * np.sin(cp), cs), axis=-1)
            normals = side * np.column_stack((np.cos(mid), np.sin(mid), np.zeros(len(mid))))
        layout[name] = (first, (len(s) - 1, len(phi) - 1))
        first += len(mid)
        parts.append(_ring_panels_from(ring, corners, normals, owner))
    return _join(parts), layout


def sheet_panels(ring: Ring) -> Panels:
    """A disc (or annulus) sheet as a regular polygon of SHEET_SIDES sides, a cylindrical sheet as a prism.

    The polygon is a little larger than the circle, so that it has the same
    area (the prism the same perimeter): the same σ then carries the same charge.
    """
    theta = np.linspace(0.0, 2 * np.pi, SHEET_SIDES + 1)
    h = ring.half_length
    n = SHEET_SIDES
    grow = np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n))) if h == 0 else np.pi / (n * np.sin(np.pi / n))
    r_in, r_out = grow * ring.r_in, grow * ring.r_out
    zero = np.zeros(SHEET_SIDES // 2)
    if h == 0 and r_in == 0:  # quads from the centre, each over two sides
        k = np.arange(0, SHEET_SIDES, 2)
        rim = [theta[k + i] for i in (0, 1, 2)]
        corners = [np.column_stack((zero, zero, zero))]
        corners += [np.column_stack((r_out * np.cos(angle), r_out * np.sin(angle), zero)) for angle in rim]
        normals = np.tile([0.0, 0.0, 1.0], (len(k), 1))
    elif h == 0:
        c0, c1 = np.cos(theta[:-1]), np.cos(theta[1:])
        s0, s1 = np.sin(theta[:-1]), np.sin(theta[1:])
        z = np.zeros(SHEET_SIDES)
        corners = [np.column_stack(c) for c in ((r_in * c0, r_in * s0, z), (r_out * c0, r_out * s0, z),
                                                 (r_out * c1, r_out * s1, z), (r_in * c1, r_in * s1, z))]
        normals = np.tile([0.0, 0.0, 1.0], (SHEET_SIDES, 1))
    else:
        assert r_in == r_out, "a sheet has no thickness"
        c0, c1 = np.cos(theta[:-1]), np.cos(theta[1:])
        s0, s1 = np.sin(theta[:-1]), np.sin(theta[1:])
        lo, hi = np.full(SHEET_SIDES, -h), np.full(SHEET_SIDES, h)
        corners = [np.column_stack(c) for c in ((r_out * c0, r_out * s0, lo), (r_out * c1, r_out * s1, lo),
                                                 (r_out * c1, r_out * s1, hi), (r_out * c0, r_out * s0, hi))]
        mid = (theta[1:] + theta[:-1]) / 2
        normals = np.column_stack((np.cos(mid), np.sin(mid), np.zeros(SHEET_SIDES)))
    return _ring_panels_from(ring, np.stack(corners, axis=1), normals, 0)


class UniformSheet:
    """A uniformly charged disc or cylindrical sheet (a Ring of zero thickness), for the charges model.

    It is a regular polygon or a prism of SHEET_SIDES flat panels (sheet_panels),
    each counted with the exact field of a flat polygon, so the field is right
    also right next to the sheet, and it carries the charge of the round sheet.
    """

    def __init__(self, ring: Ring):
        self.ring = ring
        self.panels = sheet_panels(ring)
        self.sigma = np.full(len(self.panels), ring.sigma)

    def field(self, points: np.ndarray) -> np.ndarray:
        return evaluate(self.panels, self.sigma, points, potential=False)[1]

    def potential(self, points: np.ndarray) -> np.ndarray:
        return evaluate(self.panels, self.sigma, points, field=False)[0]


def sphere_mesh(center: np.ndarray, radius: float, m: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Corner points (6 (m+1)², 3) and quads (6 m², 4) of a cube projected onto the sphere.

    The cube's faces are divided at equal angles, so the panels are nearly
    equal. The quads go face by face in the order of sphere_panels.
    """
    m = m or SPHERE_PANELS
    t = np.tan(np.linspace(-np.pi / 4, np.pi / 4, m + 1))
    s1, s2 = np.meshgrid(t, t, indexing="ij")
    one = np.ones_like(s1)
    points, quads = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            cube = [None, None, None]
            cube[axis] = sign * one
            first, second = [k for k in range(3) if k != axis]
            # every face's quads turn the same way seen from outside
            cube[first], cube[second] = (s1, s2) if sign > 0 else (s2, s1)
            p = np.stack(cube, axis=-1).reshape(-1, 3)
            start = len(points) * (m + 1) ** 2
            points.append(center + radius * p / np.linalg.norm(p, axis=1, keepdims=True))
            i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
            corner = start + i * (m + 1) + j
            quads.append(np.stack((corner, corner + m + 1, corner + m + 2, corner + 1), axis=-1).reshape(-1, 4))
    return np.concatenate(points), np.concatenate(quads)


def sphere_panels(sphere: Sphere, owner: int, m: int | None = None) -> Panels:
    """The sphere's quads, each flattened onto the plane touching the sphere at its centre."""
    points, quads = sphere_mesh(sphere.center, sphere.radius, m)
    quad = points[quads]  # (n, 4, 3)
    normal = quad.mean(axis=1) - sphere.center
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    centers = sphere.center + sphere.radius * normal
    corners = quad - centers[:, None, :]
    corners -= np.einsum("pek,pk->pe", corners, normal)[:, :, None] * normal[:, None, :]
    turn = np.einsum("pk,pk->p", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), normal)
    corners[turn < 0] = corners[turn < 0, ::-1]  # counter-clockwise seen from outside
    return Panels(centers, normal, corners, np.full(len(quads), owner))


def polygon_terms(rel: np.ndarray, corners: np.ndarray, normal: np.ndarray, field: bool = True):
    """∫∫ dA/R over flat polygons at points rel from their centres, and the field it gives.

    rel (P, 3); corners (P, k, 3) relative to the centres, in the polygons' planes, counter-clockwise
    seen from normal (P, 3). Returns I (P,) and G (P, 3) (None if not asked), with V = k σ I and
    E = k σ G. Each edge adds its log term (in the plane) and the solid angle it spans (across).
    """
    w = np.einsum("pk,pk->p", rel, normal)
    foot = rel - w[:, None] * normal  # the point's foot on the plane
    aw = np.abs(w)
    I = np.zeros(len(rel))
    G = np.zeros((len(rel), 3)) if field else None
    solid_angle = np.zeros(len(rel))
    k = corners.shape[1]
    for e in range(k):
        a, b = corners[:, e], corners[:, (e + 1) % k]
        edge = b - a
        length = np.linalg.norm(edge, axis=1)
        real = length > 0  # a triangle is a quad with two corners alike: one of its edges has no length
        along = edge / np.where(real, length, 1.0)[:, None]
        out = np.cross(along, normal)  # in the plane, out of the polygon
        l_minus = np.einsum("pk,pk->p", a - foot, along)
        l_plus = l_minus + length
        p0 = np.einsum("pk,pk->p", a - foot, out)
        r0 = np.sqrt(p0 * p0 + w * w)
        r_minus = np.sqrt(l_minus**2 + r0**2)
        r_plus = np.sqrt(l_plus**2 + r0**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(real & (r0 > 0), np.arcsinh(l_plus / r0) - np.arcsinh(l_minus / r0), 0.0)
            angle = np.where(
                real & (r0 > 0),
                np.arctan2(p0 * l_plus, r0**2 + aw * r_plus) - np.arctan2(p0 * l_minus, r0**2 + aw * r_minus),
                0.0,
            )
        I += p0 * log_term
        solid_angle += angle
        if field:
            G += out * log_term[:, None]
    I -= aw * solid_angle
    if field:
        G += normal * (np.sign(w) * solid_angle)[:, None]
    return I, G


def _near_pairs(d: np.ndarray, size: np.ndarray):
    return np.nonzero(d < NEAR * size[None, :])


def interaction_matrix(panels: Panels) -> np.ndarray:
    """A[i, j]: potential at the centre of panel i from panel j with σ = 1 C/m²."""
    n = len(panels)
    A = np.empty((n, n))
    for start in range(0, n, CHUNK):
        rel = panels.centers[start : start + CHUNK, None, :] - panels.centers[None, :, :]
        d = np.linalg.norm(rel, axis=2)
        with np.errstate(divide="ignore"):
            block = K * panels.area[None, :] / d
        i, j = _near_pairs(d, panels.size)
        block[i, j] = K * polygon_terms(rel[i, j], panels.corners[j], panels.normals[j], field=False)[0]
        A[start : start + CHUNK] = block
    return A


@dataclass
class Clusters:
    members: np.ndarray  # (C, most) panel indices, -1 where a cluster has fewer
    centers: np.ndarray  # (C, 3)
    radius: np.ndarray  # (C,) reach of the cluster's panels from its centre


def make_clusters(panels: Panels, groups: Sequence[np.ndarray]) -> Clusters:
    most = max(len(g) for g in groups)
    members = np.full((len(groups), most), -1)
    for c, g in enumerate(groups):
        members[c, : len(g)] = g
    centers = np.array([panels.centers[g].mean(axis=0) for g in groups])
    radius = np.array(
        [(np.linalg.norm(panels.centers[g] - centers[c], axis=1) + panels.size[g]).max() for c, g in enumerate(groups)]
    )
    return Clusters(members, centers, radius)


def _blocks(first: int, rows: int, cols: int, block: int) -> list[np.ndarray]:
    """Indices of a rows x cols panel grid (row-major from first), in blocks of block x block."""
    index = first + np.arange(rows * cols).reshape(rows, cols)
    return [index[r : r + block, c : c + block].ravel() for r in range(0, rows, block) for c in range(0, cols, block)]


def _accumulate(out: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
    if out.ndim == 1:
        out += np.bincount(index, weights=values, minlength=len(out))
    else:
        for k in range(out.shape[1]):
            out[:, k] += np.bincount(index, weights=values[:, k], minlength=len(out))


def _panel_terms(panels: Panels, sigma: np.ndarray, rel: np.ndarray, j: np.ndarray, field: bool, potential: bool):
    """V and E of panels j at offsets rel from their centres: rectangles if near, point charges if not.

    Never both for the same pair: a point charge's huge term right next to its
    panel would not cancel exactly.
    """
    d = np.linalg.norm(rel, axis=1)
    near = d < NEAR * panels.size[j]
    q = sigma[j] * panels.area[j]
    inv = np.where(near, 0.0, 1.0 / np.where(near, 1.0, d))
    jn = j[near]
    I, G = polygon_terms(rel[near], panels.corners[jn], panels.normals[jn], field)
    V = E = None
    if potential:
        V = q * inv
        V[near] = sigma[jn] * I
    if field:
        E = (q * inv**3)[:, None] * rel
        E[near] = G * sigma[jn][:, None]
    return V, E


def evaluate(
    panels: Panels,
    sigma: np.ndarray,
    points: np.ndarray,
    field: bool = True,
    potential: bool = True,
    clusters: Clusters | None = None,
):
    """V (M,) and E (M, 3) of the charged panels at the points; None for what is not asked.

    Without clusters every panel counts one by one (slow, exact up to NEAR).
    """
    m = len(points)
    V = np.zeros(m) if potential else None
    E = np.zeros((m, 3)) if field else None
    if clusters is None:
        groups = Clusters(np.arange(len(panels))[None, :], np.zeros((1, 3)), np.array([np.inf]))
    else:
        groups = clusters
    q = sigma * panels.area
    valid = groups.members >= 0
    safe = np.where(valid, groups.members, 0)
    charge = (q[safe] * valid).sum(axis=1)
    dipole = np.einsum("cm,cmk->ck", q[safe] * valid, panels.centers[safe] - groups.centers[:, None, :])
    for start in range(0, m, CHUNK):
        p = points[start : start + CHUNK]
        n = len(p)
        rel = p[:, None, :] - groups.centers[None, :, :]
        d = np.linalg.norm(rel, axis=2)
        near = d < CLUSTER_NEAR * groups.radius[None, :]
        # far clusters: a point charge and a dipole at the centre
        inv = np.where(near, 0.0, 1.0 / np.where(near, 1.0, d))
        p_dot_r = np.einsum("nck,ck->nc", rel, dipole)
        if potential:
            V[start : start + n] = K * (charge * inv + p_dot_r * inv**3).sum(axis=1)
        if field:
            E[start : start + n] = K * (
                np.einsum("nc,nck->nk", charge * inv**3 + 3 * p_dot_r * inv**5, rel) - (inv**3) @ dipole
            )
        # near clusters: their panels one by one
        point, cluster = np.nonzero(near)
        if point.size == 0:
            continue
        j = groups.members[cluster]
        keep = j >= 0
        point = np.broadcast_to(point[:, None], j.shape)[keep]
        j = j[keep]
        v, e = _panel_terms(panels, sigma, p[point] - panels.centers[j], j, field, potential)
        if potential:
            _accumulate(V[start : start + n], point, K * v)
        if field:
            _accumulate(E[start : start + n], point, K * e)
    return V, E


class Solver:
    """The panels of a set of conductors, with the LU factors of their interaction matrix.

    Conductors are numbered plates first, then spheres.
    """

    def __init__(self, plates: Sequence[Plate], spheres: Sequence[Sphere]):
        self.plates = list(plates)
        self.spheres = list(spheres)
        parts, groups, first = [], [], 0
        self.layouts: list[dict | None] = []  # for each plate, where its faces' panels are (None: a sheet)
        for k, plate in enumerate(self.plates):
            if isinstance(plate, Ring):
                panels, layout = ring_panels(plate, k)
                for start, shape in layout.values():
                    groups += _blocks(first + start, *shape, PLATE_BLOCK)
                self.layouts.append({name: (first + start, shape) for name, (start, shape) in layout.items()})
            elif plate.thickness > 0:
                panels, layout = box_panels(plate, k)
                for start, shape in layout.values():
                    groups += _blocks(first + start, *shape, PLATE_BLOCK)
                self.layouts.append({name: (first + start, shape) for name, (start, shape) in layout.items()})
            else:
                panels = plate_panels(plate, k)
                groups += _blocks(first, PLATE_PANELS, PLATE_PANELS, PLATE_BLOCK)
                self.layouts.append(None)
            parts.append(panels)
            first += len(panels)
        for k, sphere in enumerate(self.spheres):
            parts.append(sphere_panels(sphere, len(self.plates) + k))
            for _face in range(6):
                groups += _blocks(first, SPHERE_PANELS, SPHERE_PANELS, SPHERE_BLOCK)
                first += SPHERE_PANELS**2
        self.panels = _join(parts)
        self.lu = lu_factor(interaction_matrix(self.panels), check_finite=False)
        self.clusters = make_clusters(self.panels, groups)

    def solve(self, voltages: Sequence[float], neutral: Sequence[int] = ()) -> "ConductorSolution":
        """The charges for the conductors at these potentials (one per conductor, plates first).

        neutral: conductors whose potentials all shift by the same amount, so
        that their charges add up to zero (a capacitor's plates: equal and
        opposite charges, the potential difference unchanged).
        """
        voltages = np.asarray(voltages, dtype=float)
        sigma = lu_solve(self.lu, voltages[self.panels.owner], check_finite=False)
        if len(neutral):
            mine = np.isin(self.panels.owner, neutral)
            unit = lu_solve(self.lu, mine.astype(float), check_finite=False)  # all of them at 1 V, the rest at 0
            shift = -(sigma * self.panels.area)[mine].sum() / (unit * self.panels.area)[mine].sum()
            sigma = sigma + shift * unit
            voltages = voltages.copy()
            voltages[list(neutral)] += shift
        return ConductorSolution(self, sigma, voltages)


class ConductorSolution:
    """The charge on every panel, and the field it makes everywhere."""

    def __init__(self, solver: Solver, sigma: np.ndarray, voltages: np.ndarray):
        self.solver = solver
        self.panels = solver.panels
        self.sigma = sigma
        self.voltages = voltages
        self.faces = self._plate_faces()

    def field(self, points: np.ndarray) -> np.ndarray:
        return evaluate(self.panels, self.sigma, points, potential=False, clusters=self.solver.clusters)[1]

    def potential(self, points: np.ndarray) -> np.ndarray:
        return evaluate(self.panels, self.sigma, points, field=False, clusters=self.solver.clusters)[0]

    def charge(self, conductor: int) -> float:
        """Total charge of a conductor, C."""
        mine = self.panels.owner == conductor
        return float((self.sigma[mine] * self.panels.area[mine]).sum())

    def signed_charge(self, conductor: int, sign: int) -> float:
        """The charge of a conductor's positive (sign +1) or negative (-1) parts, as a positive number."""
        mine = self.panels.owner == conductor
        area = self.panels.area[mine]
        if conductor < len(self.solver.plates) and self.solver.layouts[conductor] is None:
            inner, outer = self.faces[conductor]
            densities = np.concatenate((inner.ravel(), outer.ravel()))
            area = np.concatenate((area, area))
        else:
            densities = self.sigma[mine]
        return float((np.maximum(sign * densities, 0.0) * area).sum())

    def sphere_sigma(self, k: int) -> np.ndarray:
        """σ on the panels of sphere k, in the order of sphere_mesh's quads."""
        return self.sigma[self.panels.owner == len(self.solver.plates) + k]

    def face(self, plate: int, name: str) -> np.ndarray:
        """σ on one face of a box plate ("w+", "w-", "u+", "u-", "v+", "v-") or of a ring ("w+", "w-",
        "out", "in"), as its grid of panels."""
        start, shape = self.solver.layouts[plate][name]
        return self.sigma[start : start + shape[0] * shape[1]].reshape(shape)

    def _plate_faces(self) -> list[tuple[np.ndarray, np.ndarray] | None]:
        """σ on each rectangular plate's inner face (towards the origin, where the other plate is) and outer face.

        A box plate has its faces' own panels. On a sheet, each face carries σ/2 ± ε₀ E·n,
        with E the field at the panel's centre: a panel's own field has no component across it there.
        A ring has None here: its faces are in face().
        """
        faces = []
        for k, plate in enumerate(self.solver.plates):
            if isinstance(plate, Ring):
                faces.append(None)
                continue
            if self.solver.layouts[k] is not None:
                plus_side = plate.center @ plate.from_local(np.array([[0.0, 0.0, 1.0]]))[0] > 0
                inner, outer = ("w-", "w+") if plus_side else ("w+", "w-")
                faces.append((self.face(k, inner), self.face(k, outer)))
                continue
            mine = np.flatnonzero(self.panels.owner == k)
            normal = self.panels.normals[mine[0]]
            inward = -normal if plate.center @ normal > 0 else normal
            across = EPS0 * self.field(self.panels.centers[mine]) @ inward
            half = self.sigma[mine] / 2
            shape = (PLATE_PANELS, PLATE_PANELS)
            faces.append(((half + across).reshape(shape), (half - across).reshape(shape)))
        return faces


def geometry_key(plates: Sequence[Plate], spheres: Sequence[Sphere]) -> tuple:
    """What the interaction matrix depends on (not the potentials)."""
    return tuple(p.key() for p in plates), tuple((tuple(np.round(s.center, 9)), round(s.radius, 9)) for s in spheres)


def touching(plates: Sequence[Plate], spheres: Sequence[Sphere], gap: float = 1e-3) -> bool:
    """Whether a sphere comes within gap of another sphere or of a plate."""
    for a, s in enumerate(spheres):
        for t in spheres[a + 1 :]:
            if np.linalg.norm(s.center - t.center) < s.radius + t.radius + gap:
                return True
        for plate in plates:
            if plate.distance(s.center) < s.radius + gap:
                return True
    return False
