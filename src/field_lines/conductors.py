"""Conductors held at fixed potentials, alone in empty space: capacitor plates and metallic spheres.

A conductor's charge sits on its surface. Each surface is cut into small flat
panels, each with a uniform surface charge density, and the densities are
chosen so that the centre of every panel is at its conductor's potential (the
method of moments, or boundary elements). That solves Laplace's equation
outside the conductors with V -> 0 far away, without a grid.

- A plate is a thin rectangle cut into PLATE_PANELS x PLATE_PANELS panels,
  finer towards its edges, where the charge piles up. Its two faces are told
  apart afterwards: each carries σ/2 plus or minus ε₀ times the field of all
  the other panels across it.
- A sphere is a cube projected onto it: 6 faces of SPHERE_PANELS x
  SPHERE_PANELS panels.

The potential or the field of a panel at a point is that of a uniformly
charged rectangle when the point is within NEAR panel sizes, and that of a
point charge farther away. The same rule builds the matrix of the solve and
evaluates the field afterwards, so the potentials come out as asked.

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

from field_lines.sources import Plate, rectangle_field, rectangle_potential
from field_lines.units import EPS0, K

PLATE_PANELS = 25  # per side of a plate; odd, so that a row of panels lies on each centre line
SPHERE_PANELS = 8  # per edge of each of the six faces of the cube
NEAR = 4.0  # closer than this many panel sizes, a panel is a rectangle, not a point charge
CHUNK = 1000  # points per block of point-panel pairs
PLATE_BLOCK = 5  # a plate's clusters are blocks of 5 x 5 panels
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
    frames: np.ndarray  # (N, 3, 3): rows e_u, e_v and the normal n
    half_u: np.ndarray  # (N,) half sides along e_u and e_v
    half_v: np.ndarray
    owner: np.ndarray  # (N,) index of the conductor the panel belongs to

    @property
    def area(self) -> np.ndarray:
        return 4 * self.half_u * self.half_v

    @property
    def size(self) -> np.ndarray:
        return 2 * np.maximum(self.half_u, self.half_v)

    def __len__(self) -> int:
        return len(self.centers)


def _join(parts: Sequence[Panels]) -> Panels:
    fields = ("centers", "frames", "half_u", "half_v", "owner")
    return Panels(*(np.concatenate([getattr(p, f) for p in parts]) for f in fields))


def panel_edges(half: float, n: int) -> np.ndarray:
    """n + 1 edges from -half to half, closer together towards the ends (cosine spacing)."""
    return -half * np.cos(np.linspace(0.0, np.pi, n + 1))


def plate_face_grid(plate: Plate, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The panel centres of a plate along its u and v, as plate_panels lays them out."""
    n = n or PLATE_PANELS
    eu, ev = panel_edges(plate.width / 2, n), panel_edges(plate.height / 2, n)
    return (eu[1:] + eu[:-1]) / 2, (ev[1:] + ev[:-1]) / 2


def plate_panels(plate: Plate, owner: int, n: int | None = None) -> Panels:
    n = n or PLATE_PANELS
    eu, ev = panel_edges(plate.width / 2, n), panel_edges(plate.height / 2, n)
    cu, cv = plate_face_grid(plate, n)
    uu, vv = np.meshgrid(cu, cv, indexing="ij")
    local = np.column_stack((uu.ravel(), vv.ravel(), np.zeros(uu.size)))
    axes = plate.from_local(np.eye(3))  # rows: the global directions of local u, v, w
    hu, hv = np.meshgrid(np.diff(eu) / 2, np.diff(ev) / 2, indexing="ij")
    return Panels(
        plate.from_local(local) + plate.center,
        np.repeat(axes[None], uu.size, axis=0),
        hu.ravel(),
        hv.ravel(),
        np.full(uu.size, owner),
    )


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
    points, quads = sphere_mesh(sphere.center, sphere.radius, m)
    corners = points[quads]  # (n, 4, 3)
    normal = corners.mean(axis=1) - sphere.center
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    e_u = corners[:, 1] - corners[:, 0]
    e_u -= np.einsum("ij,ij->i", e_u, normal)[:, None] * normal
    e_u /= np.linalg.norm(e_u, axis=1, keepdims=True)
    e_v = np.cross(normal, e_u)
    flat = 0.5 * np.linalg.norm(np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1]), axis=1)
    area = flat * 4 * np.pi * sphere.radius**2 / flat.sum()  # the whole sphere, shared as the flat quads share it
    half = np.sqrt(area) / 2  # each panel a square of its area
    return Panels(
        sphere.center + sphere.radius * normal,
        np.stack((e_u, e_v, normal), axis=1),
        half,
        half.copy(),
        np.full(len(quads), owner),
    )


def _local(rel: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.einsum("pk,pak->pa", rel, frames)


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
        local = _local(rel[i, j], panels.frames[j])
        block[i, j] = K * rectangle_potential(local[:, 0], local[:, 1], local[:, 2], panels.half_u[j], panels.half_v[j])
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
    local = _local(rel[near], panels.frames[jn])
    args = (local[:, 0], local[:, 1], local[:, 2], panels.half_u[jn], panels.half_v[jn])
    V = E = None
    if potential:
        V = q * inv
        V[near] = sigma[jn] * rectangle_potential(*args)
    if field:
        E = (q * inv**3)[:, None] * rel
        E[near] = np.einsum("pa,pak->pk", rectangle_field(*args), panels.frames[jn]) * sigma[jn][:, None]
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
        parts = [plate_panels(plate, k) for k, plate in enumerate(self.plates)]
        parts += [sphere_panels(sphere, len(self.plates) + k) for k, sphere in enumerate(self.spheres)]
        self.panels = _join(parts)
        self.lu = lu_factor(interaction_matrix(self.panels), check_finite=False)
        groups = []
        first = 0
        for _ in self.plates:
            groups += _blocks(first, PLATE_PANELS, PLATE_PANELS, PLATE_BLOCK)
            first += PLATE_PANELS**2
        for _ in self.spheres:
            for _face in range(6):
                groups += _blocks(first, SPHERE_PANELS, SPHERE_PANELS, SPHERE_BLOCK)
                first += SPHERE_PANELS**2
        self.clusters = make_clusters(self.panels, groups)

    def solve(self, voltages: Sequence[float]) -> "ConductorSolution":
        """The charges for the conductors at these potentials (one per conductor, plates first)."""
        voltages = np.asarray(voltages, dtype=float)
        sigma = lu_solve(self.lu, voltages[self.panels.owner], check_finite=False)
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
        if conductor < len(self.solver.plates):
            inner, outer = self.faces[conductor]
            densities = np.concatenate((inner.ravel(), outer.ravel()))
            area = np.concatenate((area, area))
        else:
            densities = self.sigma[mine]
        return float((np.maximum(sign * densities, 0.0) * area).sum())

    def sphere_sigma(self, k: int) -> np.ndarray:
        """σ on the panels of sphere k, in the order of sphere_mesh's quads."""
        return self.sigma[self.panels.owner == len(self.solver.plates) + k]

    def _plate_faces(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """σ on each plate's inner face (towards the origin, where the other plate is) and outer face.

        Each face carries σ/2 ± ε₀ E·n, with E the field at the panel's centre:
        a panel's own field has no component across it there.
        """
        faces = []
        for k, plate in enumerate(self.solver.plates):
            mine = np.flatnonzero(self.panels.owner == k)
            normal = self.panels.frames[mine[0], 2]
            inward = -normal if plate.center @ normal > 0 else normal
            across = EPS0 * self.field(self.panels.centers[mine]) @ inward
            half = self.sigma[mine] / 2
            shape = (PLATE_PANELS, PLATE_PANELS)
            faces.append(((half + across).reshape(shape), (half - across).reshape(shape)))
        return faces


def geometry_key(plates: Sequence[Plate], spheres: Sequence[Sphere]) -> tuple:
    """What the interaction matrix depends on (not the potentials)."""
    return (
        tuple((tuple(np.round(p.center, 9)), round(p.width, 9), round(p.height, 9), p.normal) for p in plates),
        tuple((tuple(np.round(s.center, 9)), round(s.radius, 9)) for s in spheres),
    )


def touching(plates: Sequence[Plate], spheres: Sequence[Sphere], gap: float = 1e-3) -> bool:
    """Whether a sphere comes within gap of another sphere or of a plate."""
    for a, s in enumerate(spheres):
        for t in spheres[a + 1 :]:
            if np.linalg.norm(s.center - t.center) < s.radius + t.radius + gap:
                return True
        for plate in plates:
            local = plate.to_local(s.center[None])[0]
            half = np.array([plate.width / 2, plate.height / 2, 0.0])
            if np.linalg.norm(local - np.clip(local, -half, half)) < s.radius + gap:
                return True
    return False
