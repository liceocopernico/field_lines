"""Capacitor plates as conductors at fixed potentials: Laplace's equation on a grid.

The plates are centred on the origin across their normal axis w, the positive
one at w = +d/2. In local coordinates (u, v, w), as for sources.Plate, the
potential is even in u and v and odd in w, so only the octant u, v, w >= 0 is
solved:

- w = 0 is the plane halfway between the plates: V = 0 there;
- u = 0 and v = 0 are mirror planes: no field crosses them;
- the positive plate is a one-node-thick layer at V = +1/2 (for ΔV = 1 V);
- the box around everything, BOX_MARGIN plate sides away, is held at the
  potential that the plates' charges would make in empty space. Those
  charges are not known in advance: the box starts at V = 0, and is updated
  from the charges found a few times (OPEN_BOUNDARY_PASSES). So the capacitor
  is alone in space, and outside the box its field is the field of those
  charges, which is where field lines go when they loop far out.

The grid is rectilinear: uniform and fine (spacing h) over the plate and the
gap and a little beyond the edges, then growing geometrically towards the
box, where nothing changes quickly. The Laplacian is written in finite
volumes, as a sum over the grid's edges of (face area / edge length) times
the squared difference of V; on the mirror planes the cells are half cells.
The matrix is symmetric and is solved by conjugate gradients. The charge of
a plate node is ε₀ (A V) there: ε₀ times the flux of E out of its cell.

E = -∇V is taken on the nodes by central differences and interpolated, except
in the cells next to a plate: there a central difference would mix the strong
field on one side of the plate with the weak field on the other, so E is the
gradient of V interpolated inside that cell, one-sided by construction.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import cg

from field_lines.sources import AXES
from field_lines.units import EPS0, K

CELLS_IN_GAP = 10  # at least, across the whole gap
CELLS_ON_SIDE = 40  # at least, across a whole plate side
GROWTH = 1.12  # ratio of neighbouring cells outside the fine region
BOX_MARGIN = 3.0  # plate sides between the plates and the box
OPEN_BOUNDARY_PASSES = 1  # one update is enough: C then changes by less than 1e-5
FAR_BLOCKS = 12  # outside the box the quarter plate's charges are merged into at most 12 x 12 blocks
MAX_UNKNOWNS = 600_000  # about 10 s at most
TOLERANCE = 1e-6  # C does not change in its first 5 digits from 1e-8


def spacing(width: float, height: float, separation: float, refine: float = 1.0) -> float:
    """The fine grid spacing h (refine > 1 makes it finer, for convergence checks)."""
    return min(separation / CELLS_IN_GAP, max(width, height) / CELLS_ON_SIDE) / refine


def axis_nodes(edge: float, fine_until: float, end: float, h: float) -> np.ndarray:
    """Nodes from 0 to end: spacing near h with a node exactly on edge, up to fine_until, then growing."""
    cells = max(1, math.ceil(edge / h))
    step = edge / cells
    nodes = list(np.arange(cells + 1) * step)
    while nodes[-1] < fine_until:
        nodes.append(nodes[-1] + step)
    gap = step
    while nodes[-1] < end:
        gap *= GROWTH
        nodes.append(min(nodes[-1] + gap, end) if end - nodes[-1] > 0.5 * gap else end)
        if nodes[-1] == end:
            break
    return np.array(nodes)


def grid_axes(width: float, height: float, separation: float, refine: float = 1.0):
    h = spacing(width, height, separation, refine)
    side = max(width, height)
    end = separation / 2 + BOX_MARGIN * side
    margin = max(separation, 4 * h)  # the fringe field near the edges needs the fine spacing too
    u = axis_nodes(width / 2, width / 2 + margin, end, h)
    v = axis_nodes(height / 2, height / 2 + margin, end, h)
    w = axis_nodes(separation / 2, separation / 2 + margin, end, h)
    return u, v, w


def unknowns(width: float, height: float, separation: float) -> int:
    return math.prod(len(a) for a in grid_axes(width, height, separation))


def min_separation(width: float, height: float) -> float:
    """The smallest gap whose grid fits in MAX_UNKNOWNS."""
    lo, hi = 1e-4 * max(width, height), max(width, height)
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if unknowns(width, height, mid) > MAX_UNKNOWNS else (lo, mid)
    return hi


@dataclass
class OctantGrid:
    """The solution for ΔV = 1 V in the octant: V and E on the nodes, and the plate's charge."""

    width: float
    height: float
    separation: float
    spacing: float  # the fine spacing h
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]  # node coordinates along u, v, w
    values: np.ndarray  # (nu, nv, nw, 4): V, Eu, Ev, Ew
    capacitance: float  # F
    sigma: np.ndarray  # charge density on the plate nodes of a quarter plate, C/m² for 1 V
    plate_nodes: tuple[int, int, int]  # (ip, jp, kp): the plate's edge nodes along u and v, its layer along w
    charge_positions: np.ndarray  # (n, 3) local positions of the node charges of both whole plates
    charges: np.ndarray  # (n,) their charges for 1 V, C

    def __post_init__(self):
        self.interpolator = RegularGridInterpolator(self.axes, self.values, bounds_error=False, fill_value=0.0)
        u, v, w = self.axes
        ip, jp, kp = self.plate_nodes
        self._near_plate = (u[ip + 1], v[jp + 1], w[kp - 1], w[kp + 1])

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """V, Eu, Ev, Ew for 1 V at octant points (u, v, w >= 0)."""
        values = self.interpolator(points)
        u_max, v_max, w_lo, w_hi = self._near_plate
        u, v, w = points.T
        near = (u < u_max) & (v < v_max) & (w > w_lo) & (w < w_hi)
        if near.any():
            values[near, 1:] = -_cell_gradient(self.axes, self.values[..., 0], points[near])
        values[u == 0, 1] = 0.0  # no field across the mirror planes, by symmetry
        values[v == 0, 2] = 0.0
        return values


def solve(width: float, height: float, separation: float, refine: float = 1.0) -> OctantGrid:
    axes = grid_axes(width, height, separation, refine)
    shape = tuple(len(a) for a in axes)
    if math.prod(shape) > MAX_UNKNOWNS * refine**3:
        minimum = min_separation(width, height)
        raise ValueError(f"separation {separation} m needs too fine a grid (minimum {minimum:.4g} m)")
    u, v, w = axes
    ip = int(np.searchsorted(u, width / 2 - 1e-12))
    jp = int(np.searchsorted(v, height / 2 - 1e-12))
    kp = int(np.searchsorted(w, separation / 2 - 1e-12))

    A = _laplacian(axes)
    box = np.zeros(shape, dtype=bool)
    box[-1, :, :] = box[:, -1, :] = box[:, :, -1] = True
    box[:, :, 0] = False  # the midplane stays at V = 0
    fixed = box.copy()
    fixed[:, :, 0] = True
    fixed[: ip + 1, : jp + 1, kp] = True
    V = np.zeros(shape)
    V[: ip + 1, : jp + 1, kp] = 0.5
    free = ~fixed.ravel()
    A_free = A[free][:, free].tocsr()
    A_fixed = A[free][:, ~free].tocsr()
    jacobi = sparse.diags(1 / A_free.diagonal())
    box_points = np.stack(np.meshgrid(u, v, w, indexing="ij"), axis=-1)[box]
    x = None
    for passes_left in range(OPEN_BOUNDARY_PASSES, -1, -1):
        rhs = -(A_fixed @ V.ravel()[~free])
        x, info = cg(A_free, rhs, x0=x, rtol=TOLERANCE, maxiter=10 * sum(shape), M=jacobi)
        if info != 0:
            raise RuntimeError(f"the Laplace solver did not converge (cg info {info})")
        V.ravel()[free] = x
        node_charge = EPS0 * (A @ V.ravel()).reshape(shape)[: ip + 1, : jp + 1, kp]
        positions, charges = _whole_plates(*_blocks(u[: ip + 1], v[: jp + 1], node_charge), w[kp])
        if passes_left:
            V[box] = potential_of_charges(box_points, positions, charges)

    charge = 4 * node_charge.sum()  # the octant holds a quarter of the positive plate
    cell_u, cell_v = _cell_widths(u)[: ip + 1], _cell_widths(v)[: jp + 1]
    sigma = node_charge / np.outer(cell_u, cell_v)

    Eu, Ev, Ew = (-g for g in np.gradient(V, u, v, w))
    Eu[0], Ev[:, 0] = 0.0, 0.0  # exactly, by symmetry (np.gradient is one-sided there)
    values = np.stack((V, Eu, Ev, Ew), axis=-1)
    h = spacing(width, height, separation, refine)
    return OctantGrid(
        width,
        height,
        separation,
        h,
        axes,
        values,
        capacitance=charge,
        sigma=sigma,
        plate_nodes=(ip, jp, kp),
        charge_positions=positions,
        charges=charges,
    )


def _blocks(u: np.ndarray, v: np.ndarray, node_charge: np.ndarray):
    """The quarter plate's node charges merged into at most FAR_BLOCKS x FAR_BLOCKS blocks.

    Each block's charge sits at the block's centre of charge, which keeps its
    charge and its dipole moment: seen from the box, three plate sides away,
    that is all that matters.
    """
    bu, bv, q = [], [], []
    for rows in np.array_split(np.arange(len(u)), min(FAR_BLOCKS, len(u))):
        for cols in np.array_split(np.arange(len(v)), min(FAR_BLOCKS, len(v))):
            block = node_charge[np.ix_(rows, cols)]
            total = block.sum()
            bu.append((block.sum(axis=1) @ u[rows]) / total)
            bv.append((block.sum(axis=0) @ v[cols]) / total)
            q.append(total)
    return np.array(bu), np.array(bv), np.array(q)


def _whole_plates(u: np.ndarray, v: np.ndarray, charge: np.ndarray, w_plate: float):
    """Charges of a quarter plate at (u, v), mirrored into both whole plates (the - plate with -q).

    A charge on a mirror plane is the octant's share of its cell, so its two
    mirror images together make the whole charge.
    """
    positions, charges = [], []
    for su in (1, -1):
        for sv in (1, -1):
            for sw in (1, -1):
                positions.append(np.column_stack((su * u, sv * v, np.full(len(u), sw * w_plate))))
                charges.append(sw * charge)
    return np.concatenate(positions), np.concatenate(charges)


def potential_of_charges(points: np.ndarray, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """V of point charges in empty space, in chunks of points to bound memory."""
    V = np.empty(len(points))
    for start in range(0, len(points), 256):
        r = points[start : start + 256, None, :] - positions[None, :, :]
        V[start : start + 256] = K * (charges / np.linalg.norm(r, axis=2)).sum(axis=1)
    return V


def field_of_charges(points: np.ndarray, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
    E = np.empty((len(points), 3))
    for start in range(0, len(points), 256):
        r = points[start : start + 256, None, :] - positions[None, :, :]
        d = np.linalg.norm(r, axis=2)
        E[start : start + 256] = K * np.einsum("mn,mnk->mk", charges / d**3, r)
    return E


def _cell_gradient(axes, V: np.ndarray, points: np.ndarray) -> np.ndarray:
    """∇ of the trilinear interpolation of V, inside the cell that holds each point."""
    index, t, size = [], [], []
    for axis, x in enumerate(axes):
        i = np.clip(np.searchsorted(x, points[:, axis]) - 1, 0, len(x) - 2)
        dx = x[i + 1] - x[i]
        index.append(i)
        t.append(np.clip((points[:, axis] - x[i]) / dx, 0.0, 1.0))
        size.append(dx)
    i, j, k = index

    def corner(a, b, c):
        return V[i + a, j + b, k + c]

    tu, tv, tw = t
    grad = np.empty((len(points), 3))
    # along u: the difference across the cell, blended over v and w; likewise for v and w
    grad[:, 0] = sum(
        (corner(1, b, c) - corner(0, b, c)) * (tv if b else 1 - tv) * (tw if c else 1 - tw) for b in (0, 1) for c in (0, 1)
    ) / size[0]
    grad[:, 1] = sum(
        (corner(a, 1, c) - corner(a, 0, c)) * (tu if a else 1 - tu) * (tw if c else 1 - tw) for a in (0, 1) for c in (0, 1)
    ) / size[1]
    grad[:, 2] = sum(
        (corner(a, b, 1) - corner(a, b, 0)) * (tu if a else 1 - tu) * (tv if b else 1 - tv) for a in (0, 1) for b in (0, 1)
    ) / size[2]
    return grad


def _cell_widths(x: np.ndarray) -> np.ndarray:
    """Width of each node's cell: half the edges on both sides (a half cell at the ends)."""
    edges = np.diff(x)
    return np.concatenate(([edges[0] / 2], (edges[:-1] + edges[1:]) / 2, [edges[-1] / 2]))


def _laplacian(axes) -> sparse.csr_matrix:
    """Σ over edges of (face area / length) (x_a - x_b)², on the rectilinear grid."""
    ops, cells = [], []
    for x in axes:
        n = len(x)
        D = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
        ops.append((D.T @ sparse.diags(1 / np.diff(x)) @ D).tocsr())
        cells.append(sparse.diags(_cell_widths(x)))
    (Lu, Lv, Lw), (Cu, Cv, Cw) = ops, cells
    kron = sparse.kron
    return (kron(kron(Lu, Cv), Cw) + kron(kron(Cu, Lv), Cw) + kron(kron(Cu, Cv), Lw)).tocsr()


class ConductorSolution:
    """The capacitor's field at a given ΔV and orientation, from the octant solution."""

    def __init__(self, grid: OctantGrid, voltage: float, normal: str = "z"):
        self.grid = grid
        self.voltage = voltage
        self.normal = normal
        a = AXES.index(normal)
        self.local_axes = [(a + 1) % 3, (a + 2) % 3, a]

    @property
    def spacing(self) -> float:
        return self.grid.spacing

    @property
    def capacitance(self) -> float:
        return self.grid.capacitance

    @property
    def charge(self) -> float:
        """Charge of the positive plate (the plate at +d/2 along the normal), C."""
        return self.grid.capacitance * self.voltage

    def _split(self, points: np.ndarray):
        """Local coordinates, and which points lie outside the grid's box (there: the plates' charges)."""
        local = points[:, self.local_axes]
        box_end = np.array([axis[-1] for axis in self.grid.axes])
        return local, np.any(np.abs(local) > box_end, axis=1)

    def potential(self, points: np.ndarray) -> np.ndarray:
        local, outside = self._split(points)
        V = np.empty(len(points))
        inside = ~outside
        octant, sw = np.abs(local[inside]), np.where(local[inside, 2] < 0, -1.0, 1.0)
        V[inside] = sw * self.grid.interpolator(octant)[:, 0]
        V[outside] = potential_of_charges(local[outside], self.grid.charge_positions, self.grid.charges)
        return V * self.voltage

    def field(self, points: np.ndarray) -> np.ndarray:
        local, outside = self._split(points)
        E = np.empty_like(local)
        inside = ~outside
        octant = np.abs(local[inside])
        su, sv, sw = np.where(local[inside] < 0, -1.0, 1.0).T
        values = self.grid.evaluate(octant)
        E[inside] = np.column_stack((sw * su * values[:, 1], sw * sv * values[:, 2], values[:, 3]))
        E[outside] = field_of_charges(local[outside], self.grid.charge_positions, self.grid.charges)
        E[local[:, 0] == 0, 0] = 0.0  # no field across the mirror planes, by symmetry
        E[local[:, 1] == 0, 1] = 0.0
        out = np.empty_like(E)
        out[:, self.local_axes] = E * self.voltage
        return out
