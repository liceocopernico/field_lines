"""What is being shown: the sources, 2D or 3D, the model. No Qt in here.

Two models:
- "charges": point charges and a capacitor whose plates carry a uniform σ
  (the field is the sum of theirs);
- "conductors": metallic spheres and a capacitor whose plates are held at a
  fixed potential difference, with equal and opposite charges
  (conductors.ConductorSolution finds the charges).

The capacitor has rectangular or circular plates, or is cylindrical.
"""

import copy
import math
from dataclasses import dataclass

import numpy as np

from field_lines import conductors, seeding
from field_lines.conductors import ConductorSolution, Sphere
from field_lines.sources import AXES, FieldModel, Plate, PointCharge, Ring, Superposition
from field_lines.tracing import FieldLine, trace
from field_lines.units import CM, EPS0, MM, NC, UC_PER_M2

VIEW_HALF = 10 * CM  # the view shows the cube |x|, |y|, |z| <= VIEW_HALF
# Lines are followed far outside the view: a line leaving a capacitor's outer face makes a loop
# that can be metres wide before it comes back to the negative plate. Far away the step grows
# with the distance, so such a loop takes a few hundred steps.
TRACE_HALF = 100.0  # m
STEP_FAR = 0.05  # step / distance from the origin, far away
STEP_MAX = 2 * VIEW_HALF / 150
SEED_RADIUS = 0.35 * CM  # lines start this far from a point charge (inside its sphere) ...
STOP_RADIUS = 0.25 * CM  # ... and end this close to one
PLATE_OFFSET = 0.1 * CM  # lines start this far from a plate ...
SPHERE_OFFSET = 0.2 * CM  # ... and this far from a metallic sphere
MAX_LENGTH = 10 * TRACE_HALF


def charge_radius(q: float) -> float:
    """Radius of the sphere that shows a point charge."""
    return 0.4 * CM + 0.15 * CM * math.sqrt(abs(q) / NC)


SHAPES = ("rect", "disc", "cylinder")


@dataclass
class Capacitor:
    """Two plates, rectangular or circular, or two coaxial tubes, each with a thickness.

    Parallel plates are separation apart (between their inner faces) and grow
    outwards with the thickness. The cylinder's inner tube has its outer face
    at inner_radius and grows inwards (into a rod, if it is thicker than
    that); the outer tube has its inner face at outer_radius and grows
    outwards. The positive plate is on the + side of the axis, or inside.
    """

    width: float = 10 * CM
    height: float = 10 * CM
    separation: float = 3 * CM
    sigma: float = 0.2 * UC_PER_M2  # charges model: charge density of the positive plate (towards the other)
    voltage: float = 500.0  # conductors model: the potential difference between the plates
    normal: str = "z"  # the capacitor's axis
    thickness: float = 2 * MM  # of each plate
    shape: str = "rect"  # one of SHAPES: rectangular plates, circular plates, coaxial tubes
    radius: float = 5 * CM  # of the circular plates
    inner_radius: float = 2 * CM  # of the cylinder's inner tube (its outer face)
    outer_radius: float = 4 * CM  # of the outer tube (its inner face)
    length: float = 10 * CM  # of the tubes
    cut: str = "across"  # in 2D the cylinder is cut "across" its axis (two rings) or "along" it

    @property
    def area(self) -> float:
        """The area of the positive plate's face towards the other plate (its σ = Q / area)."""
        if self.shape == "disc":
            return math.pi * self.radius**2
        if self.shape == "cylinder":
            return 2 * math.pi * self.inner_radius * self.length
        return self.width * self.height

    def _offset(self, distance: float) -> np.ndarray:
        offset = np.zeros(3)
        offset[AXES.index(self.normal)] = distance
        return offset

    def plates(self, sigma: float | None = None) -> list:
        """The plates as charged sheets at their inner faces; the positive one with density sigma.

        (default: self.sigma). The negative one has the opposite charge: on
        the outer tube, a lower σ over its larger face.
        """
        sigma = self.sigma if sigma is None else sigma
        if self.shape == "cylinder":
            a, b, h = self.inner_radius, self.outer_radius, self.length / 2
            return [
                Ring(sigma, np.zeros(3), a, a, h, self.normal, gap="out"),
                Ring(-sigma * a / b, np.zeros(3), b, b, h, self.normal, gap="in"),
            ]
        offset = self._offset(self.separation / 2)
        if self.shape == "disc":
            return [
                Ring(sigma, offset, 0.0, self.radius, 0.0, self.normal, gap="w-"),
                Ring(-sigma, -offset, 0.0, self.radius, 0.0, self.normal, gap="w+"),
            ]
        return [
            Plate(sigma, offset, self.width, self.height, self.normal),
            Plate(-sigma, -offset, self.width, self.height, self.normal),
        ]

    def bodies(self, sigma: float | None = None) -> list:
        """The plates as bodies with their thickness (slabs, discs or tubes), with the sheets' σ."""
        sigma = self.sigma if sigma is None else sigma
        t = self.thickness
        if self.shape == "cylinder":
            a, b, h = self.inner_radius, self.outer_radius, self.length / 2
            return [
                Ring(sigma, np.zeros(3), max(a - t, 0.0), a, h, self.normal, gap="out"),
                Ring(-sigma * a / b, np.zeros(3), b, b + t, h, self.normal, gap="in"),
            ]
        offset = self._offset(self.separation / 2 + t / 2)
        if self.shape == "disc":
            return [
                Ring(sigma, offset, 0.0, self.radius, t / 2, self.normal, gap="w-"),
                Ring(-sigma, -offset, 0.0, self.radius, t / 2, self.normal, gap="w+"),
            ]
        return [
            Plate(sigma, offset, self.width, self.height, self.normal, t),
            Plate(-sigma, -offset, self.width, self.height, self.normal, t),
        ]

    def ideal_sigma(self) -> float:
        """σ of the positive plate of the ideal capacitor (no edges) at self.voltage."""
        if self.shape == "cylinder":
            a, b = self.inner_radius, self.outer_radius
            return EPS0 * self.voltage / (a * math.log(b / a))
        return EPS0 * self.voltage / self.separation

    def probes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where the readouts look: E at the centre of the gap, ΔV between these two points on the plates.

        Parallel plates: the centre, and the centres of the inner faces. The
        cylinder: halfway across the gap, and the two faces across it, in the
        middle of its length.
        """
        if self.shape == "cylinder":
            across = self.bodies(0.0)[0].from_local(np.array([[1.0, 0.0, 0.0]]))[0]  # a direction across the axis
            a, b = self.inner_radius, self.outer_radius
            return (a + b) / 2 * across, a * across, b * across
        offset = self._offset(self.separation / 2)
        return np.zeros(3), offset, -offset

    def ideal(self, charge: float) -> tuple[float, float]:
        """E at the probe and C of the ideal capacitor (no edges) with this charge."""
        if self.shape == "cylinder":
            a, b, L = self.inner_radius, self.outer_radius, self.length
            return abs(charge) / (2 * math.pi * EPS0 * L * (a + b) / 2), 2 * math.pi * EPS0 * L / math.log(b / a)
        return abs(charge) / (self.area * EPS0), EPS0 * self.area / self.separation


def capacitor_axis(capacitor: Capacitor, planar: bool) -> str:
    """The capacitor's axis: in 3D as it is; in 2D in the plane, so that the plane cuts the plates.

    A cylinder cut across its axis has its axis along z instead.
    """
    if not planar:
        return capacitor.normal
    if capacitor.shape == "cylinder" and capacitor.cut == "across":
        return "z"
    return "x" if capacitor.normal == "z" else capacitor.normal


@dataclass
class PlateDensity:
    """Surface charge density over a slab's faces, on grids of its local (u, v) coordinates."""

    plate: Plate  # the slab
    u: np.ndarray  # along the plate's width
    v: np.ndarray  # along its height
    inner: np.ndarray  # (len(u), len(v)) σ on the face towards the other plate, C/m²
    outer: np.ndarray  # σ on the face away from it
    rim_u: np.ndarray  # (2, len(v)) σ on the rims at u = -width/2 and u = +width/2, along v
    rim_v: np.ndarray  # (2, len(u)) σ on the rims at v = -height/2 and v = +height/2, along u

    def at(self, u: float, v: float) -> tuple[float, float]:
        """σ of the inner and outer face at a point of the plate (nearest grid node)."""
        i = int(np.argmin(np.abs(self.u - u)))
        j = int(np.argmin(np.abs(self.v - v)))
        return float(self.inner[i, j]), float(self.outer[i, j])

    def values(self) -> np.ndarray:
        return np.concatenate([a.ravel() for a in (self.inner, self.outer, self.rim_u, self.rim_v)])


def ring_face_names(ring: Ring) -> list[str]:
    """A solid ring's faces: the flat ones at w = ±half_length, the outer cylinder and a tube's inner one."""
    names = ["w+", "w-"] if ring.r_out > ring.r_in else []
    if ring.half_length > 0:
        names += ["out"] + (["in"] if ring.r_in > 0 else [])
    return names


@dataclass
class RingDensity:
    """Surface charge density over a disc's or a tube's faces, on grids along and around each face.

    faces: for each face (see ring_face_names), the positions along it (ρ on
    a flat face, w on a cylinder) and σ there, (len(along), len(phi)) in C/m².
    phi is closed: its last angle is the first one plus 2π.
    """

    ring: Ring
    phi: np.ndarray
    faces: dict[str, tuple[np.ndarray, np.ndarray]]

    def values(self) -> np.ndarray:
        return np.concatenate([sigma.ravel() for _, sigma in self.faces.values()])


def ring_angles() -> np.ndarray:
    """The angles around a ring where its σ is known (its panels' centres), closed."""
    phi = (np.arange(conductors.RING_SEGMENTS) + 0.5) * 2 * np.pi / conductors.RING_SEGMENTS
    return np.append(phi, phi[0] + 2 * np.pi)


def uniform_density(body) -> "PlateDensity | RingDensity":
    """A plate's or a ring's density, the same σ (its own) on every face."""
    if isinstance(body, Ring):
        phi = ring_angles()
        faces = {}
        for name in ring_face_names(body):
            along = [body.r_in, body.r_out] if name in ("w+", "w-") else [-body.half_length, body.half_length]
            faces[name] = (np.array(along), np.full((2, len(phi)), body.sigma))
        return RingDensity(body, phi, faces)
    a, b = body.width / 2, body.height / 2
    same = np.full((2, 2), body.sigma)
    return PlateDensity(body, np.array([-a, a]), np.array([-b, b]), same, same.copy(), same.copy(), same.copy())


class Scene:
    def __init__(self):
        self.kind = "charges"  # the model: "charges" or "conductors"
        self.charges: list[PointCharge] = []
        self.spheres: list[Sphere] = []
        self.capacitor: Capacitor | None = None
        self.planar = False
        self.lines_per_nc = 10.0
        self.click_seeds: list[np.ndarray] = []
        self.conductor: ConductorSolution | None = None  # in the conductors model, once solved

    @property
    def uses_conductor(self) -> bool:
        return self.kind == "conductors"

    def active_charges(self) -> list[PointCharge]:
        """The point charges that act: none in the conductors model."""
        return [] if self.uses_conductor else self.charges

    def active_spheres(self) -> list[Sphere]:
        """The metallic spheres that act: only in the conductors model."""
        return self.spheres if self.uses_conductor else []

    def conductor_plates(self) -> list:
        """The capacitor's plates as conductors, with their thickness (their σ means nothing: the solution
        has the charges)."""
        return self.capacitor.bodies(0.0) if self.capacitor is not None else []

    def voltages(self) -> list[float]:
        """The conductors' potentials, plates first, then spheres."""
        plates = [self.capacitor.voltage / 2, -self.capacitor.voltage / 2] if self.capacitor is not None else []
        return plates + [sphere.voltage for sphere in self.spheres]

    def solve(self, solver: conductors.Solver) -> ConductorSolution:
        """The charges at the scene's potentials; the capacitor's plates get equal and opposite charges."""
        return solver.solve(self.voltages(), neutral=[0, 1] if self.capacitor is not None else [])

    def geometry_key(self) -> tuple:
        return conductors.geometry_key(self.conductor_plates(), self.spheres)

    def plates(self) -> list:
        """The capacitor's plates, each with its σ.

        In the charges model the charged sheets; in the conductors model the
        bodies, with their mean σ = Q/A over the face towards the other plate.
        """
        if self.capacitor is None:
            return []
        if not self.uses_conductor:
            return self.capacitor.plates()
        if self.conductor is None:  # still being computed: the ideal capacitor's densities
            return self.capacitor.bodies(self.capacitor.ideal_sigma())
        bodies = self.conductor_plates()
        for k, body in enumerate(bodies):
            body.sigma = self.conductor.charge(k) / body.area
        return bodies

    def drawn_plates(self) -> list:
        """The plates as they are drawn: bodies of the capacitor's thickness, with their σ for the colour.

        In the charges model the charged sheet is the body's face towards the other plate.
        """
        if self.capacitor is None:
            return []
        bodies = self.capacitor.bodies()
        for body, plate in zip(bodies, self.plates()):
            body.sigma = plate.sigma
        return bodies

    def plate_densities(self) -> list | None:
        """σ over both plates (PlateDensity or RingDensity): uniform in the σ model, solved in the other.

        A uniformly charged sheet has no faces: all get the sheet's σ. A
        conductor carries its charge on its surface, each face with ε₀ times
        the field just outside it. None while the conductor model is computed.
        """
        if self.capacitor is None or (self.uses_conductor and self.conductor is None):
            return None
        densities = []
        for k, body in enumerate(self.drawn_plates()):
            if not self.uses_conductor:
                densities.append(uniform_density(body))
            elif isinstance(body, Ring):
                densities.append(self._ring_density(k, body))
            else:
                densities.append(self._slab_density(k, body))
        return densities

    def _slab_density(self, k: int, slab: Plate) -> PlateDensity:
        a, b = slab.width / 2, slab.height / 2
        u, v = conductors.plate_face_grid(slab)
        # the panels' centres, and the plate's edges with the outermost panels' σ
        u, v = np.concatenate(([-a], u, [a])), np.concatenate(([-b], v, [b]))
        inner, outer = (np.pad(face, 1, mode="edge") for face in self.conductor.faces[k])

        def rim(name: str) -> np.ndarray:  # σ along a rim, averaged across the thickness
            return np.pad(self.conductor.face(k, name).mean(axis=1), 1, mode="edge")

        rim_u = np.array([rim("u-"), rim("u+")])
        rim_v = np.array([rim("v-"), rim("v+")])
        return PlateDensity(slab, u, v, inner, outer, rim_u, rim_v)

    def _ring_density(self, k: int, ring: Ring) -> RingDensity:
        faces = {}
        for name in ring_face_names(ring):
            edges, _ = conductors.ring_face_grid(ring, name)
            # the panels' centres and the face's edges (with the outermost panels' σ), all the way round
            along = np.concatenate(([edges[0]], (edges[1:] + edges[:-1]) / 2, [edges[-1]]))
            sigma = np.pad(self.conductor.face(k, name), ((1, 1), (0, 0)), mode="edge")
            faces[name] = (along, np.concatenate((sigma, sigma[:, :1]), axis=1))
        return RingDensity(ring, ring_angles(), faces)

    def sphere_densities(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] | None:
        """For each metallic sphere: its surface's corner points, quads, and σ on each quad (C/m²).

        σ is None while the charges are being solved (the sphere still shows, plain).
        """
        if not self.uses_conductor:
            return None
        densities = []
        for k, sphere in enumerate(self.spheres):
            points, quads = conductors.sphere_mesh(sphere.center, sphere.radius)
            sigma = self.conductor.sphere_sigma(k) if self.conductor is not None else None
            densities.append((points, quads, sigma))
        return densities

    def sphere_charge(self, k: int) -> float | None:
        if not self.uses_conductor or self.conductor is None:
            return None
        return self.conductor.charge(len(self.conductor_plates()) + k)

    def capacitor_readouts(self) -> dict[str, float] | None:
        """E in the gap, ΔV between the plates, Q, C, and the ideal values (no edges); see Capacitor.probes."""
        cap = self.capacitor
        model = self.model()
        if cap is None or model is None:
            return None
        center, plus, minus = cap.probes()
        E_center = float(np.linalg.norm(model.field(center[None])[0]))
        V = model.potential(np.array([plus, minus]))
        dV = float(V[0] - V[1])
        if self.uses_conductor:
            Q = self.conductor.charge(0)
            C = Q / cap.voltage if cap.voltage else math.nan
        else:
            Q = cap.sigma * cap.area
            C = Q / dV if dV else math.nan
        E_ideal, C_ideal = cap.ideal(Q)
        return dict(E_center=E_center, E_ideal=E_ideal, dV=dV, Q=Q, C=C, C_ideal=C_ideal)

    def model(self) -> FieldModel | None:
        """The field of the scene, or None while the conductor model is being computed."""
        if self.uses_conductor:
            return self.conductor
        sheets = [conductors.UniformSheet(p) if isinstance(p, Ring) else p for p in self.plates()]
        return Superposition([*self.charges, *sheets])

    def snapshot(self) -> FieldModel | None:
        """A copy of the field model that a worker thread can use while the scene keeps changing."""
        if self.uses_conductor:
            return self.conductor  # never modified once computed
        return copy.deepcopy(self.model())

    def set_planar(self, planar: bool) -> None:
        """Switch between 2D and 3D. Going to 2D moves the charges onto z = 0."""
        self.planar = planar
        self.click_seeds = []
        if planar:
            for charge in self.charges:
                charge.position[2] = 0.0
            for sphere in self.spheres:
                sphere.center[2] = 0.0
            if self.capacitor is not None:
                self.capacitor.normal = capacitor_axis(self.capacitor, True)

    def lines(self) -> list[FieldLine]:
        model = self.model()
        if model is None:
            return []
        if self.planar:
            model = InThePlane(model)
        # the lines start from and end on the plates as they are drawn, with their thickness, whatever the model
        charges, plates, spheres = self.active_charges(), self.drawn_plates(), self.active_spheres()
        points, directions = seeding.seeds(
            model,
            charges,
            plates,
            lines_per_nc=self.lines_per_nc,
            charge_radius=SEED_RADIUS,
            plate_offset=PLATE_OFFSET,
            planar=self.planar,
            spheres=spheres,
            conductor=self.conductor if self.uses_conductor else None,
            sphere_offset=SPHERE_OFFSET,
        )
        if self.click_seeds:
            clicks = np.array(self.click_seeds)
            points = np.concatenate((points, clicks, clicks))
            directions = np.concatenate((directions, np.ones(len(clicks)), -np.ones(len(clicks))))
        lines = trace(
            model,
            points,
            directions,
            charges=charges,
            plates=plates,
            plates_absorb=self.uses_conductor,
            spheres=spheres,
            bounds=(np.full(3, -TRACE_HALF), np.full(3, TRACE_HALF)),
            step_max=STEP_MAX,
            r_stop=STOP_RADIUS,
            max_length=MAX_LENGTH,
            step_far=STEP_FAR,
        )
        return [start_on_surface(line, plates, spheres) for line in lines]


class InThePlane:
    """A field model seen in the 2D mode: no field across the z = 0 plane.

    Everything shown in 2D is symmetric about that plane, so the field there
    has no z component; a sum over many panels only cancels it up to rounding,
    and lines would drift out of the plane over hundreds of steps.
    """

    def __init__(self, model: FieldModel):
        self.inner = model

    def field(self, points: np.ndarray) -> np.ndarray:
        E = self.inner.field(points)
        E[:, 2] = 0.0
        return E

    def potential(self, points: np.ndarray) -> np.ndarray:
        return self.inner.potential(points)


def start_on_surface(line: FieldLine, plates: list[Plate], spheres: list[Sphere]) -> FieldLine:
    """A line seeded just off a plate or a sphere, drawn from that surface itself."""
    first = line.points[0]
    surface = None
    for plate in plates:
        if plate.distance(first) <= 1.001 * PLATE_OFFSET:
            surface = plate.surface_point(first)
    for sphere in spheres:
        out = first - sphere.center
        d = np.linalg.norm(out)
        if d <= sphere.radius + 1.001 * SPHERE_OFFSET:
            surface = sphere.center + sphere.radius * out / d
    if surface is None:
        return line
    step = np.linalg.norm(first - surface)
    return FieldLine(
        np.vstack((surface, line.points)),
        np.concatenate(([0.0], line.s + step)),
        np.concatenate((line.strength[:1], line.strength)),
        line.end,
    )
