"""What is being shown: the sources, 2D or 3D, the model. No Qt in here.

Two models:
- "charges": point charges and a capacitor whose plates carry a uniform σ
  (the field is the sum of theirs);
- "conductors": metallic spheres and a capacitor whose plates are held at
  fixed potentials (conductors.ConductorSolution finds their charges).
"""

import copy
import math
from dataclasses import dataclass

import numpy as np

from field_lines import conductors, seeding
from field_lines.conductors import ConductorSolution, Sphere
from field_lines.sources import AXES, FieldModel, Plate, PointCharge, Superposition
from field_lines.tracing import FieldLine, trace
from field_lines.units import CM, EPS0, NC, UC_PER_M2

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


@dataclass
class Capacitor:
    width: float = 10 * CM
    height: float = 10 * CM
    separation: float = 3 * CM
    sigma: float = 0.2 * UC_PER_M2  # charges model: charge density of the positive plate
    voltage: float = 500.0  # conductors model: the plates are at +voltage/2 and -voltage/2
    normal: str = "z"  # the capacitor's axis; the positive plate is on its + side

    @property
    def area(self) -> float:
        return self.width * self.height

    def plates(self, sigma: float | None = None) -> list[Plate]:
        """The positive and the negative plate, with density ±sigma (default: self.sigma)."""
        sigma = self.sigma if sigma is None else sigma
        offset = np.zeros(3)
        offset[AXES.index(self.normal)] = self.separation / 2
        return [
            Plate(sigma, offset, self.width, self.height, self.normal),
            Plate(-sigma, -offset, self.width, self.height, self.normal),
        ]


@dataclass
class PlateDensity:
    """Surface charge density over a plate's two faces, on a grid of its local (u, v) coordinates."""

    plate: Plate
    u: np.ndarray  # along the plate's width
    v: np.ndarray  # along its height
    inner: np.ndarray  # (len(u), len(v)) σ on the face towards the other plate, C/m²
    outer: np.ndarray  # σ on the face away from it

    def at(self, u: float, v: float) -> tuple[float, float]:
        """σ of the inner and outer face at a point of the plate (nearest grid node)."""
        i = int(np.argmin(np.abs(self.u - u)))
        j = int(np.argmin(np.abs(self.v - v)))
        return float(self.inner[i, j]), float(self.outer[i, j])


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

    def conductor_plates(self) -> list[Plate]:
        """The capacitor's plates as conductors (their σ means nothing: the solution has the charges)."""
        return self.capacitor.plates(0.0) if self.capacitor is not None else []

    def voltages(self) -> list[float]:
        """The conductors' potentials, plates first, then spheres."""
        plates = [self.capacitor.voltage / 2, -self.capacitor.voltage / 2] if self.capacitor is not None else []
        return plates + [sphere.voltage for sphere in self.spheres]

    def geometry_key(self) -> tuple:
        return conductors.geometry_key(self.conductor_plates(), self.spheres)

    def plates(self) -> list[Plate]:
        """The capacitor's plates, each with its σ (in the conductors model, its mean σ = Q/A)."""
        if self.capacitor is None:
            return []
        if not self.uses_conductor:
            return self.capacitor.plates()
        plates = self.conductor_plates()
        for k, plate in enumerate(plates):
            if self.conductor is not None:
                plate.sigma = self.conductor.charge(k) / self.capacitor.area
            else:  # still being computed: the ideal capacitor's density
                plate.sigma = (1 - 2 * k) * EPS0 * self.capacitor.voltage / self.capacitor.separation
        return plates

    def plate_densities(self) -> list[PlateDensity] | None:
        """σ over both plates: uniform in the σ model, from the solution in the conductor model.

        A uniformly charged sheet has no faces: both get the sheet's σ. A
        conductor carries its charge on its surface, each face with ε₀ times
        the field just outside it. None while the conductor model is computed.
        """
        if self.capacitor is None or (self.uses_conductor and self.conductor is None):
            return None
        densities = []
        for k, plate in enumerate(self.plates()):
            a, b = plate.width / 2, plate.height / 2
            if not self.uses_conductor:
                same = np.full((2, 2), plate.sigma)
                densities.append(PlateDensity(plate, np.array([-a, a]), np.array([-b, b]), same, same.copy()))
                continue
            u, v = conductors.plate_face_grid(plate)
            # the panels' centres, and the plate's edges with the outermost panels' σ
            u, v = np.concatenate(([-a], u, [a])), np.concatenate(([-b], v, [b]))
            inner, outer = (np.pad(face, 1, mode="edge") for face in self.conductor.faces[k])
            densities.append(PlateDensity(plate, u, v, inner, outer))
        return densities

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
        """E at the centre, ΔV between the plates' centres, Q, C, and the ideal (infinite plates) values."""
        cap = self.capacitor
        model = self.model()
        if cap is None or model is None:
            return None
        plus, minus = cap.plates()
        E_center = float(np.linalg.norm(model.field(np.zeros((1, 3)))[0]))
        V = model.potential(np.array([plus.center, minus.center]))
        dV = float(V[0] - V[1])
        if self.uses_conductor:
            Q = self.conductor.charge(0)
            C = Q / cap.voltage if cap.voltage else math.nan
        else:
            Q = cap.sigma * cap.area
            C = Q / dV if dV else math.nan
        return dict(
            E_center=E_center,
            E_ideal=abs(Q) / cap.area / EPS0,
            dV=dV,
            Q=Q,
            C=C,
            C_ideal=EPS0 * cap.area / cap.separation,
        )

    def model(self) -> FieldModel | None:
        """The field of the scene, or None while the conductor model is being computed."""
        if self.uses_conductor:
            return self.conductor
        return Superposition([*self.charges, *self.plates()])

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
            if self.capacitor is not None and self.capacitor.normal == "z":
                self.capacitor.normal = "x"

    def lines(self) -> list[FieldLine]:
        model = self.model()
        if model is None:
            return []
        if self.planar:
            model = InThePlane(model)
        charges, plates, spheres = self.active_charges(), self.plates(), self.active_spheres()
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
        local = plate.to_local(first[None])[0]
        on_face = abs(local[2]) <= 1.001 * PLATE_OFFSET
        if on_face and abs(local[0]) <= plate.width / 2 and abs(local[1]) <= plate.height / 2:
            local[2] = 0.0
            surface = plate.from_local(local[None])[0] + plate.center
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
