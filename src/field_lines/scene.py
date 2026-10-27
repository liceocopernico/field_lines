"""What is being shown: the sources, 2D or 3D, the capacitor and its model. No Qt in here."""

import math
from dataclasses import dataclass

import numpy as np

from field_lines import seeding
from field_lines.sources import AXES, FieldModel, Plate, PointCharge, Superposition
from field_lines.tracing import FieldLine, trace
from field_lines.units import CM, NC, UC_PER_M2

VIEW_HALF = 10 * CM  # the view shows the cube |x|, |y|, |z| <= VIEW_HALF
BOUNDS_HALF = 1.5 * VIEW_HALF  # lines stop beyond this
STEP_MAX = 2 * VIEW_HALF / 150
SEED_RADIUS = 0.35 * CM  # lines start this far from a point charge (inside its sphere) ...
STOP_RADIUS = 0.25 * CM  # ... and end this close to one
PLATE_OFFSET = 0.1 * CM  # lines start this far from a plate
MAX_LENGTH = 1.0  # m


def charge_radius(q: float) -> float:
    """Radius of the sphere that shows a point charge."""
    return 0.4 * CM + 0.15 * CM * math.sqrt(abs(q) / NC)


@dataclass
class Capacitor:
    width: float = 10 * CM
    height: float = 10 * CM
    separation: float = 3 * CM
    sigma: float = 0.2 * UC_PER_M2  # σ model: charge density of the positive plate
    voltage: float = 500.0  # conductor model: potential difference between the plates
    model: str = "sigma"  # "sigma" or "conductor"
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


class Scene:
    def __init__(self):
        self.charges: list[PointCharge] = []
        self.capacitor: Capacitor | None = None
        self.planar = False
        self.lines_per_nc = 10.0
        self.click_seeds: list[np.ndarray] = []
        self.conductor = None  # conductors.ConductorSolution of the capacitor, in the conductor model

    @property
    def uses_conductor(self) -> bool:
        return self.capacitor is not None and self.capacitor.model == "conductor"

    def active_charges(self) -> list[PointCharge]:
        """The point charges that act: none in the conductor model."""
        return [] if self.uses_conductor else self.charges

    def plates(self) -> list[Plate]:
        if self.capacitor is None:
            return []
        if self.uses_conductor:
            if self.conductor is None:
                return []
            return self.capacitor.plates(self.conductor.charge / self.capacitor.area)
        return self.capacitor.plates()

    def model(self) -> FieldModel | None:
        """The field of the scene, or None while the conductor model is being computed."""
        if self.uses_conductor:
            return self.conductor
        return Superposition([*self.charges, *self.plates()])

    def set_planar(self, planar: bool) -> None:
        """Switch between 2D and 3D. Going to 2D moves the charges onto z = 0."""
        self.planar = planar
        self.click_seeds = []
        if planar:
            for charge in self.charges:
                charge.position[2] = 0.0
            if self.capacitor is not None and self.capacitor.normal == "z":
                self.capacitor.normal = "x"

    def lines(self) -> list[FieldLine]:
        model = self.model()
        if model is None:
            return []
        charges, plates = self.active_charges(), self.plates()
        points, directions = seeding.seeds(
            model,
            charges,
            plates,
            lines_per_nc=self.lines_per_nc,
            charge_radius=SEED_RADIUS,
            plate_offset=self.plate_offset(),
            planar=self.planar,
        )
        if self.click_seeds:
            clicks = np.array(self.click_seeds)
            points = np.concatenate((points, clicks, clicks))
            directions = np.concatenate((directions, np.ones(len(clicks)), -np.ones(len(clicks))))
        return trace(
            model,
            points,
            directions,
            charges=charges,
            plates=plates,
            bounds=(np.full(3, -BOUNDS_HALF), np.full(3, BOUNDS_HALF)),
            step_max=STEP_MAX,
            r_stop=STOP_RADIUS,
            max_length=MAX_LENGTH,
        )

    def plate_offset(self) -> float:
        if self.uses_conductor and self.conductor is not None:
            return max(PLATE_OFFSET, 1.5 * self.conductor.spacing)
        return PLATE_OFFSET
