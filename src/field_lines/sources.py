"""Charge distributions and the fields they produce, vectorized over points.

Every source (and every field model, see Superposition and
conductors.ConductorSolution) has

    field(points) -> (M, 3) array of E in V/m
    potential(points) -> (M,) array of V in volts

where points is an (M, 3) array of positions in metres.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from field_lines.units import EPS0, K

AXES = "xyz"


class FieldModel(Protocol):
    def field(self, points: np.ndarray) -> np.ndarray: ...

    def potential(self, points: np.ndarray) -> np.ndarray: ...


@dataclass
class PointCharge:
    q: float  # C
    position: np.ndarray  # (3,) m

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)

    def field(self, points: np.ndarray) -> np.ndarray:
        r = points - self.position
        d2 = np.einsum("ij,ij->i", r, r)
        with np.errstate(divide="ignore"):
            inv_d3 = np.where(d2 > 0, d2**-1.5, 0.0)
        return K * self.q * r * inv_d3[:, None]

    def potential(self, points: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(points - self.position, axis=1)
        with np.errstate(divide="ignore"):
            return np.where(d > 0, K * self.q / np.where(d > 0, d, 1.0), 0.0)


class LocalFrame:
    """Local coordinates (u, v, w) about a body's centre, w along its normal axis.

    (u, v, w) follows the axes cyclically: normal z -> (x, y, z), normal x -> (y, z, x), normal y -> (z, x, y).
    """

    center: np.ndarray
    normal: str

    def _set_frame(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        a = AXES.index(self.normal)
        self.local_axes = [(a + 1) % 3, (a + 2) % 3, a]

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (points - self.center)[:, self.local_axes]

    def from_local(self, vectors: np.ndarray) -> np.ndarray:
        out = np.empty_like(vectors)
        out[:, self.local_axes] = vectors
        return out

    def surface_point(self, point: np.ndarray) -> np.ndarray:
        """The point of the body's surface nearest to point (from outside, or from inside it)."""
        return self.surface_points(point[None])[0]

    def distance(self, point: np.ndarray) -> float:
        """Distance from the body (0 inside it)."""
        return float(self.distances(point[None])[0])


@dataclass
class Plate(LocalFrame):
    """A rectangle with uniform surface charge density sigma (C/m²).

    The plate lies across its normal axis. In local coordinates (u, v, w) it
    spans |u| <= width/2, |v| <= height/2, w = 0.

    With a thickness the plate is a slab, |w| <= thickness/2: a box-shaped
    conductor in the conductors model. Its field and potential here are those
    of the uniformly charged sheet (thickness 0), which only the charges
    model uses.
    """

    sigma: float
    center: np.ndarray
    width: float
    height: float
    normal: str = "z"
    thickness: float = 0.0

    def __post_init__(self):
        self._set_frame()

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def charge(self) -> float:
        return self.sigma * self.area

    @property
    def half(self) -> np.ndarray:
        """Half sizes along local u, v, w."""
        return np.array([self.width / 2, self.height / 2, self.thickness / 2])

    def key(self) -> tuple:
        """What its shape and place are (for telling geometries apart)."""
        return ("plate", tuple(np.round(self.center, 9)), *np.round(self.half, 9), self.normal)

    def surface_points(self, points: np.ndarray) -> np.ndarray:
        local = self.to_local(points)
        nearest = np.clip(local, -self.half, self.half)
        inside = np.flatnonzero(np.all(nearest == local, axis=1))  # out through the nearest face
        k = np.argmin(self.half - np.abs(nearest[inside]), axis=1)
        nearest[inside, k] = np.where(nearest[inside, k] >= 0, 1.0, -1.0) * self.half[k]
        return self.from_local(nearest) + self.center

    def distances(self, points: np.ndarray) -> np.ndarray:
        local = self.to_local(points)
        return np.linalg.norm(local - np.clip(local, -self.half, self.half), axis=1)

    def entry(self, p: np.ndarray, p_new: np.ndarray) -> np.ndarray:
        """Where each step from p to p_new enters the plate, as a fraction of the step; nan where it does not."""
        a = self.to_local(p)
        d = self.to_local(p_new) - a
        half = self.half
        # where the step enters and leaves each band |u| <= a, |v| <= b, |w| <= t/2 (slab method)
        with np.errstate(invalid="ignore", divide="ignore"):
            t1, t2 = (-half - a) / d, (half - a) / d
        still = d == 0  # parallel to a band: inside it all along, or never
        within = np.abs(a) <= half
        t_in = np.where(still, np.where(within, -np.inf, np.inf), np.minimum(t1, t2)).max(axis=1)
        t_out = np.where(still, np.where(within, np.inf, -np.inf), np.maximum(t1, t2)).min(axis=1)
        outside = np.any(np.abs(a) > half, axis=1)
        enters = outside & (t_in <= t_out) & (t_in >= 0) & (t_in <= 1)
        return np.where(enters, t_in, np.nan)

    def field(self, points: np.ndarray) -> np.ndarray:
        u, v, w = self.to_local(points).T
        local = rectangle_field(u, v, w, self.width / 2, self.height / 2)
        return self.from_local(K * self.sigma * local)

    def potential(self, points: np.ndarray) -> np.ndarray:
        u, v, w = self.to_local(points).T
        return K * self.sigma * rectangle_potential(u, v, w, self.width / 2, self.height / 2)

    def field_ideal(self) -> float:
        """|E| next to an infinite plate with the same σ: σ / (2 ε₀)."""
        return abs(self.sigma) / (2 * EPS0)


@dataclass
class Ring(LocalFrame):
    """A body of revolution about its normal axis: r_in <= ρ <= r_out, |w| <= half_length.

    ρ is the distance from the axis, w the local coordinate along it. A disc
    shaped plate has r_in = 0, a tube r_in > 0. Of zero thickness it is a
    sheet: a flat disc or annulus (half_length = 0) or a cylindrical surface
    (r_in = r_out).

    gap names the face towards the other conductor of a capacitor: "w+" or
    "w-" (the flat faces), "out" (ρ = r_out) or "in" (ρ = r_in). σ refers to
    that face's area; in the charges model σ sits on it (a sheet), and its
    field is in conductors.UniformSheet.
    """

    sigma: float
    center: np.ndarray
    r_in: float
    r_out: float
    half_length: float
    normal: str = "z"
    gap: str = "w+"

    def __post_init__(self):
        self._set_frame()

    @property
    def area(self) -> float:
        if self.gap in ("w+", "w-"):
            return np.pi * (self.r_out**2 - self.r_in**2)
        return 2 * np.pi * (self.r_out if self.gap == "out" else self.r_in) * 2 * self.half_length

    @property
    def charge(self) -> float:
        return self.sigma * self.area

    def key(self) -> tuple:
        sizes = np.round([self.r_in, self.r_out, self.half_length], 9)
        return ("ring", tuple(np.round(self.center, 9)), *sizes, self.normal)

    def meridian(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ρ and w of the points, and their local (u, v)."""
        local = self.to_local(points)
        return np.hypot(local[:, 0], local[:, 1]), local[:, 2], local[:, :2]

    def distances(self, points: np.ndarray) -> np.ndarray:
        rho, w, _ = self.meridian(points)
        d_rho = np.maximum(np.maximum(self.r_in - rho, rho - self.r_out), 0.0)
        d_w = np.maximum(np.abs(w) - self.half_length, 0.0)
        return np.hypot(d_rho, d_w)

    def surface_points(self, points: np.ndarray) -> np.ndarray:
        rho, w, uv = self.meridian(points)
        h = self.half_length
        near_rho, near_w = np.clip(rho, self.r_in, self.r_out), np.clip(w, -h, h)
        inside = (near_rho == rho) & (near_w == w)
        # from inside, out through the nearest face: the inner cylinder (if any), the outer one, or a flat face
        to_inner = rho - self.r_in if self.r_in > 0 else np.full(len(rho), np.inf)
        gaps = np.column_stack((to_inner, self.r_out - rho, h - np.abs(w)))
        face = np.argmin(gaps, axis=1)
        near_rho = np.where(inside & (face == 0), self.r_in, near_rho)
        near_rho = np.where(inside & (face == 1), self.r_out, near_rho)
        near_w = np.where(inside & (face == 2), np.where(w >= 0, h, -h), near_w)
        with np.errstate(invalid="ignore", divide="ignore"):
            radial = np.where(rho[:, None] > 0, uv / rho[:, None], [1.0, 0.0])  # on the axis: any direction
        local = np.column_stack((near_rho[:, None] * radial, near_w))
        return self.from_local(local) + self.center

    def entry(self, p: np.ndarray, p_new: np.ndarray) -> np.ndarray:
        """Where each step from p to p_new enters the body, as a fraction of the step; nan where it does not.

        The first crossing of any of its faces, for steps that start outside:
        the flat faces at w = ±half_length (linear in the step) and the
        cylinders at ρ = r_in and r_out (quadratic).
        """
        a = self.to_local(p)
        d = self.to_local(p_new) - a
        h = self.half_length
        first = np.full(len(p), np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            for side in (-h, h):
                t = (side - a[:, 2]) / d[:, 2]
                rho = np.hypot(a[:, 0] + t * d[:, 0], a[:, 1] + t * d[:, 1])
                hit = (t >= 0) & (t <= 1) & (rho >= self.r_in) & (rho <= self.r_out)
                first = np.where(hit & (t < first), t, first)
            qa = d[:, 0] ** 2 + d[:, 1] ** 2
            qb = a[:, 0] * d[:, 0] + a[:, 1] * d[:, 1]
            for radius in {self.r_in, self.r_out} - {0.0}:
                qc = a[:, 0] ** 2 + a[:, 1] ** 2 - radius**2
                root = np.sqrt(qb * qb - qa * qc)
                for t in ((-qb - root) / qa, (-qb + root) / qa):
                    hit = (t >= 0) & (t <= 1) & (np.abs(a[:, 2] + t * d[:, 2]) <= h)
                    first = np.where(hit & (t < first), t, first)
        enters = (self.distances(p) > 0) & np.isfinite(first)
        return np.where(enters, first, np.nan)


class Superposition:
    """The field of several sources, added together."""

    def __init__(self, sources: list):
        self.sources = list(sources)

    def field(self, points: np.ndarray) -> np.ndarray:
        total = np.zeros((len(points), 3))
        for source in self.sources:
            total += source.field(points)
        return total

    def potential(self, points: np.ndarray) -> np.ndarray:
        total = np.zeros(len(points))
        for source in self.sources:
            total += source.potential(points)
        return total


def _rectangle_terms(u, v, w, a, b):
    """Offsets of the rectangle's sides from the point (u, v, w) and the integrals over them.

    The rectangle is |u| <= a, |v| <= b, w = 0; a and b may be arrays, one per point.
    x1, x2 (y1, y2) are the offsets of its u (v) sides from the point.
    L(x) = ∫ dy/R over y1..y2 at offset x, M(y) = ∫ dx/R over x1..x2 at offset y.
    """
    x1, x2, y1, y2 = -a - u, a - u, -b - v, b - v
    w2 = w * w
    L1, L2 = _log_ratio(y1, y2, x1 * x1 + w2), _log_ratio(y1, y2, x2 * x2 + w2)
    M1, M2 = _log_ratio(x1, x2, y1 * y1 + w2), _log_ratio(x1, x2, y2 * y2 + w2)
    atan_sum = sum(
        sign * _solid_angle_term(x, y, w) for x, y, sign in ((x2, y2, 1), (x1, y1, 1), (x1, y2, -1), (x2, y1, -1))
    )
    return x1, x2, y1, y2, L1, L2, M1, M2, atan_sum


def rectangle_field(u, v, w, a, b) -> np.ndarray:
    """E/(k σ) of a uniformly charged rectangle, in its local axes, at local points (u, v, w): (P, 3)."""
    _, _, _, _, L1, L2, M1, M2, atan_sum = _rectangle_terms(u, v, w, a, b)
    return np.column_stack((L2 - L1, M2 - M1, atan_sum))


def rectangle_potential(u, v, w, a, b) -> np.ndarray:
    """V/(k σ) of a uniformly charged rectangle at local points (u, v, w): (P,)."""
    x1, x2, y1, y2, L1, L2, M1, M2, atan_sum = _rectangle_terms(u, v, w, a, b)
    return _times(x2, L2) - _times(x1, L1) + _times(y2, M2) - _times(y1, M1) - w * atan_sum


def _log_ratio(t1: np.ndarray, t2: np.ndarray, rho2: np.ndarray) -> np.ndarray:
    """∫ dt / sqrt(t² + rho2) from t1 to t2 (t1 < t2) = ln((t2 + R2) / (t1 + R1)).

    Written three ways so that no branch subtracts nearly equal numbers;
    infinite only on the edge itself (rho2 = 0 with t1 < 0 < t2).
    """
    r1 = np.sqrt(t1 * t1 + rho2)
    r2 = np.sqrt(t2 * t2 + rho2)
    with np.errstate(divide="ignore", invalid="ignore"):
        above = np.log((t2 + r2) / (t1 + r1))
        below = np.log((r1 - t1) / (r2 - t2))
        across = np.log((t2 + r2) * (r1 - t1) / rho2)
    return np.where(t1 >= 0, above, np.where(t2 <= 0, below, across))


def _solid_angle_term(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """atan(x y / (w R)) for one corner, equal to 0 on the plate's plane (w = 0)."""
    r = np.sqrt(x * x + y * y + w * w)
    return np.arctan2(x * y * np.sign(w), np.abs(w) * r)


def _times(t: np.ndarray, log_term: np.ndarray) -> np.ndarray:
    """t * log_term, taken as 0 where t = 0 (the log may be infinite there)."""
    with np.errstate(invalid="ignore"):
        return np.where(t == 0, 0.0, t * log_term)
