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


@dataclass
class Plate:
    """A rectangle with uniform surface charge density sigma (C/m²).

    The plate lies across its normal axis. In local coordinates (u, v, w) it
    spans |u| <= width/2, |v| <= height/2, w = 0, where w is the normal axis
    and (u, v, w) follows the axes cyclically: normal z -> (x, y, z),
    normal x -> (y, z, x), normal y -> (z, x, y).
    """

    sigma: float
    center: np.ndarray
    width: float
    height: float
    normal: str = "z"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        a = AXES.index(self.normal)
        self.local_axes = [(a + 1) % 3, (a + 2) % 3, a]

    @property
    def charge(self) -> float:
        return self.sigma * self.width * self.height

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (points - self.center)[:, self.local_axes]

    def from_local(self, vectors: np.ndarray) -> np.ndarray:
        out = np.empty_like(vectors)
        out[:, self.local_axes] = vectors
        return out

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
