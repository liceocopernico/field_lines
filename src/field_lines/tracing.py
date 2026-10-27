"""Field lines: all lines are advanced together, one RK4 step at a time.

The lines follow the unit vector of E (or of -E for lines traced backwards),
so the integration parameter is the arc length s. The step is proportional
to the distance from the nearest point charge and is halved where the line
turns sharply; far from the origin it may grow up to step_far times the
distance, so that a line can make a loop metres wide and come back in a few
hundred steps. A line stops when it

- comes within r_stop of a point charge ("sink"),
- crosses a plate whose charge has the opposite sign, or comes within
  plate_margin of it ("plate"); near a charged sheet the field flips, and a
  step across it would stall. With plates_absorb every plate stops lines
  (they are conductors, ending the lines that reach them),
- comes within plate_margin of a metallic sphere ("sphere"),
- turns back on itself, finds no field or stops advancing: it reached a null point ("null"),
- leaves the bounding box ("bounds") or gets longer than max_length ("length").
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from field_lines.sources import FieldModel, Plate, PointCharge

STEP_PER_DISTANCE = 0.3  # step / distance from the nearest point charge
COS_MAX_TURN = np.cos(np.radians(25))  # a step may turn the line by at most this angle
MAX_HALVINGS = 10


@dataclass
class FieldLine:
    points: np.ndarray  # (n, 3) m
    s: np.ndarray  # (n,) arc length from the seed, m
    strength: np.ndarray  # (n,) |E| at the points, V/m
    end: str  # why the line stopped (see the module docstring)

    @property
    def length(self) -> float:
        return float(self.s[-1])


def trace(
    model: FieldModel,
    seeds: np.ndarray,
    directions: np.ndarray,
    *,
    bounds: tuple[np.ndarray, np.ndarray],
    step_max: float,
    r_stop: float,
    max_length: float,
    charges: Sequence[PointCharge] = (),
    plates: Sequence[Plate] = (),
    plates_absorb: bool = False,
    spheres: Sequence = (),
    plate_margin: float | None = None,
    step_far: float = 0.0,
    max_steps: int = 3000,
) -> list[FieldLine]:
    """Trace one line from each seed, along E (direction +1) or against it (-1).

    plate_margin defaults to the smallest step (r_stop / 2).
    """
    n = len(seeds)
    if n == 0:
        return []
    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    charge_pos = np.array([c.position for c in charges]).reshape(-1, 3)
    sign = np.asarray(directions, dtype=float)
    step_min = r_stop / 2
    margin = step_min if plate_margin is None else plate_margin

    pos = np.array(seeds, dtype=float)
    E = model.field(pos)
    mag = np.linalg.norm(E, axis=1)
    s = np.zeros(n)
    h_prev = np.full(n, step_max)
    end = np.full(n, "", dtype=object)
    end[mag == 0] = "null"
    end[_nearest_charge(pos, charge_pos) < r_stop] = "sink"
    end[np.any((pos < lo) | (pos > hi), axis=1)] = "bounds"

    record_idx, record_pos, record_mag = [np.arange(n)], [pos.copy()], [mag.copy()]
    for _ in range(max_steps):
        idx = np.flatnonzero(end == "")
        if idx.size == 0:
            break
        p = pos[idx]
        sg = sign[idx]
        k1 = _direction(E[idx], sg)
        step_top = np.maximum(step_max, step_far * np.linalg.norm(p, axis=1))
        h = np.clip(STEP_PER_DISTANCE * _nearest_charge(p, charge_pos), step_min, step_top)
        h = np.minimum(h, 2 * h_prev[idx])
        p_new = _adaptive_rk4(model, p, k1, h, sg, step_min)

        E_new = model.field(p_new)
        mag_new = np.linalg.norm(E_new, axis=1)
        step = p_new - p
        seg = np.linalg.norm(step, axis=1)
        h_prev[idx] = np.maximum(seg, step_min)

        stop = np.full(idx.size, "", dtype=object)
        forward = np.einsum("ij,ij->i", _direction(E_new, sg), step)
        stop[(mag_new == 0) | (forward < 0) | (seg < 1e-6 * step_min)] = "null"
        stop[np.any((p_new < lo) | (p_new > hi), axis=1)] = "bounds"
        stop[s[idx] + seg >= max_length] = "length"
        for plate in plates:
            hit, arrival = _plate_hit(plate, p, p_new, sg, margin, plates_absorb)
            p_new[hit] = arrival[hit]
            stop[hit] = "plate"
        for sphere in spheres:
            out = p_new - sphere.center
            d = np.linalg.norm(out, axis=1)
            hit = d < sphere.radius + margin
            p_new[hit] = sphere.center + sphere.radius * out[hit] / d[hit, None]
            stop[hit] = "sphere"
        stop[_nearest_charge(p_new, charge_pos) < r_stop] = "sink"

        s[idx] += np.linalg.norm(p_new - p, axis=1)
        pos[idx], E[idx], mag[idx] = p_new, E_new, mag_new
        end[idx] = stop
        record_idx.append(idx)
        record_pos.append(p_new)
        record_mag.append(mag_new)
    end[end == ""] = "steps"

    return _assemble(n, record_idx, record_pos, record_mag, end)


def _direction(E: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """sign * E / |E|, or 0 where E = 0."""
    mag = np.linalg.norm(E, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(mag[:, None] > 0, E / mag[:, None], 0.0)
    return sign[:, None] * unit


def _nearest_charge(points: np.ndarray, charge_pos: np.ndarray) -> np.ndarray:
    if len(charge_pos) == 0:
        return np.full(len(points), np.inf)
    return np.linalg.norm(points[:, None, :] - charge_pos[None, :, :], axis=2).min(axis=1)


def _adaptive_rk4(model, p, k1, h, sign, step_min):
    """One RK4 step per line; the step is halved while the line would turn too much."""
    p_new = np.empty_like(p)
    todo = np.arange(len(p))
    h = h.copy()
    for _ in range(MAX_HALVINGS):
        hh = h[todo, None]
        pt, k1t, st = p[todo], k1[todo], sign[todo]
        k2 = _direction(model.field(pt + 0.5 * hh * k1t), st)
        k3 = _direction(model.field(pt + 0.5 * hh * k2), st)
        k4 = _direction(model.field(pt + hh * k3), st)
        p_new[todo] = pt + hh / 6 * (k1t + 2 * k2 + 2 * k3 + k4)
        retry = (np.einsum("ij,ij->i", k1t, k4) < COS_MAX_TURN) & (h[todo] > step_min)
        todo = todo[retry]
        if todo.size == 0:
            break
        h[todo] = np.maximum(h[todo] / 2, step_min)
    return p_new


def _plate_hit(plate: Plate, p: np.ndarray, p_new: np.ndarray, sign: np.ndarray, margin: float, absorb: bool):
    """Which steps reach a plate that is a sink for their line (any plate, if absorb), and where they end.

    A step reaches the plate if it crosses it, or ends within margin of it
    (then the line ends at the nearest point of the plate).
    """
    a = plate.to_local(p)
    b = plate.to_local(p_new)
    wa, wb = a[:, 2], b[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(wa != wb, wa / (wa - wb), 0.0)
    c = a + t[:, None] * (b - a)
    half = np.array([plate.width / 2, plate.height / 2])
    sink = np.ones(len(p), dtype=bool) if absorb else plate.sigma * sign < 0
    crossed = sink & (wa * wb <= 0) & (wa != 0) & np.all(np.abs(c[:, :2]) <= half, axis=1)
    near = sink & ~crossed & (np.abs(wb) <= margin) & np.all(np.abs(b[:, :2]) <= half + margin, axis=1)
    arrival_local = np.where(crossed[:, None], c, b)
    arrival_local[near, 2] = 0.0
    arrival_local[near, :2] = np.clip(arrival_local[near, :2], -half, half)
    arrival = plate.from_local(arrival_local) + plate.center
    return crossed | near, arrival


def _assemble(n, record_idx, record_pos, record_mag, end) -> list[FieldLine]:
    all_idx = np.concatenate(record_idx)
    order = np.argsort(all_idx, kind="stable")
    points = np.concatenate(record_pos)[order]
    strength = np.concatenate(record_mag)[order]
    counts = np.bincount(all_idx, minlength=n)
    lines = []
    start = 0
    for i, count in enumerate(counts):
        pts = points[start : start + count]
        mags = strength[start : start + count]
        start += count
        if count < 2:
            continue
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        lines.append(FieldLine(pts, s, mags, end[i]))
    return lines
