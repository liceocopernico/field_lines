import time

import numpy as np
import pytest

from field_lines import seeding
from field_lines.sources import Plate, PointCharge, Superposition
from field_lines.tracing import trace
from field_lines.units import CM, NC

BOX = (np.full(3, -0.2), np.full(3, 0.2))
OPTIONS = dict(bounds=BOX, step_max=0.002, r_stop=0.002, max_length=1.0)


class Counting:
    """A field model that counts how many points it was evaluated at."""

    def __init__(self, model):
        self.model = model
        self.points = 0

    def field(self, p):
        self.points += len(p)
        return self.model.field(p)

    def potential(self, p):
        return self.model.potential(p)


def trace_sources(charges, plates=(), planar=False, lines_per_nc=10, **options):
    model = Superposition([*charges, *plates])
    seeds, directions = seeding.seeds(
        model, charges, plates, lines_per_nc=lines_per_nc, charge_radius=0.004, plate_offset=0.001, planar=planar
    )
    return trace(model, seeds, directions, charges=charges, plates=plates, **(OPTIONS | options))


def dipole():
    return [PointCharge(2 * NC, [-3 * CM, 0, 0]), PointCharge(-2 * NC, [3 * CM, 0, 0])]


def test_dipole_lines_go_from_plus_to_minus_or_away():
    lines = trace_sources(dipole(), planar=True)
    assert len(lines) == 20
    for line in lines:
        assert np.linalg.norm(line.points[0] - [-0.03, 0, 0]) == pytest.approx(0.004)
        assert line.end in ("sink", "bounds")
        if line.end == "sink":
            assert np.linalg.norm(line.points[-1] - [0.03, 0, 0]) < 0.002 + 1e-9
    assert sum(line.end == "sink" for line in lines) >= 14  # equal charges: nearly every line closes


def test_dipole_lines_are_mirror_symmetric():
    lines = trace_sources(dipole(), planar=True, lines_per_nc=6)  # 12 lines, at 0°, 30°, ... 330°
    ends = np.array([line.points[-1] for line in lines])
    mirrored = ends * [1, -1, 1]
    for end in mirrored:
        assert np.min(np.linalg.norm(ends - end, axis=1)) < 1e-6


def test_planar_seeds_stay_in_the_plane():
    charges = [PointCharge(3 * NC, [-0.02, 0.01, 0]), PointCharge(-1 * NC, [0.03, -0.02, 0]), PointCharge(NC, [0, 0.04, 0])]
    for line in trace_sources(charges, planar=True):
        assert np.abs(line.points[:, 2]).max() < 1e-12


def test_line_into_a_null_point_stops_there_quickly():
    charges = [PointCharge(2 * NC, [-3 * CM, 0, 0]), PointCharge(2 * NC, [3 * CM, 0, 0])]
    model = Counting(Superposition(charges))
    lines = trace(model, np.array([[-0.025, 0.0, 0.0]]), np.array([1]), charges=charges, **OPTIONS)
    assert lines[0].end == "null"
    assert np.linalg.norm(lines[0].points[-1]) < 0.002
    assert model.points < 2000


def test_line_into_a_sink_takes_few_steps():
    charges = dipole()
    model = Counting(Superposition(charges))
    lines = trace(model, np.array([[-0.03, 0.004, 0.0]]), np.array([1]), charges=charges, **OPTIONS)
    assert lines[0].end == "sink"
    assert model.points < 3000


def test_number_of_lines_is_proportional_to_charge():
    charges = [PointCharge(3 * NC, [-0.05, 0, 0]), PointCharge(1 * NC, [0.05, 0, 0])]
    seeds, _ = seeding.seeds(
        Superposition(charges), charges, [], lines_per_nc=8, charge_radius=0.004, plate_offset=0.001, planar=False
    )
    near_first = np.linalg.norm(seeds - charges[0].position, axis=1) < 0.005
    assert near_first.sum() == 24 and (~near_first).sum() == 8


def test_negative_majority_is_traced_backwards():
    charges = [PointCharge(1 * NC, [-0.03, 0, 0]), PointCharge(-3 * NC, [0.03, 0, 0])]
    lines = trace_sources(charges, planar=True)
    assert len(lines) == 30
    assert all(np.linalg.norm(line.points[0] - [0.03, 0, 0]) == pytest.approx(0.004) for line in lines)
    assert sum(line.end == "sink" for line in lines) >= 8


def test_two_hundred_lines_are_fast():
    charges = dipole()
    trace_sources(charges, lines_per_nc=10)  # warm up
    times = []
    for _ in range(3):  # the best of three, so that a busy machine does not fail the test
        start = time.perf_counter()
        lines = trace_sources(charges, lines_per_nc=100)
        times.append(time.perf_counter() - start)
    assert len(lines) == 200
    assert min(times) < 0.2


def test_capacitor_lines_leave_plus_plate_and_end_on_minus_plate():
    sigma, d = 2e-7, 0.02
    plates = [Plate(sigma, [0, 0, d / 2], 0.1, 0.1), Plate(-sigma, [0, 0, -d / 2], 0.1, 0.1)]
    lines = trace_sources([], plates, lines_per_nc=20)
    inner = [line for line in lines if line.points[0, 2] < d / 2]
    assert len(lines) == seeding.line_count(plates[0].charge, 20)
    # most of the flux leaves the inner face: at the centre the other plate's field is
    # 0.66 σ/2ε₀ (10 cm plates 2 cm apart), so 83% goes inwards there, less near the edges
    assert 0.7 * len(lines) < len(inner) < 0.9 * len(lines)
    for line in inner:
        assert line.end == "plate"
        assert line.points[-1, 2] == pytest.approx(-d / 2)


def test_capacitor_cut_across_keeps_lines_in_the_plane():
    sigma, d = 2e-7, 0.02
    plates = [Plate(sigma, [d / 2, 0, 0], 0.1, 0.1, normal="x"), Plate(-sigma, [-d / 2, 0, 0], 0.1, 0.1, normal="x")]
    lines = trace_sources([], plates, planar=True, lines_per_nc=20)
    assert lines
    for line in lines:
        assert np.abs(line.points[:, 2]).max() < 1e-12
