"""SI constants, the display units of the GUI (nC, cm) and formatting of readouts.

Everything inside the program is in SI units (C, m, V, V/m). Only the GUI
shows charges in nC, lengths in cm and surface densities in μC/m².
"""

import math

EPS0 = 8.8541878188e-12  # vacuum permittivity, F/m (CODATA 2022)
K = 1 / (4 * math.pi * EPS0)  # Coulomb constant, N m²/C²

NC = 1e-9  # one nanocoulomb in C
CM = 1e-2  # one centimetre in m
MM = 1e-3  # one millimetre in m
UC_PER_M2 = 1e-6  # one μC/m² in C/m²

_PREFIXES = [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "μ"), (1e-9, "n"), (1e-12, "p")]


def format_quantity(value: float, unit: str, digits: int = 3) -> str:
    """Value with an SI prefix and `digits` significant figures, e.g. 22600 V/m -> '22.6 kV/m'."""
    if not math.isfinite(value):
        return f"∞ {unit}" if value > 0 else f"-∞ {unit}" if value < 0 else f"— {unit}"
    if value == 0:
        return f"0 {unit}"
    for scale, prefix in _PREFIXES:
        if abs(value) >= scale * 0.9995:
            break
    scaled = value / scale
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(scaled))))
    return f"{scaled:.{decimals}f} {prefix}{unit}"
