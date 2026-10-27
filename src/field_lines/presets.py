"""Ready-made setups. Charges are given in nC and cm, in the z = 0 plane so they work in 2D too."""

from field_lines.scene import Capacitor, Scene
from field_lines.sources import PointCharge
from field_lines.units import CM, NC

PRESETS: dict[str, list[tuple[float, float, float]]] = {
    "empty": [],
    "single": [(2, 0, 0)],
    "dipole": [(2, -3, 0), (-2, 3, 0)],
    "equal": [(2, -3, 0), (2, 3, 0)],
    "quadrupole": [(2, -3, -3), (-2, 3, -3), (2, 3, 3), (-2, -3, 3)],
    "capacitor": [],
}


def load(scene: Scene, key: str) -> None:
    """Replace the scene's sources with the preset's (keeps 2D/3D and the views)."""
    scene.charges = [PointCharge(q * NC, [x * CM, y * CM, 0.0]) for q, x, y in PRESETS[key]]
    scene.capacitor = Capacitor(normal="x" if scene.planar else "z") if key == "capacitor" else None
    scene.click_seeds = []
