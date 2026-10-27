"""Ready-made setups, in the z = 0 plane so they work in 2D too.

Each gives point charges (nC, cm) or metallic spheres (V, radius cm, centre
cm), and whether there is a capacitor. Setups with spheres use the
conductors model, setups with charges the charges model; the capacitor alone
keeps the model that is on.
"""

from field_lines.conductors import Sphere
from field_lines.scene import Capacitor, Scene
from field_lines.sources import PointCharge
from field_lines.units import CM, NC

CHARGES = "charges"
SPHERES = "spheres"

PRESETS: dict[str, dict] = {
    "empty": dict(),
    "single": dict(charges=[(2, 0, 0)]),
    "dipole": dict(charges=[(2, -3, 0), (-2, 3, 0)]),
    "equal": dict(charges=[(2, -3, 0), (2, 3, 0)]),
    "quadrupole": dict(charges=[(2, -3, -3), (-2, 3, -3), (2, 3, 3), (-2, -3, 3)]),
    "capacitor": dict(capacitor=True),
    "sphere": dict(spheres=[(1000, 2, 0, 0)]),
    "two_spheres": dict(spheres=[(1000, 1.5, -4, 0), (-1000, 1.5, 4, 0)]),
    "sphere_in_capacitor": dict(capacitor=True, spheres=[(0, 1.5, 0, 0)], gap=6),
}


def load(scene: Scene, key: str) -> None:
    """Replace the scene's sources with the preset's (keeps 2D/3D and the views)."""
    preset = PRESETS[key]
    scene.charges = [PointCharge(q * NC, [x * CM, y * CM, 0.0]) for q, x, y in preset.get("charges", [])]
    scene.spheres = [Sphere([x * CM, y * CM, 0.0], r * CM, V) for V, r, x, y in preset.get("spheres", [])]
    scene.capacitor = None
    if preset.get("capacitor"):
        scene.capacitor = Capacitor(normal="x" if scene.planar else "z")
        if "gap" in preset:
            scene.capacitor.separation = preset["gap"] * CM
    if "charges" in preset:
        scene.kind = "charges"
    elif "spheres" in preset:
        scene.kind = "conductors"
    scene.conductor = None
    scene.click_seeds = []
