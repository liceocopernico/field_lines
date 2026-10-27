# field-lines

Electric field lines, equipotentials and capacitors, in 3D space or in a 2D
plane, with a Qt window around a [PyVista](https://pyvista.org) 3D view.
The interface speaks Italian and English.

- **Point charges:** place them, drag them, change them, and watch the field
  lines grow out of the positive charges and end on the negative ones.
- **Parallel plate capacitor:** two finite plates, either with a uniform
  surface charge (superposition) or as conductors held at a potential
  difference (Laplace's equation), with E, ΔV, Q and C read out.
- **Views:** field lines, equipotential surfaces (lines in 2D), an arrow
  grid, and a movable plane coloured by V or |E|.

## Run

```bash
uv run field-lines                      # 3D, dipole, language of the system
uv run field-lines --mode 2d            # the 2D plane, like the old pygame app
uv run field-lines --preset capacitor   # empty, single, dipole, equal, quadrupole, capacitor
uv run field-lines --lang en            # it or en
```

Python 3.14; `uv` installs everything else (PySide6, pyvistaqt, PyVista/VTK,
NumPy, SciPy).

## Using it

The panel on the right holds:

- **2D / 3D:** going to 2D moves the charges onto the z = 0 plane and turns a
  capacitor so that its plates cross the plane.
- **Setup:** ready-made setups.
- **Point charges:** a table of q (nC) and x, y, z (cm), plus buttons to add
  or remove charges.
- **Capacitor:** plate size, separation d, the model, and σ or ΔV.
- **Show:** field lines (lines per nC), equipotentials (number of levels),
  arrows, and the coloured plane (V or |E|, its axis and position).
- **Animation:** replay the growth of the lines, and set its speed.
- **Language.**

In the view:

| | 3D | 2D |
|---|---|---|
| left drag | rotate | pan |
| right drag / wheel | zoom | zoom |
| drag a charge | moves it parallel to the screen | moves it in the plane |
| click | on the coloured plane: a field line through that point | a field line through that point |
| wheel on a charge | changes q by 0.25 nC | changes q by 0.25 nC |
| `C` / `M` | add a +2 / −2 nC charge under the mouse | same |
| `T` | flip the sign of the charge under the mouse | same |
| `R` | clear everything | same |

The status bar shows V and |E| under the mouse: anywhere in 2D, on the
coloured plane in 3D.

## The physics

Everything inside is in SI units; the panel shows nC, cm, μC/m², V, V/m, pF.

**Field lines.**
- The number of lines leaving a source is proportional to its charge
  ("lines per nC"). Each line ends on a source of the opposite sign or goes
  off to infinity, so this is Gauss's law you can see.
- In 3D the lines start evenly spread on a small sphere around each charge
  (a Fibonacci sphere).
- In 2D they start evenly on a circle in the plane. These are still the
  lines of 3D point charges, cut by the plane, as in the old app.
- From a plate, lines start in proportion to the flux leaving each part of
  each face. That is why only a few leave the outer faces, and why in the
  conductor model they crowd at the edges.
- Lines start from the sign with the larger total charge: forward along E
  from + sources, or backwards from − sources when the negative charge is
  larger.

**Tracing.**
- All lines advance together, with RK4 along the unit vector of E, so the
  parameter is the arc length (this is also what the growth animation
  uses).
- The step is proportional to the distance from the nearest point charge,
  and it is halved where a line bends sharply.
- A line stops at an opposite charge or plate, at a null point (where it
  turns back on itself), or when it leaves the box.

**Uniform-σ plates.** E and V of a uniformly charged rectangle come from
their closed-form integrals, not from summing many small charges. In the
middle of a large capacitor this gives the textbook σ/ε₀ (checked in the
tests).

**Conductor plates.** The plates are held at ±ΔV/2 and Laplace's equation
is solved on a grid. Only one octant is solved:
- the potential is odd across the midplane between the plates and even
  across the other two central planes;
- the grid is fine (at least 10 cells across the gap) near the plates and
  grows towards a grounded box 3 plate sides away;
- the system is solved in finite volumes by conjugate gradients.

The charge of each plate cell is ε₀ times the flux of E out of it. From
that the panel shows Q, C = Q/ΔV (larger than ε₀A/d because of fringing)
and σ, which is larger at the edges.

A 10 cm × 10 cm capacitor 3 cm apart solves in about half a second. Smaller
gaps need a finer grid. Below about d/L = 0.08 the panel raises d to the
smallest gap the grid can handle and says so. In this model the point
charges do not take part.

**Readouts.**
- E at the centre.
- σ/ε₀ with σ = Q/A, the field of infinite plates with the same charge.
- ΔV between the plates' centres.
- Q, C, and ε₀A/d.

## Tests

```bash
uv run pytest
```

**Physics:**
- The plate's closed form against brute-force sums.
- The capacitor's textbook limits.
- Field lines: dipole symmetry, null points, line counts, speed, and
  regression tests for the old app's freezes.
- The conductor model:
  - the plates at ±ΔV/2;
  - convergence of C, and agreement with Kirchhoff's fringing estimate;
  - more charge at the edges;
  - Gauss's law around each plate;
  - the octant against a solve of the whole grid.

**Render tests** draw off-screen with PyVista. **GUI tests** drive the real
window on Qt's offscreen platform, without drawing (VTK has no OpenGL context
there). Some of them send real Qt mouse, wheel and key events to the 3D
view: a click versus a drag, the wheel on a charge versus on empty space,
dragging a charge, and the keys.

## Layout

```
src/field_lines/
  units.py        SI constants, display units, formatting of readouts
  sources.py      point charges and uniformly charged plates (closed form)
  conductors.py   the conductor model: Laplace's equation on one octant
  seeding.py      where lines start (number proportional to charge or flux)
  tracing.py      vectorized RK4 tracing with its stopping rules
  sampling.py     grids for equipotentials, arrows and the coloured plane
  scene.py        what is shown (sources, 2D/3D, capacitor model); no Qt
  presets.py      ready-made setups
  view.py         PyVista actors and the 2D/3D camera
  interaction.py  draggable charges, clicks, screen <-> world
  panel.py        the Qt side panel
  strings.py      every visible text, in Italian and English
  app.py          the main window, which wires everything together
```

To change a label or a translation, edit `strings.py`.
