"""The main window: wires the panel, the 3D view and the scene together."""

import argparse
import copy
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from field_lines import conductors, presets, sampling, strings
from field_lines.conductors import Solver, Sphere
from field_lines.interaction import (
    ChargeHandles,
    MouseEvents,
    ball_under_mouse,
    charge_under_mouse,
    mouse_on_plane,
    pick_on,
)
from field_lines.panel import Panel, ViewOptions
from field_lines.sampling import Exclusion
from field_lines.scene import VIEW_HALF, Scene, charge_radius
from field_lines.sources import PointCharge
from field_lines.strings import tr
from field_lines.units import CM, NC, format_quantity
from field_lines.view import SceneView

FRAME_MS = 33  # growth animation: about 30 frames per second
FIELDS_DELAY_MS = 120  # equipotentials, arrows and plane are recomputed this long after the last change
SOLVE_DELAY_MS = 300  # the conductors are solved again this long after their geometry last changed
KEPT_SOLVERS = 4  # solved geometries kept for going back to them
NEW_SPHERE = dict(radius=1.5 * CM, voltage=1000.0)
WHEEL_STEP = 0.25 * NC  # charge change per wheel notch
NEW_CHARGE = 2 * NC


@dataclass
class FieldViews:
    """What the worker thread computed for one state of the scene."""

    generation: int
    options: ViewOptions
    planar: bool
    equipotentials: object = None
    arrows: object = None
    slice: object = None


class FieldsWorker(QtCore.QObject):
    """Carries results from the worker thread back to the GUI thread."""

    finished = QtCore.Signal(object)


class SolverWorker(QtCore.QObject):
    """Carries a solved geometry (key, conductors.Solver or None) back to the GUI thread."""

    finished = QtCore.Signal(object, object)


class StatusLight(QtWidgets.QWidget):
    """In the status bar: a coloured light and what the program is doing.

    "ready" (green, steady), "busy" (amber, flashing while something is being
    computed in the background) or "error" (red, steady).
    """

    COLOURS = {"ready": "#3a9d3a", "busy": "#f0a000", "error": "#d03030"}
    FLASH_MS = 450

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 8, 0)
        self.light = QtWidgets.QLabel()
        self.light.setFixedSize(14, 14)
        self.label = QtWidgets.QLabel()
        layout.addWidget(self.light)
        layout.addWidget(self.label)
        self.state = ""
        self._lit = True
        self._flash = QtCore.QTimer(self, interval=self.FLASH_MS)
        self._flash.timeout.connect(self._toggle)

    def set_state(self, state: str, text: str) -> None:
        self.label.setText(text)
        if state == self.state:
            return
        self.state = state
        self._lit = True
        self._paint()
        if state == "busy":
            self._flash.start()
        else:
            self._flash.stop()

    def text(self) -> str:
        return self.label.text()

    def flashing(self) -> bool:
        return self._flash.isActive()

    def _toggle(self) -> None:
        self._lit = not self._lit
        self._paint()

    def _paint(self) -> None:
        colour = self.COLOURS.get(self.state, "#999999") if self._lit else "#ececec"
        self.light.setStyleSheet(f"background-color: {colour}; border: 1px solid #888; border-radius: 7px;")


class PanelScroll(QtWidgets.QScrollArea):
    """Scrolls the panel up and down, and is always as wide as the panel needs."""

    def __init__(self, panel: QtWidgets.QWidget):
        super().__init__()
        self.setWidget(panel)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.installEventFilter(self)
        self._fit(panel)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QtCore.QEvent.Type.LayoutRequest:  # the panel's content changed size
            self._fit(watched)
        return False

    def _fit(self, panel: QtWidgets.QWidget) -> None:
        extra = self.verticalScrollBar().sizeHint().width() + 2 * self.frameWidth()
        self.setMinimumWidth(panel.minimumSizeHint().width() + extra)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, preset: str = "dipole", planar: bool = False):
        super().__init__()
        self.closed = False
        self.scene = Scene()
        self.scene.planar = planar
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter)
        self.view = SceneView(self.plotter)
        self.handles = ChargeHandles(self.plotter, self._charge_dragged)
        self.sphere_handles = ChargeHandles(self.plotter, self._sphere_dragged)
        self.mouse = MouseEvents(self.plotter, self._clicked, self._mouse_moved, self._wheel)
        for key, action in (
            ("c", lambda: self._add_charge_at_mouse(1)),
            ("m", lambda: self._add_charge_at_mouse(-1)),
            ("s", self._add_sphere_at_mouse),
            ("t", self._flip_charge_at_mouse),
            ("r", lambda: self.load_preset("empty")),
        ):
            self.plotter.add_key_event(key, action)
        for key in ("q", "b", "C"):  # PyVista defaults: close, box widget, cell picking
            self.plotter.clear_events_for_key(key)

        self.panel = Panel()
        dock = QtWidgets.QDockWidget()
        dock.setObjectName("panel")
        dock.setWidget(PanelScroll(self.panel))
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setTitleBarWidget(QtWidgets.QWidget())
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.status_light = StatusLight()
        self.hint = QtWidgets.QLabel()
        # the hint gives way when the bar is short, not the status next to it
        self.hint.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.readout = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status_light)
        self.statusBar().addWidget(self.hint, 1)
        self.statusBar().addPermanentWidget(self.readout)

        self.speed = 0.1  # m/s of arc length during the growth animation
        self.growth = 0.0  # arc length reached by the growing lines
        self.growing = np.zeros(0, dtype=bool)
        self.animation = QtCore.QTimer(self, interval=FRAME_MS)
        self.animation.timeout.connect(self._grow)
        self._animate_next = False
        self._retrace_timer = QtCore.QTimer(self, singleShot=True, interval=0)
        self._retrace_timer.timeout.connect(self._retrace)
        self.fields_worker = FieldsWorker(self)
        self.fields_worker.finished.connect(self._fields_ready)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._fields_generation = 0
        self._shown_generation = -1  # the generation whose views are on screen
        self._fields_timer = QtCore.QTimer(self, singleShot=True, interval=FIELDS_DELAY_MS)
        self._fields_timer.timeout.connect(self._compute_fields)
        self.solver_worker = SolverWorker(self)
        self.solver_worker.finished.connect(self._solved)
        self._solver = ThreadPoolExecutor(max_workers=1)
        self._solver_future = None
        self._solvers: dict[tuple, Solver] = {}
        self._solve_timer = QtCore.QTimer(self, singleShot=True, interval=SOLVE_DELAY_MS)
        self._solve_timer.timeout.connect(self._start_solve)
        self._touching = False  # spheres touching each other or the plates: nothing can be solved
        self._grow_after_solve = False  # whether the lines grow again once the pending solve is done

        p = self.panel
        p.planar_changed.connect(self.set_planar)
        p.kind_changed.connect(self.set_kind)
        p.sphere_edited.connect(self._sphere_edited)
        p.add_sphere_requested.connect(self._add_sphere_from_panel)
        p.remove_sphere_requested.connect(self._remove_sphere)
        p.preset_chosen.connect(self.load_preset)
        p.charge_edited.connect(self._charge_edited)
        p.add_charge_requested.connect(self._add_charge_from_panel)
        p.remove_requested.connect(self._remove_charge)
        p.capacitor_changed.connect(self._capacitor_edited)
        p.views_changed.connect(self._views_edited)
        p.replay_requested.connect(self.replay)
        p.speed_changed.connect(self._set_speed)
        p.language_changed.connect(self.set_language)

        self.panel.set_planar(planar)
        self.view.set_planar(planar)
        self.retranslate()
        self.load_preset(preset)

    # --- scene changes --------------------------------------------------

    def load_preset(self, key: str) -> None:
        presets.load(self.scene, key)
        self.panel.set_preset(key)
        self._update_conductor(grow=True)
        self.refresh(animate=True)

    def set_kind(self, kind: str) -> None:
        """Switch between the point-charge model and the conductors model."""
        self.scene.kind = kind
        self.scene.click_seeds = []
        self._update_conductor(grow=True)
        self.refresh(animate=True)

    def set_planar(self, planar: bool) -> None:
        self.scene.set_planar(planar)
        self._update_conductor(grow=True)  # the capacitor may have turned
        self.panel.set_planar(planar)
        self.view.set_planar(planar)
        self.hint.setText(tr("hint_2d" if planar else "hint_3d"))
        self.refresh(animate=True)

    def refresh(self, animate: bool) -> None:
        """Show the whole scene again after its sources changed."""
        self.panel.set_kind(self.scene.kind)
        self.panel.set_charges(self.scene.charges)
        self.panel.set_spheres(self.scene.spheres, self._sphere_charges())
        self.panel.set_capacitor(self.scene.capacitor)
        self.handles.set_charges(self.scene.active_charges())
        self.sphere_handles.set_spheres(self.scene.active_spheres())
        self.show_plates()
        self._update_readouts()
        self.request_lines(animate)

    def show_plates(self) -> None:
        """The plates (coloured by σ if the panel asks for it) and the metallic spheres (always by σ)."""
        options = self.panel.view_options()
        densities = self.scene.plate_densities() if options.plate_sigma else None
        self.view.set_plates(
            self.scene.plates(),
            densities,
            self.scene.planar,
            self.scene.sphere_densities(),
            options.sigma_scale,
            options.sigma_palette,
        )

    def _sphere_charges(self) -> list[float | None]:
        return [self.scene.sphere_charge(k) for k in range(len(self.scene.spheres))]

    def _computing(self) -> bool:
        """Whether the conductors are being solved (their old lines stay until then)."""
        scene = self.scene
        has_conductors = scene.capacitor is not None or scene.spheres
        return scene.uses_conductor and scene.conductor is None and has_conductors and not self._touching

    def request_lines(self, animate: bool) -> None:
        """Trace the lines again, once the current Qt events are handled (many requests, one trace)."""
        self._animate_next = self._animate_next or animate
        self._retrace_timer.start()

    def _retrace(self) -> None:
        if self._computing():
            return  # keep the old lines until the new charges are known
        lines = self.scene.lines()
        self.view.set_lines(lines)
        if self._animate_next:
            self.growing = np.ones(len(lines), dtype=bool)
            self._start_growth()
        else:
            self.animation.stop()
            self.growing = np.zeros(len(lines), dtype=bool)
            self.view.show_lines()
            self.plotter.render()
        self._animate_next = False
        self.schedule_fields()
        self._update_readouts()

    # --- equipotentials, arrows, coloured plane -------------------------

    def schedule_fields(self) -> None:
        """Recompute the sampled views a little later, in the worker thread."""
        self._fields_generation += 1
        self._fields_timer.start()
        self._update_status()

    def _compute_fields(self) -> None:
        options = self.panel.view_options()
        result = FieldViews(self._fields_generation, options, self.scene.planar)
        model = self.scene.snapshot()
        if model is None or not (options.equipotentials or options.arrows or options.slice):
            self._fields_ready(result)
            return
        exclusions = [Exclusion(c.position.copy(), charge_radius(c.q)) for c in self.scene.active_charges()]
        exclusions += [Exclusion(s.center.copy(), s.radius) for s in self.scene.active_spheres()]
        conductors_model = self.scene.uses_conductor

        def job():
            try:
                if options.equipotentials:
                    resolution = None
                    if conductors_model and not result.planar:
                        resolution = sampling.EQUIPOTENTIAL_RES_CONDUCTORS
                    result.equipotentials, _ = sampling.equipotentials(
                        model, result.planar, options.levels, exclusions, resolution
                    )
                if options.arrows:
                    result.arrows = sampling.arrows(model, result.planar, exclusions)
                if options.slice:
                    result.slice = sampling.slice_plane(model, options.slice_axis, options.slice_position, result.planar)
            except Exception:
                traceback.print_exc()
                return
            self.fields_worker.finished.emit(result)

        self._executor.submit(job)

    def _fields_ready(self, result: FieldViews) -> None:
        if self.closed or result.generation != self._fields_generation:
            return  # the scene changed meanwhile (a newer result is on its way), or the window is gone
        self.view.set_equipotentials(result.equipotentials, result.planar)
        self.view.set_arrows(result.arrows)
        self.view.set_slice(result.slice, result.options.slice_quantity)
        self._shown_generation = result.generation
        self._update_status()
        self.plotter.render()

    def replay(self) -> None:
        self.growing = np.ones(len(self.view.lines), dtype=bool)
        self._start_growth()

    def _start_growth(self) -> None:
        self.growth = 0.0
        self._show_growth()
        self.animation.start()

    def _grow(self) -> None:
        self.growth += self.speed * FRAME_MS / 1000
        self._show_growth()
        if self.growth >= self.view.max_length:
            self.animation.stop()

    def _show_growth(self) -> None:
        limits = np.where(self.growing, self.growth, np.inf)
        self.view.show_lines(limits)
        self.plotter.render()

    def _set_speed(self, speed: float) -> None:
        self.speed = speed

    # --- point charges --------------------------------------------------

    def _charge_dragged(self, index: int, position: np.ndarray, finished: bool) -> None:
        if self.scene.planar:
            position[2] = 0.0
            self.handles.move(index, position)
        charge = self.scene.charges[index]
        charge.position = position
        self.panel.update_charge(index, charge)
        self.request_lines(animate=False)

    def _charge_edited(self, row: int, q: float, x: float, y: float, z: float) -> None:
        charge = self.scene.charges[row]
        sign_or_size_changed = q * NC != charge.q
        charge.q = q * NC
        charge.position = np.array([x, y, 0.0 if self.scene.planar else z]) * CM
        if sign_or_size_changed:
            self.handles.set_charges(self.scene.active_charges())
        else:
            self.handles.move(row, charge.position)
        self.request_lines(animate=False)

    def add_charge(self, q: float, position: np.ndarray) -> None:
        if self.scene.planar:
            position = np.array([position[0], position[1], 0.0])
        self.scene.charges.append(PointCharge(q, np.clip(position, -VIEW_HALF, VIEW_HALF)))
        self.refresh(animate=True)

    def _add_charge_from_panel(self, sign: int) -> None:
        rng = np.random.default_rng()
        position = rng.uniform(-0.6, 0.6, 3) * VIEW_HALF
        self.add_charge(sign * NEW_CHARGE, position)

    def _remove_charge(self, row: int) -> None:
        del self.scene.charges[row]
        self.refresh(animate=True)

    def _add_charge_at_mouse(self, sign: int) -> None:
        if self.scene.uses_conductor:
            return
        point = self._mouse_point(self.plotter.iren.interactor.GetEventPosition())
        if point is not None:
            self.add_charge(sign * NEW_CHARGE, point)

    def _flip_charge_at_mouse(self) -> None:
        index = self._charge_at(self.plotter.iren.interactor.GetEventPosition())
        if index is not None:
            self.scene.charges[index].q *= -1
            self.refresh(animate=False)

    def _charge_at(self, display) -> int | None:
        if self.scene.uses_conductor:
            return None
        return charge_under_mouse(self.plotter, self.scene.charges, display)

    def _sphere_at(self, display) -> int | None:
        balls = [(s.center, s.radius) for s in self.scene.active_spheres()]
        return ball_under_mouse(self.plotter, balls, display)

    def _wheel(self, display, step: int) -> bool:
        """A wheel notch on a charge changes its charge (and does not zoom)."""
        index = self._charge_at(display)
        if index is None:
            return False
        charge = self.scene.charges[index]
        charge.q += step * WHEEL_STEP
        if abs(charge.q) < 1e-3 * NC:
            charge.q = step * WHEEL_STEP
        self.panel.update_charge(index, charge)
        self.handles.set_charges(self.scene.active_charges())
        self.request_lines(animate=False)
        return True

    # --- mouse on the view ----------------------------------------------

    def _mouse_point(self, display) -> np.ndarray | None:
        """The world point under the mouse: on z = 0 in 2D, on the plane through the origin facing the camera in 3D."""
        if self.scene.planar:
            normal = (0.0, 0.0, 1.0)
        else:
            normal = np.array(self.plotter.camera.direction_of_projection)
        return mouse_on_plane(self.plotter, display, (0.0, 0.0, 0.0), normal)

    def _point_under_mouse(self, display) -> np.ndarray | None:
        """2D: the point of the plane under the mouse; 3D: the point of the coloured plane, if it is shown."""
        if self.scene.planar:
            point = self._mouse_point(display)
            if point is not None:
                point[2] = 0.0
            return point
        if self.view.slice_actor is not None:
            return pick_on(self.plotter, self.view.slice_actor, display)
        return None

    def _clicked(self, display) -> None:
        if self._charge_at(display) is not None or self._sphere_at(display) is not None:
            return  # a click on a charge or a sphere (that grabbed it) is not a click on the space behind it
        point = self._point_under_mouse(display)
        if point is None or np.any(np.abs(point) > VIEW_HALF):
            return
        self.scene.click_seeds.append(point)
        self._add_click_line()

    def _add_click_line(self) -> None:
        """Trace again and let only the new line (two halves) grow."""
        self._retrace_timer.stop()
        lines = self.scene.lines()
        old = len(self.growing)
        self.view.set_lines(lines)
        self.growing = np.zeros(len(lines), dtype=bool)
        self.growing[old:] = True
        self._start_growth()

    def _mouse_moved(self, display) -> None:
        model = self.scene.model()
        point = self._point_under_mouse(display)
        if model is None or point is None:
            self.readout.setText("")
            return
        p = point[None, :]
        V = model.potential(p)[0]
        E = np.linalg.norm(model.field(p)[0])
        self.readout.setText(tr("readout", V=format_quantity(V, "V"), E=format_quantity(E, "V/m")))

    # --- panel ----------------------------------------------------------

    def _capacitor_edited(self) -> None:
        normal = self.scene.capacitor.normal if self.scene.capacitor else ("x" if self.scene.planar else "z")
        self.scene.capacitor = self.panel.capacitor(normal)
        self._update_conductor()
        self.refresh(animate=False)

    # --- metallic spheres -------------------------------------------------

    def _sphere_dragged(self, index: int, position: np.ndarray, finished: bool) -> None:
        if self.scene.planar:
            position[2] = 0.0
            self.sphere_handles.move(index, position)
        sphere = self.scene.spheres[index]
        sphere.center = position
        self.panel.update_sphere(index, sphere)
        self.view.move_sphere(index, position)
        self.plotter.render()
        if finished:
            # its charges change once it is let go; not inside the widget's own release event,
            # because refreshing rebuilds the handles, the widget among them
            QtCore.QTimer.singleShot(0, self._sphere_released)

    def _sphere_released(self) -> None:
        if not self.closed:
            self._update_conductor()
            self.refresh(animate=False)

    def _sphere_edited(self, row: int, voltage: float, radius: float, x: float, y: float, z: float) -> None:
        sphere = self.scene.spheres[row]
        sphere.voltage = voltage
        sphere.radius = radius * CM
        sphere.center = np.array([x, y, 0.0 if self.scene.planar else z]) * CM
        self._update_conductor()
        self.refresh(animate=False)

    def add_sphere(self, position: np.ndarray) -> None:
        if self.scene.planar:
            position = np.array([position[0], position[1], 0.0])
        center = np.clip(position, -VIEW_HALF, VIEW_HALF)
        self.scene.spheres.append(Sphere(center, NEW_SPHERE["radius"], NEW_SPHERE["voltage"]))
        self._update_conductor(grow=True)
        self.refresh(animate=True)

    def _add_sphere_from_panel(self) -> None:
        self.add_sphere(np.random.default_rng().uniform(-0.6, 0.6, 3) * VIEW_HALF)

    def _add_sphere_at_mouse(self) -> None:
        if not self.scene.uses_conductor:
            return
        point = self._mouse_point(self.plotter.iren.interactor.GetEventPosition())
        if point is not None:
            self.add_sphere(point)

    def _remove_sphere(self, row: int) -> None:
        del self.scene.spheres[row]
        self._update_conductor(grow=True)
        self.refresh(animate=True)

    # --- the conductors model ---------------------------------------------

    def _update_conductor(self, grow: bool = False) -> None:
        """Give the scene the charges of its conductors: at once if their geometry was solved, else soon.

        grow: the lines grow again once solved (a new setup), instead of just being updated (an edit).
        """
        self._grow_after_solve = grow
        scene = self.scene
        scene.conductor = None
        self._touching = False
        if not scene.uses_conductor or (scene.capacitor is None and not scene.spheres):
            return
        if conductors.touching(scene.conductor_plates(), scene.spheres):
            self._touching = True
            return
        solver = self._solvers.get(scene.geometry_key())
        if solver is None:
            self._solve_timer.start()
        else:  # new potentials on a known geometry: a quick solve
            scene.conductor = solver.solve(scene.voltages())

    def _start_solve(self) -> None:
        if not self._computing():
            return
        key = self.scene.geometry_key()
        plates = self.scene.conductor_plates()
        spheres = copy.deepcopy(self.scene.spheres)
        if self._solver_future is not None:
            self._solver_future.cancel()  # a solve for an older geometry that has not started yet

        def job():
            try:
                solver = Solver(plates, spheres)
            except Exception:
                traceback.print_exc()
                solver = None
            self.solver_worker.finished.emit(key, solver)

        self._solver_future = self._solver.submit(job)

    def _solved(self, key, solver) -> None:
        if self.closed or solver is None:
            return
        self._solvers[key] = solver
        while len(self._solvers) > KEPT_SOLVERS:
            del self._solvers[next(iter(self._solvers))]  # the oldest
        if self.scene.uses_conductor and self.scene.geometry_key() == key:
            grow = self._grow_after_solve
            self._update_conductor()
            self.refresh(animate=grow)

    def _update_readouts(self) -> None:
        values = self.scene.capacitor_readouts()
        texts = {}
        if values is not None:
            units = dict(E_center="V/m", E_ideal="V/m", dV="V", Q="C", C="F", C_ideal="F")
            texts = {f"readout_{name}": format_quantity(values[name], unit) for name, unit in units.items()}
        self.panel.set_readouts(texts)
        self.panel.set_sphere_charges(self._sphere_charges())
        self._update_status()

    def _update_status(self) -> None:
        """The light and text in the status bar: what the program is doing now."""
        updating_views = self._fields_timer.isActive() or self._shown_generation != self._fields_generation
        if self._touching:
            self.status_light.set_state("error", tr("spheres_touch"))
        elif self._computing():
            self.status_light.set_state("busy", tr("computing"))
        elif updating_views:
            self.status_light.set_state("busy", tr("updating_views"))
        else:
            self.status_light.set_state("ready", tr("ready"))

    def _views_edited(self) -> None:
        options = self.panel.view_options()
        self.view.set_lines_visible(options.lines)
        self.show_plates()
        if options.lines_per_nc != self.scene.lines_per_nc:
            self.scene.lines_per_nc = options.lines_per_nc
            self.request_lines(animate=False)
        self.schedule_fields()
        self.plotter.render()

    # --- language -------------------------------------------------------

    def set_language(self, language: str) -> None:
        strings.set_language(language)
        self.retranslate()
        self._update_readouts()
        self.show_plates()  # the colour bar title
        self.schedule_fields()  # the colour bar titles

    def retranslate(self) -> None:
        self.setWindowTitle(tr("window_title"))
        self.panel.retranslate()
        self.hint.setText(tr("hint_2d" if self.scene.planar else "hint_3d"))

    def closeEvent(self, event) -> None:
        self.closed = True
        for timer in (self.animation, self._retrace_timer, self._fields_timer, self._solve_timer):
            timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._solver.shutdown(wait=False, cancel_futures=True)
        self.handles.clear()
        self.sphere_handles.clear()
        self.plotter.close()
        super().closeEvent(event)


def default_language() -> str:
    return "it" if QtCore.QLocale.system().name().startswith("it") else "en"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="field-lines", description=strings.STRINGS["window_title"]["en"])
    parser.add_argument("--lang", choices=list(strings.LANGUAGES), help="interface language (default: system)")
    parser.add_argument("--mode", choices=["3d", "2d"], default="3d", help="start in 3D space or in the 2D plane")
    parser.add_argument("--preset", choices=list(presets.PRESETS), default="dipole", help="starting setup")
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    strings.set_language(args.lang or default_language())
    window = MainWindow(preset=args.preset, planar=args.mode == "2d")
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
