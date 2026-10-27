"""The main window: wires the panel, the 3D view and the scene together."""

import argparse
import math
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from field_lines import conductors, presets, sampling, strings
from field_lines.conductors import ConductorSolution
from field_lines.interaction import ChargeHandles, MouseEvents, charge_under_mouse, mouse_on_plane, pick_on
from field_lines.panel import Panel, ViewOptions
from field_lines.sampling import Exclusion
from field_lines.scene import VIEW_HALF, Capacitor, Scene, charge_radius
from field_lines.sources import PointCharge
from field_lines.strings import tr
from field_lines.units import CM, NC, format_quantity
from field_lines.view import SceneView

FRAME_MS = 33  # growth animation: about 30 frames per second
FIELDS_DELAY_MS = 120  # equipotentials, arrows and plane are recomputed this long after the last change
SOLVE_DELAY_MS = 300  # the conductor model is solved this long after the capacitor's last change
KEPT_SOLUTIONS = 4  # solved geometries kept for going back to them (up to ~20 MB each)
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
    """Carries a solved conductor model (geometry key, OctantGrid or None) back to the GUI thread."""

    finished = QtCore.Signal(object, object)


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


def geometry_key(capacitor: Capacitor) -> tuple[float, float, float]:
    return (round(capacitor.width, 9), round(capacitor.height, 9), round(capacitor.separation, 9))


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
        self.mouse = MouseEvents(self.plotter, self._clicked, self._mouse_moved, self._wheel)
        for key, action in (
            ("c", lambda: self._add_charge_at_mouse(1)),
            ("m", lambda: self._add_charge_at_mouse(-1)),
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
        self.hint = QtWidgets.QLabel()
        self.readout = QtWidgets.QLabel()
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
        self._solutions: dict[tuple[float, float, float], conductors.OctantGrid] = {}
        self._solve_timer = QtCore.QTimer(self, singleShot=True, interval=SOLVE_DELAY_MS)
        self._solve_timer.timeout.connect(self._start_solve)
        self._clamped_to: float | None = None  # separation forced on the conductor model, m

        p = self.panel
        p.planar_changed.connect(self.set_planar)
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
        self._clamped_to = None
        self._update_conductor()
        self.refresh(animate=True)

    def set_planar(self, planar: bool) -> None:
        self.scene.set_planar(planar)
        self._update_conductor()  # the capacitor may have turned
        self.panel.set_planar(planar)
        self.view.set_planar(planar)
        self.hint.setText(tr("hint_2d" if planar else "hint_3d"))
        self.refresh(animate=True)

    def refresh(self, animate: bool) -> None:
        """Show the whole scene again after its sources changed."""
        charges_enabled = not self.scene.uses_conductor
        self.panel.set_charges(self.scene.charges, enabled=charges_enabled)
        self.panel.set_capacitor(self.scene.capacitor)
        self.handles.set_charges(self.scene.active_charges())
        self.show_plates()
        self._update_readouts()
        self.request_lines(animate)

    def show_plates(self) -> None:
        """The plates, coloured by their surface charge density if the panel asks for it."""
        densities = self.scene.plate_densities() if self.panel.view_options().plate_sigma else None
        self.view.set_plates(self.scene.plates(), densities, self.scene.planar)

    def request_lines(self, animate: bool) -> None:
        """Trace the lines again, once the current Qt events are handled (many requests, one trace)."""
        self._animate_next = self._animate_next or animate
        self._retrace_timer.start()

    def _retrace(self) -> None:
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

    def _compute_fields(self) -> None:
        options = self.panel.view_options()
        result = FieldViews(self._fields_generation, options, self.scene.planar)
        model = self.scene.snapshot()
        if model is None or not (options.equipotentials or options.arrows or options.slice):
            self._fields_ready(result)
            return
        exclusions = [Exclusion(c.position.copy(), charge_radius(c.q)) for c in self.scene.active_charges()]

        def job():
            try:
                if options.equipotentials:
                    result.equipotentials, _ = sampling.equipotentials(model, result.planar, options.levels, exclusions)
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
        if self._charge_at(display) is not None:
            return  # a click on a charge (that grabbed it) is not a click on the space behind it
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
        capacitor = self.panel.capacitor(normal)
        self._clamped_to = None
        if capacitor is not None and capacitor.model == "conductor":
            smallest = conductors.min_separation(capacitor.width, capacitor.height)
            if capacitor.separation < smallest:
                capacitor.separation = math.ceil(smallest / CM * 10) / 10 * CM  # the spin box shows 0.1 cm
                self._clamped_to = capacitor.separation
        self.scene.capacitor = capacitor
        self._update_conductor()
        self.refresh(animate=False)

    # --- the conductor model ----------------------------------------------

    def _update_conductor(self) -> None:
        """Give the scene the conductor solution of its capacitor, or start computing it."""
        capacitor = self.scene.capacitor
        if not self.scene.uses_conductor:
            self.scene.conductor = None
            return
        grid = self._solutions.get(geometry_key(capacitor))
        if grid is None:
            self.scene.conductor = None
            self._solve_timer.start()
        else:
            self.scene.conductor = ConductorSolution(grid, capacitor.voltage, capacitor.normal)

    def _start_solve(self) -> None:
        if not self.scene.uses_conductor:
            return
        key = geometry_key(self.scene.capacitor)
        if self._solver_future is not None:
            self._solver_future.cancel()  # a solve for an older geometry that has not started yet

        def job():
            try:
                grid = conductors.solve(*key)
            except Exception:
                traceback.print_exc()
                grid = None
            self.solver_worker.finished.emit(key, grid)

        self._solver_future = self._solver.submit(job)

    def _solved(self, key, grid) -> None:
        if self.closed or grid is None:
            return
        self._solutions[key] = grid
        while len(self._solutions) > KEPT_SOLUTIONS:
            del self._solutions[next(iter(self._solutions))]  # the oldest
        if self.scene.uses_conductor and geometry_key(self.scene.capacitor) == key:
            self._update_conductor()
            self.refresh(animate=True)

    def _update_readouts(self) -> None:
        values = self.scene.capacitor_readouts()
        texts = {}
        if values is not None:
            units = dict(E_center="V/m", E_ideal="V/m", dV="V", Q="C", C="F", C_ideal="F")
            texts = {f"readout_{name}": format_quantity(values[name], unit) for name, unit in units.items()}
        status = []
        if self.scene.uses_conductor and self.scene.conductor is None:
            status.append(tr("computing"))
        if self._clamped_to is not None:
            status.append(tr("clamped", d=f"{self._clamped_to / CM:.1f}"))
        self.panel.set_readouts(texts, " ".join(status))

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
