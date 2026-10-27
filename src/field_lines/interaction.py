"""Mouse interaction in the 3D view: draggable charges, clicks, and screen <-> world conversions."""

from collections.abc import Callable, Sequence

import numpy as np
import pyvista as pv
from PySide6 import QtCore
from vtkmodules.vtkInteractionWidgets import vtkSphereWidget
from vtkmodules.vtkRenderingCore import vtkCellPicker

from field_lines.scene import charge_radius
from field_lines.sources import PointCharge
from field_lines.view import sign_color

CLICK_TOLERANCE = 4  # pixels the mouse may move between press and release for a click


class ChargeHandles:
    """One draggable sphere per point charge; on_move(index, position, finished) reports drags."""

    def __init__(self, plotter: pv.Plotter, on_move: Callable[[int, np.ndarray, bool], None]):
        self.pl = plotter
        self.on_move = on_move
        self.widgets: list[vtkSphereWidget] = []

    def set_charges(self, charges: Sequence[PointCharge], enabled: bool = True) -> None:
        self.clear()
        for index, charge in enumerate(charges):
            widget = vtkSphereWidget()
            widget.SetInteractor(self.pl.iren.interactor)
            widget.SetCurrentRenderer(self.pl.renderer)
            widget.SetRepresentationToSurface()
            widget.SetThetaResolution(32)
            widget.SetPhiResolution(24)
            widget.ScaleOff()  # the right button would resize the sphere
            color = sign_color(charge.q)
            widget.GetSphereProperty().SetColor(color)
            widget.GetSelectedSphereProperty().SetColor(tuple(0.5 + 0.5 * c for c in color))
            widget.SetRadius(charge_radius(charge.q))
            widget.SetCenter(*charge.position)
            widget.AddObserver("InteractionEvent", self._callback(index, finished=False))
            widget.AddObserver("EndInteractionEvent", self._callback(index, finished=True))
            if enabled:
                widget.On()
            self.widgets.append(widget)

    def _callback(self, index: int, finished: bool):
        def callback(widget, _event):
            self.on_move(index, np.array(widget.GetCenter()), finished)

        return callback

    def move(self, index: int, position: np.ndarray) -> None:
        self.widgets[index].SetCenter(*position)

    def clear(self) -> None:
        for widget in self.widgets:
            widget.Off()
            widget.SetInteractor(None)
        self.widgets = []


def mouse_ray(plotter: pv.Plotter, display: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The line of world points under a display position: (origin, unit direction)."""
    renderer = plotter.renderer
    ends = []
    for depth in (0.0, 1.0):
        renderer.SetDisplayPoint(display[0], display[1], depth)
        renderer.DisplayToWorld()
        x, y, z, w = renderer.GetWorldPoint()
        ends.append(np.array([x, y, z]) / w)
    direction = ends[1] - ends[0]
    return ends[0], direction / np.linalg.norm(direction)


def mouse_on_plane(plotter: pv.Plotter, display, point, normal) -> np.ndarray | None:
    """Where the line under the mouse meets the plane through point with the given normal."""
    origin, direction = mouse_ray(plotter, display)
    normal = np.asarray(normal, dtype=float)
    denom = direction @ normal
    if abs(denom) < 1e-12:
        return None
    t = (np.asarray(point, dtype=float) - origin) @ normal / denom
    return origin + t * direction


def pick_on(plotter: pv.Plotter, actor, display) -> np.ndarray | None:
    """The point of the actor under the display position, or None."""
    picker = vtkCellPicker()
    picker.SetTolerance(0.0005)
    picker.PickFromListOn()
    picker.AddPickList(actor)
    if picker.Pick(display[0], display[1], 0, plotter.renderer):
        return np.array(picker.GetPickPosition())
    return None


def to_display(plotter: pv.Plotter, point: np.ndarray) -> np.ndarray:
    renderer = plotter.renderer
    renderer.SetWorldPoint(*point, 1.0)
    renderer.WorldToDisplay()
    return np.array(renderer.GetDisplayPoint()[:2])


def charge_under_mouse(plotter: pv.Plotter, charges: Sequence[PointCharge], display) -> int | None:
    """Index of the charge drawn under the display position, the nearest one if several."""
    up = np.array(plotter.camera.up)
    best, best_distance = None, np.inf
    for index, charge in enumerate(charges):
        centre = to_display(plotter, charge.position)
        edge = to_display(plotter, charge.position + charge_radius(charge.q) * up)
        distance = np.linalg.norm(centre - np.asarray(display))
        if distance <= max(np.linalg.norm(edge - centre), 6.0) and distance < best_distance:
            best, best_distance = index, distance
    return best


class MouseEvents(QtCore.QObject):
    """Clicks (press and release without dragging), mouse moves and wheel turns on the view.

    Clicks are watched on the Qt widget, before VTK: the camera styles grab
    VTK's event focus on a button press, and the release then never reaches
    other VTK observers. Positions passed on are VTK display coordinates.
    """

    def __init__(
        self,
        plotter,
        on_click: Callable[[tuple[int, int]], None],
        on_move: Callable[[tuple[int, int]], None],
        on_wheel: Callable[[tuple[int, int], int], bool],
    ):
        super().__init__(plotter)
        self.iren = plotter.iren.interactor
        self.on_click, self.on_move, self.on_wheel = on_click, on_move, on_wheel
        self.pressed_at = None
        plotter.installEventFilter(self)
        self.iren.AddObserver("MouseMoveEvent", self._move)
        # above the camera style, so that a wheel turn on a charge does not also zoom
        self._wheel_tags = {
            self.iren.AddObserver("MouseWheelForwardEvent", self._wheel, 10.0): 1,
            self.iren.AddObserver("MouseWheelBackwardEvent", self._wheel, 10.0): -1,
        }
        # VTK's own one-letter keys (w: wireframe, e/q: exit, 3: stereo...) are not wanted here
        self._char_tag = self.iren.AddObserver("CharEvent", self._swallow, 10.0)

    def eventFilter(self, watched, event) -> bool:
        left = QtCore.Qt.MouseButton.LeftButton
        if event.type() == QtCore.QEvent.Type.Wheel:
            # pyvistaqt passes wheel turns to VTK without their position: VTK would use the last mouse move
            ratio = watched.devicePixelRatioF()
            self.iren.SetEventPositionFlipY(round(event.position().x() * ratio), round(event.position().y() * ratio))
        elif event.type() == QtCore.QEvent.Type.MouseButtonPress and event.button() == left:
            self.pressed_at = event.position()
        elif event.type() == QtCore.QEvent.Type.MouseButtonRelease and event.button() == left:
            if self.pressed_at is not None:
                moved = (event.position() - self.pressed_at).manhattanLength()
                self.pressed_at = None
                if moved <= CLICK_TOLERANCE:
                    # once the widget has passed the release to VTK, whose position is then up to date
                    QtCore.QTimer.singleShot(0, lambda: self.on_click(self.iren.GetEventPosition()))
        return False

    def _move(self, _obj, _event):
        self.on_move(self.iren.GetEventPosition())

    def _wheel(self, obj, event):
        step = 1 if event == "MouseWheelForwardEvent" else -1
        if self.on_wheel(self.iren.GetEventPosition(), step):
            for tag, tag_step in self._wheel_tags.items():
                if tag_step == step:
                    self.iren.GetCommand(tag).SetAbortFlag(1)

    def _swallow(self, _obj, _event):
        self.iren.GetCommand(self._char_tag).SetAbortFlag(1)
