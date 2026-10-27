import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from field_lines.interaction import to_display
from field_lines.strings import tr
from field_lines.units import CM, NC


def test_preset_fills_the_table_and_the_view(window):
    assert window.panel.table.rowCount() == 2
    assert window.panel.preset_combo.currentData() == "dipole"
    assert len(window.view.lines) == 20
    assert len(window.handles.widgets) == 2


def test_editing_the_table_moves_the_charge_and_retraces(window, qtbot):
    before = [line.points.copy() for line in window.view.lines]
    window.panel.table.cellWidget(0, 2).setValue(2.0)  # y of the first charge, cm
    window._retrace()
    assert window.scene.charges[0].position[1] == 2.0 * CM
    assert np.array_equal(window.handles.widgets[0].GetCenter(), window.scene.charges[0].position)
    assert any(not np.array_equal(a, b.points) for a, b in zip(before, window.view.lines))


def test_changing_q_in_the_table_changes_the_number_of_lines(window):
    window.panel.table.cellWidget(0, 0).setValue(3.0)
    window._retrace()
    assert window.scene.charges[0].q == 3 * NC
    assert len(window.view.lines) == 30


def test_dragging_updates_scene_and_table(window):
    window._charge_dragged(1, np.array([0.05, 0.02, 0.01]), False)
    assert np.allclose(window.scene.charges[1].position, [0.05, 0.02, 0.01])
    assert window.panel.table.cellWidget(1, 1).value() == 5.0


def test_dragging_in_2d_stays_in_the_plane(window):
    window.panel.radio_2d.setChecked(True)
    window._charge_dragged(0, np.array([0.01, 0.02, 0.03]), True)
    assert window.scene.charges[0].position[2] == 0.0


def test_2d_switch_locks_the_camera_and_hides_z(window):
    window.panel.radio_2d.setChecked(True)
    assert window.scene.planar
    assert window.plotter.camera.parallel_projection
    assert window.panel.table.isColumnHidden(3)
    window.panel.radio_3d.setChecked(True)
    assert not window.scene.planar
    assert not window.plotter.camera.parallel_projection
    assert not window.panel.table.isColumnHidden(3)


def test_add_and_remove_charges(window, qtbot):
    qtbot.mouseClick(window.panel.add_negative, QtCore.Qt.MouseButton.LeftButton)
    assert len(window.scene.charges) == 3 and window.scene.charges[-1].q < 0
    assert window.panel.table.rowCount() == 3
    qtbot.mouseClick(window.panel.table.cellWidget(0, 4), QtCore.Qt.MouseButton.LeftButton)
    assert len(window.scene.charges) == 2 and window.panel.table.rowCount() == 2


def test_click_in_2d_adds_a_line_through_the_point(window):
    window.panel.radio_2d.setChecked(True)
    window._retrace()
    before = len(window.view.lines)
    window.plotter.render()
    display = to_display(window.plotter, np.array([0.0, 0.04, 0.0]))
    window._clicked(tuple(display))
    assert len(window.scene.click_seeds) == 1
    np.testing.assert_allclose(window.scene.click_seeds[0], [0.0, 0.04, 0.0], atol=2e-4)
    assert len(window.view.lines) == before + 2
    assert window.growing.sum() == 2  # only the new line grows


def test_wheel_on_a_charge_changes_it_instead_of_zooming(window):
    window.panel.radio_2d.setChecked(True)
    window.plotter.render()
    display = to_display(window.plotter, window.scene.charges[0].position)
    assert window._wheel(tuple(display), 1)
    assert window.scene.charges[0].q == 2.25 * NC
    assert window.panel.table.cellWidget(0, 0).value() == 2.25
    assert not window._wheel((5, 5), 1)  # empty space: let the camera zoom


def test_language_switch_changes_every_label(window):
    title_it = window.windowTitle()
    window.panel.language_combo.setCurrentIndex(window.panel.language_combo.findData("en"))
    assert window.windowTitle() == tr("window_title") == "Electric field lines" != title_it
    getters = {"setText": "text", "setTitle": "title", "setToolTip": "toolTip"}
    for setter, key in window.panel._texts:
        assert getattr(setter.__self__, getters[setter.__name__])() == tr(key)
    assert window.panel.preset_combo.itemText(window.panel.preset_combo.findData("dipole")) == "Dipole"


def wait_for_fields(window, qtbot):
    """Let the debounced worker thread deliver the sampled views."""
    window._fields_timer.setInterval(0)
    window.schedule_fields()

    def settled():
        pending = window._retrace_timer.isActive() or window._fields_timer.isActive()
        return not pending and window._shown_generation == window._fields_generation

    qtbot.waitUntil(settled, timeout=10000)


def test_view_checkboxes_add_and_remove_actors(window, qtbot):
    for box in (window.panel.equipotentials_box, window.panel.arrows_box, window.panel.slice_box):
        qtbot.mouseClick(box, QtCore.Qt.MouseButton.LeftButton)
    wait_for_fields(window, qtbot)
    for name in ("equipotentials", "arrows", "slice"):
        assert name in window.plotter.actors
    qtbot.mouseClick(window.panel.lines_box, QtCore.Qt.MouseButton.LeftButton)
    assert not window.view.line_actor.visibility
    for box in (window.panel.equipotentials_box, window.panel.arrows_box, window.panel.slice_box):
        box.setChecked(False)
    wait_for_fields(window, qtbot)
    for name in ("equipotentials", "arrows", "slice"):
        assert name not in window.plotter.actors


def test_changing_the_plane_quantity_replaces_its_colour_bar(window, qtbot):
    window.panel.slice_box.setChecked(True)
    wait_for_fields(window, qtbot)
    assert tr("bar_V") in window.plotter.scalar_bars
    window.panel.slice_quantity.setCurrentIndex(1)
    wait_for_fields(window, qtbot)
    assert tr("bar_slice_E") in window.plotter.scalar_bars
    assert tr("bar_V") not in window.plotter.scalar_bars


def test_click_on_the_plane_in_3d_seeds_a_line_and_reads_out(window, qtbot):
    window.panel.slice_box.setChecked(True)
    wait_for_fields(window, qtbot)
    window.plotter.render()
    display = to_display(window.plotter, np.array([0.0, 0.04, 0.0]))
    window._mouse_moved(tuple(display))
    assert "V =" in window.readout.text()
    window._clicked(tuple(display))
    assert len(window.scene.click_seeds) == 1
    np.testing.assert_allclose(window.scene.click_seeds[0], [0.0, 0.04, 0.0], atol=1e-3)


def test_capacitor_preset_shows_plates_and_readouts(window):
    window.load_preset("capacitor")
    window._retrace()
    assert len(window.view.plate_actors) == 2
    texts = {key: label.text() for key, label in window.panel.readouts.items()}
    assert texts["readout_Q"] == "2.00 nC"
    assert texts["readout_E_ideal"] == "22.6 kV/m"  # σ/ε₀ with σ = 0.2 μC/m²
    assert texts["readout_C_ideal"] == "2.95 pF"


def test_conductor_model_is_solved_in_the_background(window, qtbot):
    window.load_preset("capacitor")
    window.panel.model_combo.setCurrentIndex(window.panel.model_combo.findData("conductor"))
    assert window.scene.uses_conductor
    assert window.scene.conductor is None
    assert window.panel.capacitor_status.text() == tr("computing")
    assert not window.panel.table.isEnabled()  # point charges do not take part in this model
    window._solve_timer.setInterval(0)
    window._solve_timer.start()
    qtbot.waitUntil(lambda: window.scene.conductor is not None, timeout=20000)
    window._retrace()
    assert window.panel.capacitor_status.text() == ""
    assert window.panel.readouts["readout_dV"].text() == "500 V"
    assert window.panel.readouts["readout_C"].text() == "5.43 pF"
    assert all(line.end in ("plate", "bounds") for line in window.view.lines)
    # a new ΔV needs no new solve
    window.panel.voltage_box.setValue(1000)
    assert window.scene.conductor is not None and window.scene.conductor.voltage == 1000


def test_too_small_a_gap_is_clamped_in_the_conductor_model(window):
    window.load_preset("capacitor")
    window.panel.model_combo.setCurrentIndex(window.panel.model_combo.findData("conductor"))
    window.panel.separation_box.setValue(0.2)
    assert window.scene.capacitor.separation >= 0.0076
    assert window.panel.separation_box.value() == pytest.approx(window.scene.capacitor.separation * 100)
    assert "0.8 cm" in window.panel.capacitor_status.text().replace(",", ".")


# --- real Qt events, through VTK's event dispatch -------------------------


def qt_point(window, world):
    """Where to put the Qt mouse for a world point (Qt counts y from the top, VTK from the bottom)."""
    x, y = to_display(window.plotter, np.asarray(world, dtype=float))
    return QtCore.QPoint(round(x), round(window.plotter.height() - 1 - y))


def in_2d(window):
    window.panel.radio_2d.setChecked(True)
    window._retrace()


def test_keys_add_a_charge_and_vtk_keys_are_disabled(window, qtbot):
    in_2d(window)
    qtbot.mouseMove(window.plotter, qt_point(window, [0.0, 0.05, 0.0]))
    qtbot.keyClick(window.plotter, "c")
    assert len(window.scene.charges) == 3
    np.testing.assert_allclose(window.scene.charges[-1].position, [0.0, 0.05, 0.0], atol=5e-4)
    assert window.scene.charges[-1].q > 0
    window.load_preset("capacitor")
    plate = window.view.plate_actors[0]
    qtbot.keyClick(window.plotter, "w")  # VTK would turn every actor into wireframe
    assert plate.prop.style == "Surface"


def test_flip_key_on_a_charge(window, qtbot):
    in_2d(window)
    qtbot.mouseMove(window.plotter, qt_point(window, window.scene.charges[0].position))
    qtbot.keyClick(window.plotter, "t")
    assert window.scene.charges[0].q < 0


def test_a_click_seeds_a_line_but_a_drag_does_not(window, qtbot):
    in_2d(window)
    start = qt_point(window, [0.0, 0.04, 0.0])
    qtbot.mouseClick(window.plotter, QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.waitUntil(lambda: len(window.scene.click_seeds) == 1)
    np.testing.assert_allclose(window.scene.click_seeds[0], [0.0, 0.04, 0.0], atol=5e-4)
    qtbot.mousePress(window.plotter, QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(window.plotter, start + QtCore.QPoint(20, 0))
    qtbot.mouseRelease(window.plotter, QtCore.Qt.MouseButton.LeftButton, pos=start + QtCore.QPoint(20, 0))
    qtbot.wait(50)
    assert len(window.scene.click_seeds) == 1


def wheel(window, qtbot, pos, steps=1):
    qtbot.mouseMove(window.plotter, pos)
    event = QtGui.QWheelEvent(
        QtCore.QPointF(pos),
        QtCore.QPointF(window.plotter.mapToGlobal(pos)),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120 * steps),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(window.plotter, event)


def test_wheel_on_a_charge_changes_it_and_does_not_zoom(window, qtbot):
    in_2d(window)
    scale = window.plotter.camera.parallel_scale
    wheel(window, qtbot, qt_point(window, window.scene.charges[0].position))
    assert window.scene.charges[0].q == 2.25 * NC
    assert window.plotter.camera.parallel_scale == scale
    wheel(window, qtbot, qt_point(window, [0.0, 0.07, 0.0]))
    assert window.scene.charges[0].q == 2.25 * NC
    assert window.plotter.camera.parallel_scale != scale  # on empty space the wheel zooms


def test_dragging_a_charge_with_the_mouse(window, qtbot):
    in_2d(window)
    start = qt_point(window, window.scene.charges[0].position)
    end = qt_point(window, [-0.03, 0.03, 0.0])
    qtbot.mousePress(window.plotter, QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(window.plotter, (start + end) / 2)
    qtbot.mouseMove(window.plotter, end)
    qtbot.mouseRelease(window.plotter, QtCore.Qt.MouseButton.LeftButton, pos=end)
    qtbot.wait(50)
    np.testing.assert_allclose(window.scene.charges[0].position, [-0.03, 0.03, 0.0], atol=1e-3)
    assert window.panel.table.cellWidget(0, 2).value() == pytest.approx(3.0, abs=0.1)
    assert window.scene.click_seeds == []  # pressing on a charge is not a click on the space behind it


def test_reading_the_camera_does_not_move_it(window):
    for planar in (False, True):
        window.panel.radio_2d.setChecked(planar)
        before = window.plotter.renderer.GetActiveCamera().GetPosition()
        window.plotter.camera.up  # PyVista resets a camera it thinks was never placed
        assert window.plotter.renderer.GetActiveCamera().GetPosition() == before


def test_colouring_the_plates_by_density(window):
    window.load_preset("capacitor")
    assert len(window.view.plate_actors) == 2
    window.panel.plate_sigma_box.setChecked(True)
    assert len(window.view.plate_actors) == 4
    assert tr("bar_sigma") in window.plotter.scalar_bars
    window.panel.language_combo.setCurrentIndex(window.panel.language_combo.findData("en"))
    assert "charge density [μC/m²]" in window.plotter.scalar_bars
    window.panel.plate_sigma_box.setChecked(False)
    assert len(window.view.plate_actors) == 2
    assert not any("density" in title or "densità" in title for title in window.plotter.scalar_bars.keys())
