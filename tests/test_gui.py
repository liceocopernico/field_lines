import numpy as np
from PySide6 import QtCore

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
