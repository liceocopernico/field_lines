"""The Qt side panel. It only emits signals and shows values; MainWindow does the work."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

from field_lines.presets import PRESETS
from field_lines.scene import VIEW_HALF, Capacitor
from field_lines.sources import AXES, PointCharge
from field_lines.strings import LANGUAGES, language, tr
from field_lines.units import CM, NC, UC_PER_M2

COORDINATE_RANGE = VIEW_HALF / CM  # cm
SLICE_STEPS = 100  # positions of the slice slider across the view


@dataclass
class ViewOptions:
    lines: bool = True
    lines_per_nc: float = 10.0
    equipotentials: bool = False
    levels: int = 9
    arrows: bool = False
    slice: bool = False
    slice_quantity: str = "V"  # "V" or "E"
    slice_axis: str = "z"
    slice_position: float = 0.0  # m


def spin_box(lo: float, hi: float, step: float, decimals: int, value: float) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setKeyboardTracking(False)  # typing "2.5" should not stop at "2"
    return box


class Panel(QtWidgets.QWidget):
    planar_changed = QtCore.Signal(bool)
    preset_chosen = QtCore.Signal(str)
    charge_edited = QtCore.Signal(int, float, float, float, float)  # row, q [nC], x, y, z [cm]
    add_charge_requested = QtCore.Signal(int)  # sign
    remove_requested = QtCore.Signal(int)  # row
    capacitor_changed = QtCore.Signal()
    views_changed = QtCore.Signal()
    replay_requested = QtCore.Signal()
    speed_changed = QtCore.Signal(float)  # m/s
    language_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._texts: list[tuple[Callable[[str], None], str]] = []
        self._combo_keys: list[tuple[QtWidgets.QComboBox, list[str]]] = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._space_group())
        layout.addWidget(self._preset_group())
        layout.addWidget(self._charges_group())
        layout.addWidget(self._capacitor_group())
        layout.addWidget(self._views_group())
        layout.addWidget(self._animation_group())
        layout.addWidget(self._language_group())
        layout.addStretch(1)
        self.retranslate()

    # --- building -------------------------------------------------------

    def _text(self, setter: Callable[[str], None], key: str) -> None:
        self._texts.append((setter, key))

    def _group(self, key: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
        group = QtWidgets.QGroupBox()
        self._text(group.setTitle, key)
        return group, QtWidgets.QVBoxLayout(group)

    def _combo(self, keys: Sequence[str], data: Sequence[str]) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        for value in data:
            combo.addItem("", value)
        self._combo_keys.append((combo, list(keys)))
        return combo

    def _label(self, key: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel()
        self._text(label.setText, key)
        return label

    def _space_group(self):
        group, layout = self._group("group_space")
        row = QtWidgets.QHBoxLayout()
        self.radio_3d = QtWidgets.QRadioButton()
        self.radio_2d = QtWidgets.QRadioButton()
        self._text(self.radio_3d.setText, "mode_3d")
        self._text(self.radio_2d.setText, "mode_2d")
        self.radio_3d.setChecked(True)
        self.radio_2d.toggled.connect(self.planar_changed.emit)
        row.addWidget(self.radio_3d)
        row.addWidget(self.radio_2d)
        layout.addLayout(row)
        return group

    def _preset_group(self):
        group, layout = self._group("group_preset")
        keys = list(PRESETS)
        self.preset_combo = self._combo([f"preset_{key}" for key in keys], keys)
        self.preset_combo.activated.connect(lambda i: self.preset_chosen.emit(self.preset_combo.itemData(i)))
        layout.addWidget(self.preset_combo)
        return group

    def _charges_group(self):
        group, layout = self._group("group_charges")
        self.charges_group = group
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["q [nC]", "x [cm]", "y [cm]", "z [cm]", ""])
        header = self.table.horizontalHeader()
        for column in range(4):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setMinimumHeight(140)
        layout.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.add_positive = QtWidgets.QPushButton()
        self.add_negative = QtWidgets.QPushButton()
        self._text(self.add_positive.setText, "add_positive")
        self._text(self.add_negative.setText, "add_negative")
        self.add_positive.clicked.connect(lambda: self.add_charge_requested.emit(1))
        self.add_negative.clicked.connect(lambda: self.add_charge_requested.emit(-1))
        row.addWidget(self.add_positive)
        row.addWidget(self.add_negative)
        layout.addLayout(row)
        self.charges_note = self._label("charges_disabled")
        self.charges_note.setWordWrap(True)
        self.charges_note.hide()
        layout.addWidget(self.charges_note)
        return group

    def _capacitor_group(self):
        group, layout = self._group("group_capacitor")
        self.capacitor_box = QtWidgets.QCheckBox()
        self._text(self.capacitor_box.setText, "capacitor_show")
        self.capacitor_box.toggled.connect(self._capacitor_toggled)
        layout.addWidget(self.capacitor_box)
        self.capacitor_form = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.capacitor_form)
        form.setContentsMargins(0, 0, 0, 0)
        defaults = Capacitor()
        self.width_box = spin_box(1, 20, 0.5, 1, defaults.width / CM)
        self.height_box = spin_box(1, 20, 0.5, 1, defaults.height / CM)
        self.separation_box = spin_box(0.2, 15, 0.1, 1, defaults.separation / CM)
        self.model_combo = self._combo(["model_sigma", "model_conductor"], ["sigma", "conductor"])
        self.sigma_box = spin_box(-10, 10, 0.05, 2, defaults.sigma / UC_PER_M2)
        self.voltage_box = spin_box(-10000, 10000, 50, 0, defaults.voltage)
        for box in (self.width_box, self.height_box, self.separation_box):
            box.setSuffix(" cm")
        self.sigma_box.setSuffix(" μC/m²")
        self.voltage_box.setSuffix(" V")
        self.sigma_label, self.voltage_label = self._label("sigma"), self._label("voltage")
        for key, widget in (
            ("plate_width", self.width_box),
            ("plate_height", self.height_box),
            ("separation", self.separation_box),
            ("model", self.model_combo),
        ):
            form.addRow(self._label(key), widget)
        form.addRow(self.sigma_label, self.sigma_box)
        form.addRow(self.voltage_label, self.voltage_box)
        for box in (self.width_box, self.height_box, self.separation_box, self.sigma_box, self.voltage_box):
            box.valueChanged.connect(self.capacitor_changed.emit)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.readouts: dict[str, QtWidgets.QLabel] = {}
        for key in ("readout_E_center", "readout_E_ideal", "readout_dV", "readout_Q", "readout_C", "readout_C_ideal"):
            value = QtWidgets.QLabel("—")
            value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self.readouts[key] = value
            form.addRow(self._label(key), value)
        self.capacitor_status = QtWidgets.QLabel()
        self.capacitor_status.setWordWrap(True)
        form.addRow(self.capacitor_status)
        layout.addWidget(self.capacitor_form)
        self.capacitor_form.hide()
        self._model_changed()
        return group

    def _views_group(self):
        group, layout = self._group("group_views")
        grid = QtWidgets.QGridLayout()
        defaults = ViewOptions()
        self.lines_box = QtWidgets.QCheckBox()
        self.lines_box.setChecked(defaults.lines)
        self.lines_per_nc = spin_box(1, 100, 1, 0, defaults.lines_per_nc)
        self.equipotentials_box = QtWidgets.QCheckBox()
        self.levels_box = QtWidgets.QSpinBox()
        self.levels_box.setRange(1, 30)
        self.levels_box.setValue(defaults.levels)
        self.arrows_box = QtWidgets.QCheckBox()
        self.slice_box = QtWidgets.QCheckBox()
        self.slice_quantity = self._combo(["slice_V", "slice_E"], ["V", "E"])
        self.slice_axis = QtWidgets.QComboBox()
        for axis in AXES:
            self.slice_axis.addItem(axis, axis)
        self.slice_axis.setCurrentIndex(2)
        self.slice_position = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice_position.setRange(-SLICE_STEPS // 2, SLICE_STEPS // 2)
        self.slice_position.setValue(0)
        for key, box in (
            ("view_lines", self.lines_box),
            ("view_equipotentials", self.equipotentials_box),
            ("view_arrows", self.arrows_box),
            ("view_slice", self.slice_box),
        ):
            self._text(box.setText, key)
        grid.addWidget(self.lines_box, 0, 0)
        grid.addWidget(self.lines_per_nc, 0, 1)
        grid.addWidget(self._label("lines_per_nc"), 0, 2)
        grid.addWidget(self.equipotentials_box, 1, 0)
        grid.addWidget(self.levels_box, 1, 1)
        grid.addWidget(self._label("levels"), 1, 2)
        grid.addWidget(self.arrows_box, 2, 0)
        grid.addWidget(self.slice_box, 3, 0)
        grid.addWidget(self.slice_quantity, 3, 1, 1, 2)
        layout.addLayout(grid)
        self.slice_form = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.slice_form)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(self._label("slice_axis"), self.slice_axis)
        form.addRow(self._label("slice_position"), self.slice_position)
        layout.addWidget(self.slice_form)
        for box in (self.lines_box, self.equipotentials_box, self.arrows_box, self.slice_box):
            box.toggled.connect(self.views_changed.emit)
        self.lines_per_nc.valueChanged.connect(self.views_changed.emit)
        self.levels_box.valueChanged.connect(self.views_changed.emit)
        self.slice_quantity.currentIndexChanged.connect(self.views_changed.emit)
        self.slice_axis.currentIndexChanged.connect(self.views_changed.emit)
        self.slice_position.valueChanged.connect(self.views_changed.emit)
        return group

    def _animation_group(self):
        group, layout = self._group("group_animation")
        self.replay_button = QtWidgets.QPushButton()
        self._text(self.replay_button.setText, "replay")
        self.replay_button.clicked.connect(self.replay_requested.emit)
        layout.addWidget(self.replay_button)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._label("speed"))
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 40)  # cm/s
        self.speed_slider.setValue(10)
        self.speed_value = QtWidgets.QLabel()
        self.speed_slider.valueChanged.connect(self._speed_changed)
        row.addWidget(self.speed_slider, 1)
        row.addWidget(self.speed_value)
        layout.addLayout(row)
        self._speed_changed(self.speed_slider.value())
        return group

    def _language_group(self):
        group, layout = self._group("group_language")
        self.language_combo = QtWidgets.QComboBox()
        for key, name in LANGUAGES.items():
            self.language_combo.addItem(name, key)
        self.language_combo.setCurrentIndex(list(LANGUAGES).index(language()))
        self.language_combo.currentIndexChanged.connect(
            lambda i: self.language_changed.emit(self.language_combo.itemData(i))
        )
        layout.addWidget(self.language_combo)
        return group

    # --- reactions ------------------------------------------------------

    def _capacitor_toggled(self, checked: bool) -> None:
        self.capacitor_form.setVisible(checked)
        self.capacitor_changed.emit()

    def _model_changed(self) -> None:
        conductor = self.model_combo.currentData() == "conductor"
        self.sigma_label.setVisible(not conductor)
        self.sigma_box.setVisible(not conductor)
        self.voltage_label.setVisible(conductor)
        self.voltage_box.setVisible(conductor)
        self.capacitor_changed.emit()

    def _speed_changed(self, value: int) -> None:
        self.speed_value.setText(f"{value} cm/s")
        self.speed_changed.emit(value * CM)

    # --- showing state --------------------------------------------------

    def retranslate(self) -> None:
        for setter, key in self._texts:
            setter(tr(key))
        for combo, keys in self._combo_keys:
            for index, key in enumerate(keys):
                combo.setItemText(index, tr(key))
        for row in range(self.table.rowCount()):
            self.table.cellWidget(row, 4).setToolTip(tr("remove"))

    def set_planar(self, planar: bool) -> None:
        with QtCore.QSignalBlocker(self.radio_2d), QtCore.QSignalBlocker(self.radio_3d):
            (self.radio_2d if planar else self.radio_3d).setChecked(True)
        self.table.setColumnHidden(3, planar)

    def set_preset(self, key: str) -> None:
        self.preset_combo.setCurrentIndex(self.preset_combo.findData(key))

    def set_charges(self, charges: Sequence[PointCharge], enabled: bool = True) -> None:
        """Rebuild the table, one row of spin boxes per charge."""
        self.table.setRowCount(len(charges))
        for row, charge in enumerate(charges):
            values = [charge.q / NC, *(charge.position / CM)]
            ranges = [(-20, 20, 0.25, 2)] + [(-COORDINATE_RANGE, COORDINATE_RANGE, 0.5, 1)] * 3
            for column, (value, (lo, hi, step, decimals)) in enumerate(zip(values, ranges)):
                box = spin_box(lo, hi, step, decimals, value)
                box.setFrame(False)
                box.valueChanged.connect(lambda _value, r=row: self._row_edited(r))
                self.table.setCellWidget(row, column, box)
            remove = QtWidgets.QToolButton()
            remove.setText("✕")
            remove.setToolTip(tr("remove"))
            remove.clicked.connect(lambda _checked=False, r=row: self.remove_requested.emit(r))
            self.table.setCellWidget(row, 4, remove)
        self.table.setEnabled(enabled)
        self.add_positive.setEnabled(enabled)
        self.add_negative.setEnabled(enabled)
        self.charges_note.setVisible(not enabled)

    def update_charge(self, row: int, charge: PointCharge) -> None:
        """Show a charge's new values without emitting charge_edited (e.g. while it is dragged)."""
        values = [charge.q / NC, *(charge.position / CM)]
        for column, value in enumerate(values):
            box = self.table.cellWidget(row, column)
            if box is not None:
                with QtCore.QSignalBlocker(box):
                    box.setValue(value)

    def _row_edited(self, row: int) -> None:
        values = [self.table.cellWidget(row, column).value() for column in range(4)]
        self.charge_edited.emit(row, *values)

    def set_capacitor(self, capacitor: Capacitor | None) -> None:
        boxes = (self.capacitor_box, self.width_box, self.height_box, self.separation_box, self.sigma_box)
        blockers = [QtCore.QSignalBlocker(w) for w in (*boxes, self.voltage_box, self.model_combo)]
        self.capacitor_box.setChecked(capacitor is not None)
        self.capacitor_form.setVisible(capacitor is not None)
        if capacitor is not None:
            self.width_box.setValue(capacitor.width / CM)
            self.height_box.setValue(capacitor.height / CM)
            self.separation_box.setValue(capacitor.separation / CM)
            self.sigma_box.setValue(capacitor.sigma / UC_PER_M2)
            self.voltage_box.setValue(capacitor.voltage)
            self.model_combo.setCurrentIndex(self.model_combo.findData(capacitor.model))
        for blocker in blockers:
            blocker.unblock()
        self._model_changed_quietly()

    def _model_changed_quietly(self) -> None:
        with QtCore.QSignalBlocker(self):
            self._model_changed()

    def capacitor(self, normal: str) -> Capacitor | None:
        """The capacitor as the panel shows it, or None if it is switched off."""
        if not self.capacitor_box.isChecked():
            return None
        return Capacitor(
            width=self.width_box.value() * CM,
            height=self.height_box.value() * CM,
            separation=self.separation_box.value() * CM,
            sigma=self.sigma_box.value() * UC_PER_M2,
            voltage=self.voltage_box.value(),
            model=self.model_combo.currentData(),
            normal=normal,
        )

    def set_readouts(self, values: dict[str, str], status: str = "") -> None:
        for key, label in self.readouts.items():
            label.setText(values.get(key, "—"))
        self.capacitor_status.setText(status)
        self.capacitor_status.setVisible(bool(status))

    def view_options(self) -> ViewOptions:
        return ViewOptions(
            lines=self.lines_box.isChecked(),
            lines_per_nc=self.lines_per_nc.value(),
            equipotentials=self.equipotentials_box.isChecked(),
            levels=self.levels_box.value(),
            arrows=self.arrows_box.isChecked(),
            slice=self.slice_box.isChecked(),
            slice_quantity=self.slice_quantity.currentData(),
            slice_axis=self.slice_axis.currentData(),
            slice_position=self.slice_position.value() / (SLICE_STEPS / 2) * VIEW_HALF,
        )
