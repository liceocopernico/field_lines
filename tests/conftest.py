import os

# before Qt is imported: GUI tests run without opening windows
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
import pyvista as pv  # noqa: E402

from field_lines import strings  # noqa: E402

WINDOW = 400


@pytest.fixture(autouse=True)
def italian():
    """Every test starts in Italian, whatever the previous test switched to."""
    strings.set_language("it")
    yield
    strings.set_language("it")


@pytest.fixture
def plotter():
    pl = pv.Plotter(off_screen=True, window_size=(WINDOW, WINDOW))
    yield pl
    pl.close()


@pytest.fixture
def window(qtbot, monkeypatch):
    from pyvistaqt import QtInteractor

    from field_lines import app

    class SilentInteractor(QtInteractor):
        """A QtInteractor that never draws.

        Qt's offscreen platform gives VTK no OpenGL context, and VTK crashes when
        it asks the missing context for its driver. The GUI tests check the state
        of the window; what gets drawn is checked by test_render.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.iren.interactor.EnableRenderOff()  # also covers paintEvent and the widgets

        def render(self, *args, **kwargs):
            pass

        def _render(self, *args, **kwargs):
            pass

    monkeypatch.setattr(app, "QtInteractor", SilentInteractor)
    w = app.MainWindow(preset="dipole")
    qtbot.addWidget(w)
    w.resize(1000, 700)
    w.show()
    w._retrace()
    yield w
    w.close()
