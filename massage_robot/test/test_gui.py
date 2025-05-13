import sys
from PyQt5 import QtWidgets
import pytest
from massage_robot.gui import MassageGUI

def test_gui_smoke(tmp_path, qtbot):
    # Dummy env and executor
    class DummyEnv: pass
    class DummyExecutor:
        grid_map = [[1]]
        region_vertices_map = {1:[0]}
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    gui = MassageGUI(env=DummyEnv(), executor=DummyExecutor())
    qtbot.addWidget(gui)
    gui.show()
    # Cycle the event loop once
    QtWidgets.QApplication.processEvents()
    assert gui.windowTitle() == "Robotic Massage Simulator"
    gui.close()
