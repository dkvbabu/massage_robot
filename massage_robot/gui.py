## massage_robot/gui.py
from PyQt5 import QtWidgets, QtCore
import sys

class MassageGUI(QtWidgets.QMainWindow):
    def __init__(self, env, executor):
        super().__init__()
        self.env = env
        self.executor = executor
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Robotic Massage Simulator")
        self.setGeometry(100,100,800,600)
        # Start button
        btn = QtWidgets.QPushButton("Start Massage", self)
        btn.clicked.connect(self.run_massage)
        self.status = QtWidgets.QLabel("Ready", self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(btn)
        layout.addWidget(self.status)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_massage(self):
        try:
            self.status.setText("Running...")
            self.executor.execute(self.executor.grid_map, self.executor.region_vertices_map)
            self.status.setText("Completed")
        except Exception as e:
            self.status.setText(f"Error: {e}")

if __name__ == '__main__':
    from massage_robot.env import MassageEnv
    from massage_robot.path_planner import PathPlanner
    from massage_robot.force_control import ForceController
    from massage_robot.safety import SafetyModule
    from massage_robot.execution import GridExecutor

    app = QtWidgets.QApplication(sys.argv)
    env = MassageEnv(gui=True)
    # Example simple grid and mapping
    grid_map = [[1,1,2],[2,2,3]]  # region IDs
    region_vertices_map = {1: [0,1,2], 2: [3,4,5], 3: [6,7,8]}
    planner = PathPlanner(*env.reconstruct_surface())
    controller = ForceController()
    safety = SafetyModule()
    executor = GridExecutor(planner, controller, safety, env)
    executor.grid_map = grid_map
    executor.region_vertices_map = region_vertices_map
    gui = MassageGUI(env, executor)
    gui.show()
    sys.exit(app.exec_())
