import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from massage_robot.execution import execute_massage, stop_massage, get_last_stats, update_simulation_parameters, update_simulation_camera
import matplotlib.pyplot as plt
import numpy as np

class MassageRobotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Massage Robot Simulation Control')
        self.setGeometry(100, 100, 500, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Target Body Area
        area_group = QGroupBox('Target Body Area')
        area_layout = QHBoxLayout()
        self.area_combo = QComboBox()
        self.area_combo.addItems(['Lower Back', 'Upper Back', 'Shoulders', 'Neck'])
        area_layout.addWidget(self.area_combo)
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)

        # Massage Technique
        technique_group = QGroupBox('Massage Technique')
        technique_layout = QHBoxLayout()
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(['Kneading', 'Pressure', 'Tapping', 'Rolling'])
        technique_layout.addWidget(self.technique_combo)
        technique_group.setLayout(technique_layout)
        layout.addWidget(technique_group)

        # Force/Pressure
        force_group = QGroupBox('Force / Pressure (N)')
        force_layout = QHBoxLayout()
        self.force_slider = QSlider(Qt.Horizontal)
        self.force_slider.setMinimum(0)
        self.force_slider.setMaximum(100)
        self.force_slider.setValue(20)
        self.force_label = QLabel('20')
        self.force_slider.valueChanged.connect(self.on_param_change)
        force_layout.addWidget(self.force_slider)
        force_layout.addWidget(self.force_label)
        force_group.setLayout(force_layout)
        layout.addWidget(force_group)

        # Angle/Orientation (Camera View)
        angle_group = QGroupBox('Camera Angle / Orientation')
        angle_layout = QHBoxLayout()
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-180)
        self.yaw_slider.setMaximum(180)
        self.yaw_slider.setValue(0)
        self.yaw_label = QLabel('Yaw: 0°')
        self.yaw_slider.valueChanged.connect(self.on_camera_change)
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setMinimum(-90)
        self.pitch_slider.setMaximum(90)
        self.pitch_slider.setValue(0)
        self.pitch_label = QLabel('Pitch: 0°')
        self.pitch_slider.valueChanged.connect(self.on_camera_change)
        angle_layout.addWidget(self.yaw_label)
        angle_layout.addWidget(self.yaw_slider)
        angle_layout.addWidget(self.pitch_label)
        angle_layout.addWidget(self.pitch_slider)
        angle_group.setLayout(angle_layout)
        layout.addWidget(angle_group)

        # Session Duration
        duration_group = QGroupBox('Session Duration (seconds)')
        duration_layout = QHBoxLayout()
        self.duration_spin = QSpinBox()
        self.duration_spin.setMinimum(10)
        self.duration_spin.setMaximum(3600)
        self.duration_spin.setValue(60)
        duration_layout.addWidget(self.duration_spin)
        duration_group.setLayout(duration_layout)
        layout.addWidget(duration_group)

        # Repetitions
        reps_group = QGroupBox('Repetitions')
        reps_layout = QHBoxLayout()
        self.reps_spin = QSpinBox()
        self.reps_spin.setMinimum(1)
        self.reps_spin.setMaximum(100)
        self.reps_spin.setValue(3)
        reps_layout.addWidget(self.reps_spin)
        reps_group.setLayout(reps_layout)
        layout.addWidget(reps_group)

        # Path Speed
        speed_group = QGroupBox('Massage Path Speed (cm/s)')
        speed_layout = QHBoxLayout()
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setMinimum(0.1)
        self.speed_spin.setMaximum(10.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.valueChanged.connect(self.on_param_change)
        speed_layout.addWidget(self.speed_spin)
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        # Path Pattern
        pattern_group = QGroupBox('Path Pattern')
        pattern_layout = QHBoxLayout()
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(['sine', 'circular', 'linear'])
        self.pattern_combo.currentTextChanged.connect(self.on_param_change)
        pattern_layout.addWidget(self.pattern_combo)
        pattern_group.setLayout(pattern_layout)
        layout.addWidget(pattern_group)

        # Amplitude
        amp_group = QGroupBox('Amplitude (m)')
        amp_layout = QHBoxLayout()
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setMinimum(0.0)
        self.amp_spin.setMaximum(0.1)
        self.amp_spin.setSingleStep(0.005)
        self.amp_spin.setValue(0.02)
        self.amp_spin.valueChanged.connect(self.on_param_change)
        amp_layout.addWidget(self.amp_spin)
        amp_group.setLayout(amp_layout)
        layout.addWidget(amp_group)

        # Frequency
        freq_group = QGroupBox('Frequency (Hz)')
        freq_layout = QHBoxLayout()
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setMinimum(0.1)
        self.freq_spin.setMaximum(10.0)
        self.freq_spin.setSingleStep(0.1)
        self.freq_spin.setValue(2.0)
        self.freq_spin.valueChanged.connect(self.on_param_change)
        freq_layout.addWidget(self.freq_spin)
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)

        # Approach/Main/Retraction Samples
        samples_group = QGroupBox('Path Samples')
        samples_layout = QHBoxLayout()
        self.approach_spin = QSpinBox()
        self.approach_spin.setMinimum(1)
        self.approach_spin.setMaximum(500)
        self.approach_spin.setValue(50)
        self.main_spin = QSpinBox()
        self.main_spin.setMinimum(1)
        self.main_spin.setMaximum(1000)
        self.main_spin.setValue(200)
        self.retract_spin = QSpinBox()
        self.retract_spin.setMinimum(1)
        self.retract_spin.setMaximum(500)
        self.retract_spin.setValue(50)
        self.approach_spin.valueChanged.connect(self.on_param_change)
        self.main_spin.valueChanged.connect(self.on_param_change)
        self.retract_spin.valueChanged.connect(self.on_param_change)
        samples_layout.addWidget(QLabel('Approach'))
        samples_layout.addWidget(self.approach_spin)
        samples_layout.addWidget(QLabel('Main'))
        samples_layout.addWidget(self.main_spin)
        samples_layout.addWidget(QLabel('Retract'))
        samples_layout.addWidget(self.retract_spin)
        samples_group.setLayout(samples_layout)
        layout.addWidget(samples_group)

        # Show Graph Button
        self.graph_btn = QPushButton('Show Graph')
        self.graph_btn.clicked.connect(self.show_graph)
        self.graph_btn.setEnabled(False)
        layout.addWidget(self.graph_btn)

        # Start/Stop Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton('Start Session')
        self.stop_btn = QPushButton('Stop Session')
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.start_btn.clicked.connect(self.start_session)
        self.stop_btn.clicked.connect(self.stop_session)

    def start_session(self):
        if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
            QMessageBox.warning(self, 'Simulation Running', 'Simulation is already running.')
            return
        self.start_btn.setEnabled(False)
        self.graph_btn.setEnabled(False)
        self.sim_running = True
        region = self.area_combo.currentText().replace(' ', '_').lower()
        technique = self.technique_combo.currentText().lower()
        force = self.force_slider.value()
        speed = self.speed_spin.value()
        duration = self.duration_spin.value()
        repetitions = self.reps_spin.value()
        yaw = self.yaw_slider.value()
        pitch = self.pitch_slider.value()
        pattern = self.pattern_combo.currentText()
        amp = self.amp_spin.value()
        freq = self.freq_spin.value()
        approach_samples = self.approach_spin.value()
        main_samples = self.main_spin.value()
        retract_samples = self.retract_spin.value()
        def run_sim():
            execute_massage(region, technique, force, speed, duration, repetitions, yaw, pitch, pattern, amp, freq, approach_samples, main_samples, retract_samples)
            self.sim_running = False
            self.start_btn.setEnabled(True)
            self.poll_for_stats()
        self.sim_thread = threading.Thread(target=run_sim, daemon=True)
        self.sim_thread.start()

    def stop_session(self):
        if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
            stop_massage()
            self.sim_thread.join(timeout=1)
        self.sim_running = False
        self.start_btn.setEnabled(True)
        self.graph_btn.setEnabled(True)
        print("Session stopped by user.")

    def show_graph(self):
        if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
            QMessageBox.information(self, 'Graph', 'Please stop the simulation before showing the graph.')
            return
        stats = get_last_stats()
        print(f"[DEBUG] show_graph: stats={stats}")
        if not stats or not stats['Forces']:
            QMessageBox.information(self, 'Graph', 'No session data available. Run a session first.')
            return
        Forces = np.array(stats['Forces'])
        armparts = np.array(stats['armparts'])
        bodyparts = np.array(stats['bodyparts'])
        old_path = np.array(stats['old_path'])
        new_path = np.array(stats['new_path'])
        actual_path = np.array(stats['actual_path'])
        print(f"Show Graph: Forces={len(Forces)}, armparts={len(armparts)}, bodyparts={len(bodyparts)}")
        def plot_graph():
            plt.figure(figsize=(10,8))
            # Massage Pressure
            plt.subplot(411)
            if Forces.size > 0 and not np.isnan(Forces).all():
                mean_force = np.nanmean(Forces)
                plt.title(f'Massage Pressure: Mean {mean_force:.2f}')
                plt.ylabel('Newton')
                plt.plot(Forces)
            else:
                plt.title('Massage Pressure: No data collected')
                plt.ylabel('Newton')
            # Arm Part Contact
            plt.subplot(412)
            if armparts.size > 0 and not np.isnan(armparts).all():
                plt.title('Arm Part Contact')
                plt.plot(armparts)
            else:
                plt.title('Arm Part Contact: No data collected')
            # Body Part Contact
            plt.subplot(413)
            if bodyparts.size > 0 and not np.isnan(bodyparts).all():
                plt.title('Body Part Contact')
                plt.plot(bodyparts)
            else:
                plt.title('Body Part Contact: No data collected')
            # Massage Paths
            plt.subplot(414)
            plt.title('Massage Paths')
            has_path = False
            if old_path.size > 0 and not np.isnan(old_path).all():
                plt.plot(old_path, label='sinusoidal commands')
                has_path = True
            if new_path.size > 0 and not np.isnan(new_path).all():
                plt.plot(new_path, label='surface level')
                has_path = True
            if actual_path.size > 0 and not np.isnan(actual_path).all():
                plt.plot(actual_path, label='modified path')
                has_path = True
            plt.ylabel('meters')
            if has_path:
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'No path data collected', ha='center', va='center', transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.show()
        QTimer.singleShot(0, plot_graph)

    def on_param_change(self, *args):
        params = {
            'force': self.force_slider.value(),
            'speed': self.speed_spin.value(),
            'pattern': self.pattern_combo.currentText(),
            'amp': self.amp_spin.value(),
            'freq': self.freq_spin.value(),
            'approach_samples': self.approach_spin.value(),
            'main_samples': self.main_spin.value(),
            'retract_samples': self.retract_spin.value(),
        }
        update_simulation_parameters(**params)

    def on_camera_change(self, *args):
        yaw = self.yaw_slider.value()
        pitch = self.pitch_slider.value()
        update_simulation_camera(yaw=yaw, pitch=pitch)

    def poll_for_stats(self, attempts=0):
        stats = get_last_stats()
        if stats and stats['Forces']:
            self.graph_btn.setEnabled(True)
        elif attempts < 10:
            QTimer.singleShot(200, lambda: self.poll_for_stats(attempts+1))
        else:
            QMessageBox.warning(self, 'No Data', 'Simulation finished but no data was collected. Try running a longer session.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MassageRobotGUI()
    gui.show()
    sys.exit(app.exec_())
