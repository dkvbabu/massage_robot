import pytest
import numpy as np
from massage_robot.execution import GridExecutor, execute_massage

class DummyEnv:
    def __init__(self):
        self.default_orientation = [0,0,0,1]
        self.force = 10
        self.speed = 1.0
        self.TimeStep = 0.01
        self.SimID = 0
    def calculate_ik(self, pos, orn):
        # pretend 7-DOF to match real env expectations
        return [0.0]*7
    def _read_force_sensors(self):
        return {0: 10.0}
    def step(self, cmds, render=False):
        # each joint state is (pos, vel)
        return {
            'contact_forces': {0: 10.0},
            'joint_states': [(0.0, 0.0)]*7
        }
    def get_action(self):
        return [0.0, 0.0, 0.0]
    def set_camera(self, yaw, pitch):
        pass
    def run_until_exit(self):
        pass
    def close(self):
        pass
    def get_stats(self):
        return {'Forces': [10.0], 'armparts': [0], 'bodyparts': [0], 'old_path': [0.0], 'new_path': [0.0], 'actual_path': [0.0]}

class DummyPlanner:
    def plan_grid(self, grid_map, region_map, samples_per_region):
        # return a flat array of points: one per cell * samples
        rows, cols = np.shape(grid_map)
        total = rows*cols*samples_per_region
        return np.zeros((total,3))

class DummyController:
    def compute_torque(self, current_force, dt):
        return 0.0

class DummySafety:
    def verify(self, forces, velocities):
        return True, 'Safe'

def test_execute_grid():
    grid_map = [[1,2],[2,1]]
    region_map = {1:[0], 2:[1]}
    env = DummyEnv()
    planner = DummyPlanner()
    controller = DummyController()
    safety = DummySafety()
    executor = GridExecutor(planner, controller, safety, env)
    results = executor.execute(grid_map, region_map, samples_per_region=5, dt=0.01)
    # expect len(path) = 2*2*5 = 20 steps
    assert isinstance(results, list)
    assert len(results) == 20
    # each result is an observation dict
    assert all(isinstance(r, dict) for r in results)

def test_execute_massage_runs():
    # Run a minimal session to check for errors
    execute_massage(
        region='lower_back',
        technique='kneading',
        force=10,
        speed=1.0,
        duration=1,
        repetitions=1,
        yaw=0,
        pitch=0,
        pattern='sine',
        amp=0.01,
        freq=1.0,
        approach_samples=2,
        main_samples=2,
        retract_samples=2
    )
