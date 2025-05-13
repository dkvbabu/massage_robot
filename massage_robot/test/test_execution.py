import pytest
import numpy as np
from massage_robot.execution import GridExecutor

class DummyEnv:
    def __init__(self):
        self.default_orientation = [0,0,0,1]
    def calculate_ik(self, pos, orn):
        # pretend 7-DOF
        return [0.0]*7
    def _read_force_sensors(self):
        return {0: 10.0}
    def step(self, cmds, render=False):
        # each joint state is (pos, vel)
        return {
            'contact_forces': {0: 10.0},
            'joint_states': [(0.0, 0.0)]*7
        }

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
