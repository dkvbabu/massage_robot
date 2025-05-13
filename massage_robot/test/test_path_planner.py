import numpy as np
import pytest
from massage_robot.path_planner import PathPlanner

def test_plan_through_waypoints():
    waypoints = [[0, 0, 0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    planner = PathPlanner(vertices=np.zeros((0,3)), simplices=np.zeros((0,3), dtype=int))
    samples = 50
    path = planner.plan_through_waypoints(waypoints, samples=samples)
    assert isinstance(path, np.ndarray)
    assert path.shape == (samples, 3)

def test_plan_grid():
    # create a fake mesh of 9 vertices
    vertices = np.arange(27).reshape((9,3)).astype(float)
    simplices = np.zeros((0,3), dtype=int)
    # two regions: 1 covers verts [0,1,2], 2 covers [3,4,5]
    region_map = [[1, 2], [2, 1]]
    region_vertices_map = {
        1: [0,1,2],
        2: [3,4,5],
    }
    samples_per_region = 10
    planner = PathPlanner(vertices, simplices)
    trajectory = planner.plan_grid(region_map, region_vertices_map, samples_per_region)
    # Expect (2*2*10) = 40 points
    assert isinstance(trajectory, np.ndarray)
    assert trajectory.shape == (4 * samples_per_region, 3)
