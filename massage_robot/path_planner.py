# Fully Implemented Modules: Path Planning, Force Control, Safety, Execution, and GUI

## massage_robot/path_planner.py
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline

class PathPlanner:
    """
    Plans a massage path over a reconstructed surface mesh using waypoints or grid regions.
    """
    def __init__(self, vertices, simplices):
        self.vertices = vertices  # Nx3 array
        self.simplices = simplices  # Mx3 indices

    def region_centroid(self, region_indices):
        pts = self.vertices[region_indices]
        return np.mean(pts, axis=0)

    def plan_through_waypoints(self, waypoints, samples=200):
        # Cubic spline through 3D waypoints
        waypoints = np.array(waypoints)
        t = np.linspace(0,1,len(waypoints))
        cs = CubicSpline(t, waypoints, axis=0)
        t_samples = np.linspace(0,1,samples)
        return cs(t_samples)

    def plan_grid(self, grid_map, region_vertices_map, samples_per_region=100):
        # grid_map: 2D array of region IDs
        # region_vertices_map: dict regionID -> vertex indices
        trajectory = []
        for row in grid_map:
            for region in row:
                cent = self.region_centroid(region_vertices_map[region])
                pts = self.plan_through_waypoints([cent], samples_per_region)
                trajectory.append(pts)
        return np.vstack(trajectory)

