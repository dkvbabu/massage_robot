import numpy as np
from scipy.interpolate import CubicSpline

class PathPlanner:
    """
    Plans paths over a triangulated surface mesh.
    """
    def __init__(self, vertices, simplices):
        self.vertices = vertices
        self.simplices = simplices

    def plan_waypoints(self, region_indices, total_samples=200):
        pts = self.vertices[region_indices]
        t_wp = np.linspace(0, 1, len(pts))
        cs = CubicSpline(t_wp, pts, axis=0)
        t = np.linspace(0, 1, total_samples)
        return cs(t)

    def add_pattern(self, path, pattern='sine', amp=0.02, freq=2):
        if pattern == 'sine':
            t = np.linspace(0,1,len(path))
            path[:,1] += amp * np.sin(2*np.pi*freq*t)
        return path
