import numpy as np
from scipy.interpolate import CubicSpline

class PathGenerationProgram:
    """
    Generates a time-parameterized list of end-effector poses over the human model.
    - region_waypoints: list of [x,y,z] keypoints
    - total_samples: length of trajectory
    - pattern: 'linear' | 'sine' | 'circular'
    """
    def __init__(self, region_waypoints, orientation=[0,0,0,1]):
        self.waypoints = np.array(region_waypoints)
        self.orientation = orientation

    def _interpolate(self, n):
        t_wp = np.linspace(0,1,len(self.waypoints))
        cs = CubicSpline(t_wp, self.waypoints, axis=0)
        t = np.linspace(0,1,n)
        return cs(t), t

    def _apply_pattern(self, pos, t, pattern):
        if pattern=='sine':
            amp, freq = 0.02, 2
            pos[:,1] += amp*np.sin(2*np.pi*freq*t)
        elif pattern=='circular':
            r = 0.03
            theta = 2*np.pi*t
            pos[:,0] += r*np.cos(theta)
            pos[:,1] += r*np.sin(theta)
        return pos

    def generate(self, total_samples=200, pattern='sine'):
        positions, t = self._interpolate(total_samples)
        positions = self._apply_pattern(positions, t, pattern)
        trajectory = [
            {"position": p.tolist(), "orientation": self.orientation}
            for p in positions
        ]
        return trajectory
