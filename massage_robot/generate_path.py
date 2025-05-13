# File: massage_robot/generate_path.py
import numpy as np
from scipy.interpolate import CubicSpline

class PathGenerationProgram:
    """
    Generates a full end-effector trajectory with approach, main stroke, and retract phases.
    Can generate for lower-back and upper-back regions with configurable patterns.

    Usage:
        pg = PathGenerationProgram(waypoints, orientation)
        trajectory = pg.generate(
            main_samples=200,
            approach_samples=50,
            retract_samples=50,
            pattern='sine', amp=0.02, freq=2
        )

    Methods:
        generate(): returns a list of dicts with 'position' and 'orientation'.
        test_generate_settings(): runs built-in tests and reports errors.
    """
    def __init__(self, region_waypoints, orientation=[0,0,0,1], home_offset=[-0.2,0,0]):
        self.waypoints = np.array(region_waypoints)
        self.orientation = orientation
        # Compute default home pose by offsetting the first waypoint
        first = self.waypoints[0]
        home_pos = first + np.array(home_offset)
        self.home_pose = {'position': home_pos.tolist(), 'orientation': orientation}

    def _interpolate_spline(self, points, samples):
        """Cubic spline interpolation through a set of waypoints."""
        t_wp = np.linspace(0,1,len(points))
        cs = CubicSpline(t_wp, points, axis=0)
        t = np.linspace(0,1,samples)
        return cs(t), t

    @staticmethod
    def _linspace_positions(p0, p1, samples):
        """Linear interpolation from p0 to p1 over given samples."""
        p0, p1 = np.array(p0), np.array(p1)
        return np.linspace(p0, p1, samples)

    def _apply_pattern(self, positions, pattern, amp, freq):
        """Modulate the path with a stroke pattern."""
        if pattern == 'sine':
            t = np.linspace(0,1,len(positions))
            positions[:,1] += amp * np.sin(2*np.pi*freq*t)
        elif pattern == 'circular':
            t = np.linspace(0,1,len(positions))
            positions[:,0] += amp * np.cos(2*np.pi*freq*t)
            positions[:,1] += amp * np.sin(2*np.pi*freq*t)
        # 'linear' leaves positions unchanged
        return positions

    def generate(self,
                 main_samples=200,
                 approach_samples=50,
                 retract_samples=50,
                 pattern='sine',
                 amp=0.02,
                 freq=2):
        """
        Build a complete trajectory:
        1) Approach from home pose to first waypoint.
        2) Main stroke over region waypoints.
        3) Retract back to home pose.

        Parameters:
            main_samples: int, number of samples for the main stroke spline.
            approach_samples: int, samples for the approach segment.
            retract_samples: int, samples for the retract segment.
            pattern: str, 'sine' | 'circular' | 'linear'.
            amp: float, amplitude for pattern.
            freq: float, frequency for pattern.

        Returns:
            List[dict]: each with 'position' (list[float]) and 'orientation' (list[float]).
        """
        traj = []
        try:
            # 1) Approach
            first = self.waypoints[0]
            approach_pts = self._linspace_positions(self.home_pose['position'], first, approach_samples)
            for pos in approach_pts:
                traj.append({'position': pos.tolist(), 'orientation': self.orientation})

            # 2) Main stroke
            spline_pts, _ = self._interpolate_spline(self.waypoints, main_samples)
            stroke_pts = self._apply_pattern(spline_pts.copy(), pattern, amp, freq)
            for pos in stroke_pts:
                traj.append({'position': pos.tolist(), 'orientation': self.orientation})

            # 3) Retract
            last = self.waypoints[-1]
            retract_pts = self._linspace_positions(last, self.home_pose['position'], retract_samples)
            for pos in retract_pts:
                traj.append({'position': pos.tolist(), 'orientation': self.orientation})

        except Exception as e:
            raise RuntimeError(f"Error generating trajectory: {e}")

        return traj


def test_generate_settings():
    """Test various regions and patterns to validate trajectory generation."""
    regions = [
        ('lower-back', [[0,0,0],[0.1,0,0],[0.2,0,0]]),
        ('upper-back', [[0.2,0,0.1],[0.3,0,0.1],[0.4,0,0.1]])
    ]
    patterns = [
        {'pattern':'sine','amp':0.02,'freq':2},
        {'pattern':'circular','amp':0.03,'freq':1},
        {'pattern':'linear','amp':0.0,'freq':0.0},
    ]
    errors = []
    for region_name, waypts in regions:
        for p in patterns:
            try:
                pg = PathGenerationProgram(waypts)
                traj = pg.generate(pattern=p['pattern'], amp=p['amp'], freq=p['freq'])
                print(f"{region_name} | {p['pattern']} -> {len(traj)} points")
            except Exception as e:
                errors.append(f"{region_name} with {p}: {e}")
    if errors:
        for err in errors:
            print("ERROR:", err)
        raise AssertionError("Trajectory generation tests failed.")
    print("All trajectory generation tests passed.")


if __name__ == '__main__':
    test_generate_settings()
