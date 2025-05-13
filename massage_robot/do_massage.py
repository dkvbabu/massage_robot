# File: massage_robot/do_massage.py
"""
Implement supervised massage techniques for lower-back and upper-back:
- Kneading: oscillatory strokes (sine pattern)
- Pressure sweep: linear strokes applying constant pressure
"""
from massage_robot.generate_path import PathGenerationProgram

# Region-specific waypoints (x, y, z) in meters
LOWER_BACK_WAYPOINTS = [
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.2, 0.0, 0.0]
]
UPPER_BACK_WAYPOINTS = [
    [0.2, 0.0, 0.1],
    [0.3, 0.0, 0.1],
    [0.4, 0.0, 0.1]
]
# Default end-effector orientation (quaternion)
DEFAULT_ORIENTATION = [0, 0, 0, 1]


def kneading_lower_back(num_strokes=5, stroke_amplitude=0.02, samples_per_stroke=100):
    """
    Perform kneading on the lower back using a sine-wave pattern.
    :param num_strokes: Number of back-and-forth cycles
    :param stroke_amplitude: Lateral amplitude of the stroke (meters)
    :param samples_per_stroke: Samples per stroke cycle (higher = smoother)
    :return: List of pose dicts for the trajectory
    """
    pg = PathGenerationProgram(
        region_waypoints=LOWER_BACK_WAYPOINTS,
        orientation=DEFAULT_ORIENTATION
    )
    # Total main samples = cycles * samples per cycle
    main_samples = num_strokes * samples_per_stroke
    return pg.generate(
        main_samples=main_samples,
        approach_samples=50,
        retract_samples=50,
        pattern='sine',
        amp=stroke_amplitude,
        freq=num_strokes
    )


def pressure_sweep_lower_back(passes=3, main_samples=200):
    """
    Perform linear pressure sweeps on the lower back.
    :param passes: Number of repeated sweeps
    :param main_samples: Samples for each main sweep segment
    :return: List of pose dicts for the trajectory
    """
    pg = PathGenerationProgram(
        region_waypoints=LOWER_BACK_WAYPOINTS,
        orientation=DEFAULT_ORIENTATION
    )
    # Generate a single linear sweep
    traj = []
    for _ in range(passes):
        traj += pg.generate(
            main_samples=main_samples,
            approach_samples=50,
            retract_samples=50,
            pattern='linear',
            amp=0.0,
            freq=0.0
        )
    return traj


def kneading_upper_back(num_strokes=5, stroke_amplitude=0.02, samples_per_stroke=100):
    """
    Perform kneading on the upper back using a sine-wave pattern.
    :param num_strokes: Number of back-and-forth cycles
    :param stroke_amplitude: Lateral amplitude of the stroke (meters)
    :param samples_per_stroke: Samples per stroke cycle
    :return: List of pose dicts for the trajectory
    """
    pg = PathGenerationProgram(
        region_waypoints=UPPER_BACK_WAYPOINTS,
        orientation=DEFAULT_ORIENTATION
    )
    main_samples = num_strokes * samples_per_stroke
    return pg.generate(
        main_samples=main_samples,
        approach_samples=50,
        retract_samples=50,
        pattern='sine',
        amp=stroke_amplitude,
        freq=num_strokes
    )


def pressure_sweep_upper_back(passes=3, main_samples=200):
    """
    Perform linear pressure sweeps on the upper back.
    :param passes: Number of repeated sweeps
    :param main_samples: Samples for each main sweep segment
    :return: List of pose dicts for the trajectory
    """
    pg = PathGenerationProgram(
        region_waypoints=UPPER_BACK_WAYPOINTS,
        orientation=DEFAULT_ORIENTATION
    )
    traj = []
    for _ in range(passes):
        traj += pg.generate(
            main_samples=main_samples,
            approach_samples=50,
            retract_samples=50,
            pattern='linear',
            amp=0.0,
            freq=0.0
        )
    return traj


if __name__ == '__main__':
    # Example usage and basic testing
    for fn, name in [
        (kneading_lower_back, 'Kneading Lower Back'),
        (pressure_sweep_lower_back, 'Pressure Sweep Lower Back'),
        (kneading_upper_back, 'Kneading Upper Back'),
        (pressure_sweep_upper_back, 'Pressure Sweep Upper Back')
    ]:
        traj = fn()
        print(f"{name}: generated {len(traj)} poses.")
