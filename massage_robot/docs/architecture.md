# System Architecture

The robotic massage system comprises:

- **Simulation Core** (`env.py`): PyBullet-based physics and human phantom.
- **Sensors** (`sensors.py`): RGB-D and LiDAR emulation.
- **Mapping** (`mapping.py`): Surface reconstruction from point clouds.
- **Planning** (`path_planner.py`): Spline-based waypoint generation over mesh.
- **Control** (`force_control.py`): PID and RL-based force controllers.
- **Safety** (`safety.py`): Force/velocity monitoring and e-stop.
- **Execution** (`execution.py`): Grid-based stroke dispatcher.
- **UI** (`gui.py`): Interactive visualization and metrics.
- **RL Training** (`gym_wrapper.py`, `dqn.py`, `ppo.py`)
