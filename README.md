# Autonomous Robotic Massage Simulation

This repository contains a modular simulation framework for an autonomous robotic massage system. Components include:

- **Environment**: PyBullet-based simulator with a 7-DOF arm and deformable human phantom (`env.py`).
- **Sensors**: RGB-D and LiDAR simulation (`sensors.py`).
- **Mapping**: Surface reconstruction from depth scans (`mapping.py`).
- **Transforms**: Coordinate frame utilities (`transforms.py`).
- **Path Planning**: Spline-based trajectory planning over mesh surfaces (`path_planner.py`).
- **Control**: PID and RL-based pressure/force controllers (`force_control.py`, `pressure_control.py`).
- **Safety**: Force/velocity checks and emergency-stop logic (`safety.py`).
- **Execution**: Grid-based massage stroke executor (`execution.py`).
- **Gym Wrapper**: Custom Gym environment for RL training (`gym_wrapper.py`).
- **RL Agents**: DQN (`dqn.py`) and PPO (`ppo.py`) training entrypoints.
- **UI**: Stub for interactive user interface (`gui.py`).
- **Tests**: Unit tests for key modules (`test/`).
- **Docs**: Architecture, user manual, and API reference (`docs/`).

## Setup

1. Clone:
   ```bash
   git clone <repo_url> massage_robot
   cd massage_robot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize Git:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: project skeleton"
   ```

## Usage

- Launch UI (stub): `python -m massage_robot.gui`
- Run viewer: `python -m massage_robot.test_viewer`
- Train DQN: `python -m massage_robot.dqn`
- Train PPO: `python -m massage_robot.ppo`

See [docs](docs/) for details.
