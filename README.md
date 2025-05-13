# Autonomous Robotic Massage Simulation

A modular Python framework for simulating and controlling an autonomous robotic massage system. Key components:

- **Simulation Core** (`env.py`): PyBullet-based environment with URDF loading, inverse kinematics, simulation stepping, and TensorBoard logging.
- **Sensor Simulation** (`sensors.py`): RGB-D camera and LiDAR scan emulation.
- **Surface Mapping** (`mapping.py`): Back-projection of depth images and mesh reconstruction via Delaunay.
- **Coordinate Transforms** (`transforms.py`): Utilities for pose matrix creation and point transformations.
- **Path Generation** (`generate_path.py`): Full trajectory generation including approach, main stroke (sine/circular/linear), and retract phases.
- **Grid Planning** (`path_planner.py`): High-level route planning over a region grid on reconstructed mesh.
- **Supervised Massage Primitives** (`do_massage.py`): Kneading and pressure-sweep techniques for lower- and upper-back.
- **Force Control** (`force_control.py`, `pressure_control.py`): PID torque commands, reward shaping, and RL training scripts for DQN/PPO.
- **Safety Mechanisms** (`safety.py`): Force and velocity limit checks with emergency-stop logic.
- **Execution Engine** (`execution.py`): Grid-based stroke dispatcher integrating planning, control, and safety.
- **Reinforcement Learning Interface** (`gym_wrapper.py`): `MassageEnv-v1` combining vision, path, control, and safety for RL.
- **RL Agents** (`dqn.py`, `ppo.py`): Training entrypoints with evaluation callbacks and TensorBoard logging.
- **User Interface** (`gui.py`): PyQt5 application for region selection, execution control, and status display.
- **URDF & Mesh Assets** (`massage_robot/urdf`, `meshes/`): Xacro URDF for the Panda+tool+bench+phantom and placeholder meshes.
- **Testing Suite** (`test/`): pytest tests covering all core modules, RL env, and GUI smoke tests.
- **Documentation** (`docs/`): Architecture overview, user manual, and API reference.

## Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:USERNAME/massage_robot.git
   cd massage_robot
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **URDF & Meshes**:
   - Place your robot, tool, bench, and phantom URDFs under `massage_robot/urdf/`.
   - Place mesh files (STL) under `massage_robot/meshes/`.

## Quick Start

- **Run the simulation viewer**:
  ```bash
  python -m massage_robot.test_viewer
  ```
- **Execute supervised massage primitives**:
  ```bash
  python -c "from massage_robot.do_massage import kneading_lower_back; print(len(kneading_lower_back()))"
  ```
- **Launch the GUI**:
  ```bash
  python -m massage_robot.gui
  ```

## Reinforcement Learning

- **DQN Training**:
  ```bash
  python -m massage_robot.dqn
  ```
- **PPO Training**:
  ```bash
  python -m massage_robot.ppo
  ```
- **Monitor with TensorBoard**:
  ```bash
  tensorboard --logdir=runs/ --port=6006
  ```

## Testing

Run all unit tests and smoke tests:
```bash
pytest -q
```

## Documentation

HTML and markdown docs are located in the `docs/` directory:
- **Architecture**: high-level design and module interactions (`architecture.md`).
- **User Manual**: setup, usage, and extension guidelines (`user_manual.md`).
- **API Reference**: detailed class and function definitions (`api_reference.md`).

## Extending and Customizing

- **Path Patterns**: modify or add patterns in `generate_path.py`.
- **Controller Gains**: tune PID parameters in `force_control.py` and `pressure_control.py`.
- **Grid Layouts**: configure region maps and vertex indices in `execution.py` or via the GUI.
- **Visual Inputs**: enable/disable camera observations in `gym_wrapper.py`.

Feel free to explore, extend, and contribute!
