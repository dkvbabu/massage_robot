# User Manual

## 1. Setup

1. **Clone the repository**

   ```bash
   git clone git@github.com:USERNAME/massage_robot.git
   cd massage_robot
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Manage assets**

   * Place URDF files under `massage_robot/urdf/` (robot arm, tool, bench, phantom).
   * Place mesh files (STL/OBJ) under `massage_robot/meshes/`.
   * Ensure code paths in `env._load_models()` match your asset locations.

## 2. Quick Simulation & Primitives

* **Run the simulation viewer**

  ```bash
  python -m massage_robot.test_viewer
  ```

  Opens a PyBullet GUI showing the robot, bench, and phantom. Use `Ctrl+C` to exit.

* **Test supervised massage primitives**

  ```bash
  python -c "from massage_robot.do_massage import kneading_lower_back; print(len(kneading_lower_back()))"
  ```

  Prints the total poses generated for a default lower-back kneading sequence.

## 3. User Interface

Launch the PyQt5 GUI:

```bash
python -m massage_robot.gui
```

* **Region Configuration**: modify `executor.grid_map` and `executor.region_vertices_map` in code, or extend the GUI to select regions interactively.
* **Start Massage**: click **Start Massage** to execute grid-based strokes.
* **Status Display**: shows `Running...`, `Completed`, or error messages upon safety triggers.

## 4. Reinforcement Learning

### DQN Training

```bash
python -m massage_robot.dqn
```

* Trains a DQN agent on `MassageEnv-v1`.
* Logs metrics to TensorBoard under `runs/rl_training/dqn/`.
* Best models saved to `models/dqn_best/`.

### PPO Training

```bash
python -m massage_robot.ppo
```

* Trains a PPO agent using 4 parallel environments.
* Logs metrics under `runs/ppo_training/` and evaluations under `logs/ppo_eval/`.
* Best models saved to `models/ppo_best/`.

### Monitor with TensorBoard

```bash
tensorboard --logdir=runs/ --port=6006
```

Visit [http://localhost:6006](http://localhost:6006) to inspect:

* **Scalars**: reward curves, losses, etc.
* **Images**: if `use_visual=True`, grayscale camera frames.

## 5. Testing

Run all tests using pytest:

```bash
pytest -q
```

Includes:

* Path planning (`test_path_planner.py`)
* Force control (`test_force_control.py`)
* Safety checks (`test_safety.py`)
* Grid execution (`test_execution.py`)
* GUI smoke tests (`test_gui.py`)

## 6. Extending & Customization

* **Stroke Patterns**: modify or add patterns in `generate_path.py`.
* **Grid Layouts**: configure region grid and vertex mappings in `execution.py` or via GUI.
* **Controller Tuning**: adjust PID gains (`kp`, `ki`, `kd`) and `target_force` in `force_control.py` / `pressure_control.py`.
* **Visual Input**: toggle `use_visual` and change `visual_resolution` in `gym_wrapper.py`.
* **URDF & Meshes**: swap robot/tool/phantom models; update `<inertial>` metadata for realism.
* **CI/CD**: add GitHub Actions for automated testing, linting, and deployment.

---

For full API details, refer to `docs/api_reference.md`.
For architecture diagrams and design rationale, see `docs/architecture.md`.   
