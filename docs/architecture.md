# System Architecture

The autonomous robotic massage simulation system comprises:

- **Simulation Core**: `env.py`  
  • URDF loading, IK, physics stepping, TensorBoard logging  
- **Sensor Simulation**: `sensors.py`  
  • RGB-D camera and LiDAR emulation  
- **Surface Mapping**: `mapping.py`  
  • Depth back-projection and mesh reconstruction  
- **Coordinate Transforms**: `transforms.py`  
  • Frame and point conversion utilities  
- **Path Planning**:  
  - `generate_path.py`: approach → main stroke → retract splines + patterns  
  - `path_planner.py`: grid/region-based planning over reconstructed mesh  
- **Force Control**:  
  - `force_control.py`: PID torque computation  
  - `pressure_control.py`: reward shaping + DQN/PPO training wrappers  
- **Safety Mechanisms**: `safety.py`  
  • Force & velocity limits, emergency-stop triggers  
- **Execution Engine**: `execution.py`  
  • Grid-based stroke dispatcher integrating IK, control, and safety  
- **Supervised Massage Primitives**: `do_massage.py`  
  • Kneading & linear pressure sweeps for lower- and upper-back  
- **Reinforcement Learning Interface**: `gym_wrapper.py`  
  • `MassageEnv-v1` combining vision, path, control, and safety  
- **RL Agents**:  
  - `dqn.py`: DQN training  
  - `ppo.py`: PPO training (vectorized envs & EvalCallback)  
- **User Interface**: `gui.py`  
  • PyQt5 app with region selection, “Start Massage” button, status display  
- **Tests**: `test/`  
  • Pytest suite covering core modules, RL env, and GUI smoke tests
