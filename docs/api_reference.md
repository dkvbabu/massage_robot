

**docs/api_reference.md**

# API Reference

## Simulation Core
- **`MassageEnv(gui=True, time_step=1/240., log_dir)`**  
  - `reset() → observation`  
  - `step(joint_commands, render=True) → observation`  
  - `calculate_ik(position, orientation) → joint_angles`  
  - `disconnect()`

## Sensor Simulation
- **`SensorSimulator(env, depth_noise=0.01)`**  
  - `get_rgbd(cam_pose, fov, resolution) → (rgb, depth)`  
  - `get_lidar_scan(origin, angles, max_range) → ranges`

## Surface Mapping
- **`SurfaceMapper()`**  
  - `add_scan(depth_image, intrinsics)`  
  - `reconstruct() → (points, faces)`

## Coordinate Transforms
- **`CoordinateTransforms.pose_to_matrix(position, quaternion) → 4×4 matrix`**  
- **`CoordinateTransforms.transform_points(points, matrix) → transformed_points`**

## Path Generation & Planning
- **`PathGenerationProgram(region_waypoints, orientation, home_offset)`**  
  - `generate(main_samples, approach_samples, retract_samples, pattern, amp, freq) → trajectory`  
  - `test_generate_settings()`
- **`PathPlanner(vertices, simplices)`**  
  - `plan_through_waypoints(waypoints, samples) → ndarray`  
  - `plan_grid(grid_map, region_vertices_map, samples_per_region) → ndarray`

## Massage Primitives
- **`kneading_lower_back(num_strokes, stroke_amplitude, samples_per_stroke) → trajectory`**  
- **`pressure_sweep_lower_back(passes, main_samples) → trajectory`**  
- **`kneading_upper_back(...)`**, **`pressure_sweep_upper_back(...)`**

## Force Control & Training Helpers
- **`ForceController(kp, ki, kd, target_force)`**  
  - `compute_torque(current_force, dt) → torque`  
  - `reward(current_force) → float`
- **`train_dqn(...)`** & **`train_ppo(...)`** in `pressure_control.py`

## Safety
- **`SafetyModule(max_force, max_velocity)`**  
  - `verify(forces, velocities) → (bool, message)`

## Execution Engine
- **`GridExecutor(planner, controller, safety, env)`**  
  - `execute(grid_map, region_vertices_map, samples_per_region, dt) → list[observations]`

## Reinforcement Learning Interface
- **`MassageGymEnv`** (Gym Env v1)  
  - Integrates vision, path planning, force control, and safety as a single RL environment.

## RL Agents
- **`dqn.py`**: entrypoint for DQN training  
- **`ppo.py`**: entrypoint for PPO training

## User Interface
- **`MassageGUI(env, executor)`** (PyQt5)  
  - `show()`, `run_massage()`
