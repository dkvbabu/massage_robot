# API Reference

Each module provides:

- `env.MassageEnv(gui, time_step)`: core simulation class.
- `sensors.SensorSimulator(env)`: simulator class.
- `mapping.SurfaceMapper()`: mapper class.
- `transforms.CoordinateTransforms`: static transform utilities.
- `path_planner.PathPlanner(vertices, simplices)`: path planning class.
- `force_control.ForceController(kp, ki, kd, target_force)`: PID control class.
- `safety.SafetyModule(max_force, max_velocity)`: safety monitoring class.
- `execution.GridExecutor(planner, controller, safety)`: executor class.
- `gui.MassageGUI()`: UI stub.
``