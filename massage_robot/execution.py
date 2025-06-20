from massage_robot.do_massage import *
from massage_robot.env import MassageEnv
import time
import numpy as np

class GridExecutor:
    def __init__(self, planner, controller, safety, env):
        self.planner = planner
        self.controller = controller
        self.safety = safety
        self.env = env

    def execute(self, grid_map, region_map, samples_per_region=5, dt=0.01):
        # Get path from planner
        path = self.planner.plan_grid(grid_map, region_map, samples_per_region)
        results = []
        
        # Execute path
        for point in path:
            # Get joint angles
            joint_angles = self.env.calculate_ik(point, self.env.default_orientation)
            
            # Read forces
            forces = self.env._read_force_sensors()
            
            # Check safety
            velocities = [0.0] * len(joint_angles)  # Simplified
            safe, msg = self.safety.verify(forces, velocities)
            if not safe:
                print(f"Safety violation: {msg}")
                break
                
            # Compute control
            torque = self.controller.compute_torque(forces[0], dt)
            
            # Step simulation
            obs = self.env.step(joint_angles, render=False)
            results.append(obs)
            
        return results

# Store a reference to the running environment for stop control
_running_env = None
_last_stats = None

# This module orchestrates the massage session and is ready for RL, advanced control, and extensibility.

def execute_massage(region, technique, force, speed, duration, repetitions, yaw=0, pitch=0, pattern='sine', amp=0.02, freq=2.0, approach_samples=50, main_samples=200, retract_samples=50):
    global _running_env, _last_stats
    func_name = f"{technique}_{region}"
    func = globals().get(func_name)
    if not func:
        print(f"Unsupported technique/region: {technique}, {region}")
        return
    env = MassageEnv(render=True, pattern=pattern, amp=amp, freq=freq, approach_samples=approach_samples, main_samples=main_samples, retract_samples=retract_samples, force=force, speed=speed)
    _running_env = env
    # Set camera
    env.set_camera(yaw, pitch)
    for rep in range(repetitions):
        params = func(force, speed, duration)
        print(f"Session {rep+1}: {params}")
        steps = int(duration * speed)
        for step in range(steps):
            if _running_env is None:
                print("Session stopped.")
                env.close()
                return
            action = env.get_action()
            env.step(action, force=env.force, speed=env.speed)
            time.sleep(env.TimeStep)
    env.run_until_exit()
    _last_stats = env.get_stats()
    print(f"[DEBUG] Simulation ended. _last_stats: {type(_last_stats)}, Forces length: {len(_last_stats['Forces']) if _last_stats and 'Forces' in _last_stats else 'N/A'}")
    _running_env = None
    return _last_stats

def update_simulation_parameters(**kwargs):
    global _running_env
    if _running_env is not None:
        _running_env.update_parameters(**kwargs)

def update_simulation_camera(yaw=None, pitch=None):
    global _running_env
    if _running_env is not None:
        _running_env.update_camera(yaw=yaw, pitch=pitch)

def stop_massage():
    global _running_env
    _running_env = None

def get_last_stats():
    global _last_stats
    return _last_stats
