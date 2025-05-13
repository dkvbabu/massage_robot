# Updated RL integration: gym_wrapper.py & dqn.py

# File: massage_robot/gym_wrapper.py
import gym
import numpy as np
import pybullet as p
from gym import spaces
from gym.envs.registration import register
from torch.utils.tensorboard import SummaryWriter
from massage_robot.env import MassageEnv
from massage_robot.generate_path import PathGenerationProgram
from massage_robot.force_control import ForceController
from massage_robot.safety import SafetyModule
from massage_robot.sensors import SensorSimulator

register(
    id='MassageEnv-v1',
    entry_point='massage_robot.gym_wrapper:MassageGymEnv',
    kwargs={
        'waypoints_lower': [[0,0,0],[0.1,0,0],[0.2,0,0]],
        'waypoints_upper': [[0.2,0,0.1],[0.3,0,0.1],[0.4,0,0.1]],
        'orientation': [0,0,0,1],
        'main_samples': 200,
        'approach_samples': 50,
        'retract_samples': 50,
        'use_visual': True,
        'visual_resolution': (84,84)
    }
)

class MassageGymEnv(gym.Env):
    """Gym adapter combining path-gen, IK, force control, safety, and vision."""
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 waypoints_lower,
                 waypoints_upper,
                 orientation,
                 main_samples,
                 approach_samples,
                 retract_samples,
                 use_visual=False,
                 visual_resolution=(84,84)):
        super().__init__()
        # Core simulation
        self.base_env = MassageEnv(gui=False)
        self.sensor = SensorSimulator(self.base_env)
        self.fc = ForceController()
        self.safety = SafetyModule()
        self.writer = SummaryWriter("runs/rl_training")
        # Path generators
        self.pg_lower = PathGenerationProgram(waypoints_lower, orientation)
        self.pg_upper = PathGenerationProgram(waypoints_upper, orientation)
        # Config
        self.use_visual = use_visual
        self.resolution = visual_resolution
        # Pre-generate two trajectories
        self.traj_lower = self.pg_lower.generate(main_samples, approach_samples, retract_samples)
        self.traj_upper = self.pg_upper.generate(main_samples, approach_samples, retract_samples)
        # Combined segments: alternate lower+upper
        self.trajectory = self.traj_lower + self.traj_upper
        self.step_idx = 0
        # Observation & action spaces
        num_joints = p.getNumJoints(self.base_env.robot)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        # State = [joint_pos, joint_vel, forces] (+ image if visual)
        obs_dim = num_joints*2 + num_joints
        if self.use_visual:
            # Add flattened grayscale image
            obs_dim += visual_resolution[0]*visual_resolution[1]
        self.observation_space = spaces.Box(low=0, high=255, shape=(obs_dim,), dtype=np.float32)

    def reset(self):
        obs_dict = self.base_env.reset()
        self.step_idx = 0
        return self._build_obs(obs_dict)

    def step(self, action):
        # Path target
        target = self.trajectory[self.step_idx]
        pos, orn = target['position'], target['orientation']
        # IK -> joint targets
        joint_targets = self.base_env.calculate_ik(pos, orn)
        # Apply action residual
        commands = [jt + action[i] for i, jt in enumerate(joint_targets)]
        # Step sim
        obs_dict = self.base_env.step(commands, render=False)
        # Compute reward
        eff_idx = self.base_env.end_effector_link
        curr_force = obs_dict['contact_forces'].get(eff_idx, 0.0)
        reward_force = self.fc.reward(curr_force)
        # Safety check
        ok, msg = self.safety.check(obs_dict['contact_forces'].values(),
                                     [v for _, v in obs_dict['joint_states']])
        done = not ok or (self.step_idx >= len(self.trajectory)-1)
        # Build observation
        obs = self._build_obs(obs_dict)
        # Logging
        self.writer.add_scalar("reward/force", reward_force, self.step_idx)
        if self.use_visual:
            # log a single image per episode step
            img = self._get_visual()
            self.writer.add_image("env/vision", img, self.step_idx, dataformats='HW')
        self.step_idx += 1
        info = {} if ok else {'error': msg}
        return obs, reward_force, done, info

    def _get_visual(self):
        cam_pose = {
            'target': [0.3,0,0.5],
            'distance': 1.0,
            'yaw': 45,
            'pitch': -30,
            'roll': 0
        }
        rgb, depth = self.sensor.get_rgbd(cam_pose, resolution=self.resolution)
        # convert to grayscale
        gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        return gray.astype(np.uint8)

    def _build_obs(self, obs_dict):
        joints = np.array(obs_dict['joint_states']).reshape(-1)
        forces = np.array([obs_dict['contact_forces'].get(i,0.0)
                           for i in range(p.getNumJoints(self.base_env.robot))])
        obs = np.concatenate([joints, forces])
        if self.use_visual:
            img = self._get_visual().flatten()
            obs = np.concatenate([obs, img])
        return obs.astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        self.base_env.disconnect()
