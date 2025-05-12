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

register(
    id='MassageEnv-v0',
    entry_point='massage_robot.gym_wrapper:MassageGymEnv',
    kwargs={
        'waypoints': [[0,0,0],[0.1,0,0],[0.2,0,0],[0.3,0,0],[0.4,0,0]],
        'orientation': [0,0,0,1],
        'total_samples': 200
    }
)

class MassageGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, waypoints, orientation, total_samples=200):
        super().__init__()
        self.env = MassageEnv(gui=False)
        self.pg = PathGenerationProgram(waypoints, orientation)
        self.fc = ForceController()
        self.safety = SafetyModule()
        self.writer = SummaryWriter("runs/massage_run")

        num_joints = p.getNumJoints(self.env.robot)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_joints,), dtype=np.float32)
        obs_dim = num_joints*2 + num_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.trajectory = self.pg.generate(total_samples=total_samples)
        self.step_idx = 0

    def reset(self):
        obs_dict = self.env.reset()
        self.step_idx = 0
        return self._dict_to_obs(obs_dict)

    def step(self, action):
        pose = self.trajectory[self.step_idx]
        pos, orn = pose['position'], pose['orientation']
        joint_targets = self.env.calculate_ik(pos, orn)
        current_forces = self.env._read_force_sensors()
        eff_idx = self.env.end_effector_link
        curr_force = current_forces.get(eff_idx, 0.0)
        tau = self.fc.compute(curr_force)
        joint_commands = [jt + action[i] for i, jt in enumerate(joint_targets)]
        obs_dict = self.env.step(joint_commands, render=False)
        obs = self._dict_to_obs(obs_dict)
        reward = self.fc.reward(obs_dict['contact_forces'].get(eff_idx, 0.0))
        done = self.step_idx >= (len(self.trajectory) - 1)
        info = {}
        self.writer.add_scalar("env/reward", reward, self.step_idx)
        self.writer.add_scalar("env/pressure", obs_dict['contact_forces'].get(eff_idx, 0.0), self.step_idx)
        self.step_idx += 1
        return obs, reward, done, info

    def _dict_to_obs(self, obs_dict):
        joint_states = np.array(obs_dict['joint_states']).reshape(-1)
        forces = np.array([obs_dict['contact_forces'].get(i, 0.0)
                           for i in range(p.getNumJoints(self.env.robot))])
        return np.concatenate([joint_states, forces]).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.disconnect()
