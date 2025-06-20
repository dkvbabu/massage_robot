"""DQN RL agent training stub with TensorBoard logging and custom environment support."""
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
import random
import time
import os
from massage_robot.gym_wrapper import MassageEnvGym

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions),
        )
    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def train_dqn(env_id="CartPole-v1", use_massage_env=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_massage_env:
        envs = gym.vector.SyncVectorEnv([lambda: MassageEnvGym()])
    else:
        envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id)])
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    n_actions = envs.single_action_space.n
    q_network = QNetwork(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=2.5e-4)
    target_network = QNetwork(obs_dim, n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    rb = ReplayBuffer(10000, envs.single_observation_space, envs.single_action_space, device, handle_timeout_termination=False)
    writer = SummaryWriter(log_dir="runs/dqn_example")
    obs, _ = envs.reset()
    for global_step in range(10000):
        epsilon = linear_schedule(1.0, 0.05, 5000, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        if global_step > 1000 and global_step % 10 == 0:
            data = rb.sample(128)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + 0.99 * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        if global_step % 500 == 0:
            target_network.load_state_dict(q_network.state_dict())
    envs.close()
    writer.close()

if __name__ == "__main__":
     train_dqn(use_massage_env=True)
    # To use your massage robot env, call: train_dqn(use_massage_env=True)
