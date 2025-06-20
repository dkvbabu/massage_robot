"""PPO RL agent training stub with TensorBoard logging and custom environment support."""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import time
import random
from massage_robot.gym_wrapper import MassageEnvGym

class Agent(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_actions)
        )
    def get_value(self, x):
        return self.critic(x)
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def train_ppo(env_id="CartPole-v1", use_massage_env=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 1
    num_steps = 128
    total_timesteps = 5000
    update_epochs = 4
    num_minibatches = 4
    gamma = 0.99
    gae_lambda = 0.95
    learning_rate = 2.5e-4
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    if use_massage_env:
        envs = gym.vector.SyncVectorEnv([lambda: MassageEnvGym() for _ in range(num_envs)])
    else:
        envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)])
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    n_actions = envs.single_action_space.n
    agent = Agent(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    writer = SummaryWriter(log_dir="runs/ppo_example")
    for iteration in range(1, total_timesteps // batch_size + 1):
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        envs.close()
        writer.close()

if __name__ == "__main__":
     train_ppo(use_massage_env=True)
    # To use your massage robot env, call: train_ppo(use_massage_env=True)
