import torch
import numpy as np
import pybullet as p
import pybullet_data
from env import MassageEnv  # Import MassageEnv from env.py as in TD3_train.py
from TD3_train import Actor, local_step, get_action_bounds  # Import other needed components
from torch.utils.tensorboard import SummaryWriter


def load_agent(state_dim, action_dim, max_action, actor_path):
    agent = Actor(state_dim, action_dim, max_action)
    agent.load_state_dict(torch.load(actor_path))
    agent.eval()
    return agent


def run_inference(env, agent, episode_length, writer=None):
    alpha = 0.01  # smoothing factor for interpolation
    prev_action = np.zeros(3)  # since action_dim is 3
    smooth_target_prev = None

    stats = env.collect_stats()
    state = env.get_state(stats)
    total_reward = 0.0

    min_bounds, max_bounds = get_action_bounds(env, margin=0)
    print(f"Inference action bounds: min {min_bounds}, max {max_bounds}")

    for t in range(episode_length):
        # Normalize state as in training
        state_norm = state / np.linalg.norm(state) if np.linalg.norm(state) > 0 else state
        state_tensor = torch.FloatTensor(state_norm.reshape(1, -1))

        action = agent(state_tensor).cpu().data.numpy().flatten()

        # Map action from [-1,1] to environment bounds
        workspace_range = max_bounds - min_bounds
        action = min_bounds + (action + 1) / 2 * workspace_range

        # Oscillate x target back and forth
        x_min = np.min(env.pntsAndReturn[:, 0])
        x_max = np.max(env.pntsAndReturn[:, 0])
        x_range = x_max - x_min
        oscillation_period = episode_length
        x_oscillate = x_min + (x_range / 2) * (1 + np.sin(2 * np.pi * t / oscillation_period))

        y_fixed = 0.3
        z_fixed = 0.95

        action[0] = x_oscillate
        action[1] = y_fixed
        action[2] = np.clip(action[2], z_fixed - 0.05, z_fixed + 0.05)

        # Smooth action blending
        action_smooth = 0.8 * prev_action + 0.2 * action
        prev_action = action_smooth

        # Smooth target interpolation
        if smooth_target_prev is None:
            smooth_target = action_smooth
        else:
            smooth_target = alpha * action_smooth + (1 - alpha) * smooth_target_prev
        smooth_target_prev = smooth_target

        smooth_target = np.clip(smooth_target, min_bounds, max_bounds)

        # Use local_step instead of env.step
        local_step(env, smooth_target)

        stats = env.collect_stats()
        state = env.get_state(stats)

        reward = env.get_reward(stats)
        total_reward += reward

        if writer is not None:
            writer.add_scalar('Reward/Step', reward, t)

        p.stepSimulation()

    if writer is not None:
        writer.add_scalar('Reward/Episode', total_reward, 0)

    print(f'Total reward in inference: {total_reward}')


def main():
    render = True
    env = MassageEnv(render=render)

    stats = env.collect_stats()
    state_dim = len(env.get_state(stats))
    action_dim = 3  # Adjust if your action space differs
    max_action = 1.0

    actor_path = "models/actor_final.pth"  # Path to your saved actor model

    agent = load_agent(state_dim, action_dim, max_action, actor_path)

    episode_length = 720

    writer = SummaryWriter(log_dir="./runs/td3_inference")

    run_inference(env, agent, episode_length, writer)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()