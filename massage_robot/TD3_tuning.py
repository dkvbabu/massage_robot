import numpy as np
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from collections import deque
import pybullet as p

from env import MassageEnv  # Import your environment class from env.py
from TD3_train import Actor, Critic, ReplayBuffer, TD3, local_step, local_reset, get_action_bounds  # Import other classes and functions


def hyperparameter_tuning():
    learning_rates = [1e-3, 3e-4, 1e-4]
    batch_sizes = [64, 100, 256]
    discount_factors = [0.95, 0.99, 0.999]
    tau_values = [0.005, 0.01]
    policy_noise_values = [0.1, 0.2, 0.3]
    noise_clip_values = [0.3, 0.5]
    policy_freq_values = [2, 3]

    os.makedirs("tuning_results", exist_ok=True)

    all_avg_rewards = []
    all_avg_actor_losses = []
    all_avg_critic_losses = []
    all_param_sets = []

    episodes = 10
    start_timesteps = 300

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for discount in discount_factors:
                for tau in tau_values:
                    for policy_noise in policy_noise_values:
                        for noise_clip in noise_clip_values:
                            for policy_freq in policy_freq_values:
                                print(f"Testing params: lr={lr}, batch_size={batch_size}, discount={discount}, tau={tau}, "
                                      f"policy_noise={policy_noise}, noise_clip={noise_clip}, policy_freq={policy_freq}")

                                env = MassageEnv(render=False)
                                stats = env.collect_stats()
                                state_dim = len(env.get_state(stats))
                                action_dim = 3
                                max_action = 1.0

                                agent = TD3(state_dim, action_dim, max_action)
                                agent.actor_optimizer = optim.Adam(agent.actor.parameters(), lr=lr)
                                agent.critic_optimizer = optim.Adam(agent.critic.parameters(), lr=lr)
                                agent.discount = discount
                                agent.tau = tau
                                agent.policy_noise = policy_noise
                                agent.noise_clip = noise_clip
                                agent.policy_freq = policy_freq

                                replay_buffer = ReplayBuffer()

                                episode_length = 1440
                                total_steps = 0
                                episode_rewards = []
                                actor_losses = []
                                critic_losses = []

                                min_bounds, max_bounds = get_action_bounds(env, margin=0)

                                alpha = 0.01  # smoothing factor for interpolation
                                prev_action = np.zeros(action_dim)
                                smooth_target_prev = None

                                early_stop = False

                                for episode in range(episodes):
                                    local_reset(env)
                                    env.make_path()
                                    stats = env.collect_stats()
                                    state = env.get_state(stats)
                                    state = state / np.linalg.norm(state) if np.linalg.norm(state) > 0 else state

                                    episode_reward = 0

                                    for t in range(episode_length):
                                        # Disable collisions for all arm links except end effector and link 7
                                        for link_idx in range(p.getNumJoints(env.armId)):
                                            if link_idx not in [env.EndEfferctorId, 7]:
                                                p.setCollisionFilterPair(env.armId, env.human_inst.body, link_idx, -1, enableCollision=0, physicsClientId=env.SimID)

                                        action = agent.select_action(state)
                                        action = action + np.random.normal(0, 0.1, size=action_dim)

                                        workspace_range = max_bounds - min_bounds
                                        action = min_bounds + (action + 1) / 2 * workspace_range

                                        # Oscillate x target back and forth
                                        x_min = np.min(env.pntsAndReturn[:, 0])
                                        x_max = np.max(env.pntsAndReturn[:, 0])
                                        x_range = x_max - x_min
                                        oscillation_period = episode_length / 2
                                        x_oscillate = x_min + (x_range / 2) * (1 + np.sin(2 * np.pi * t / oscillation_period))

                                        y_fixed = 0.3
                                        z_fixed = 0.95

                                        action[0] = x_oscillate
                                        action[1] = y_fixed  # fixed y since min and max are equal
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

                                        local_step(env, smooth_target)

                                        stats = env.collect_stats()
                                        next_state = env.get_state(stats)
                                        next_state = next_state / np.linalg.norm(next_state) if np.linalg.norm(next_state) > 0 else next_state

                                        reward = env.get_reward(stats)
                                        done = (t == episode_length - 1)

                                        replay_buffer.add((state, smooth_target, reward, next_state, float(done)))

                                        state = next_state
                                        episode_reward += reward
                                        total_steps += 1

                                        if total_steps >= start_timesteps:
                                            actor_loss, critic_loss = agent.train(replay_buffer, batch_size)
                                            if actor_loss is not None:
                                                actor_losses.append(actor_loss)
                                                critic_losses.append(critic_loss)

                                        if done:
                                            break

                                    episode_rewards.append(episode_reward)
                                    print(f"Episode {episode + 1}, Reward: {episode_reward:.3f}")

                                    if episode_rewards and np.mean(episode_rewards[-5:]) < 50.0:
                                        print(f"Early stopping: Average reward below threshold at episode {episode+1}")
                                        early_stop = True

                                    if early_stop:
                                        break

                                avg_reward = np.mean(episode_rewards)
                                avg_actor_loss = np.mean(actor_losses) if actor_losses else float('nan')
                                avg_critic_loss = np.mean(critic_losses) if critic_losses else float('nan')
                                print(f"Average reward for current params: {avg_reward:.3f}")
                                print(f"Average actor loss: {avg_actor_loss:.6f}")
                                print(f"Average critic loss: {avg_critic_loss:.6f}")

                                all_avg_rewards.append(avg_reward)
                                all_avg_actor_losses.append(avg_actor_loss)
                                all_avg_critic_losses.append(avg_critic_loss)
                                all_param_sets.append({
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'discount': discount,
                                    'tau': tau,
                                    'policy_noise': policy_noise,
                                    'noise_clip': noise_clip,
                                    'policy_freq': policy_freq
                                })

                                env.close()

    # Normalize metrics for scoring
    r_min, r_max = min(all_avg_rewards), max(all_avg_rewards)
    c_min, c_max = min(all_avg_critic_losses), max(all_avg_critic_losses)
    a_min, a_max = min(all_avg_actor_losses), max(all_avg_actor_losses)

    # Weights for metrics (adjust as needed)
    w_r, w_c, w_a = 0.6, 0.25, 0.15

    best_score = -float('inf')
    best_params = None
    best_index = -1

    for i in range(len(all_avg_rewards)):
        norm_r = (all_avg_rewards[i] - r_min) / (r_max - r_min) if r_max > r_min else 1.0
        norm_c = (all_avg_critic_losses[i] - c_min) / (c_max - c_min) if c_max > c_min else 0.0
        norm_a = (all_avg_actor_losses[i] - a_min) / (a_max - a_min) if a_max > a_min else 0.0

        score = w_r * norm_r - w_c * norm_c - w_a * norm_a

        if score > best_score:
            best_score = score
            best_params = all_param_sets[i]
            best_index = i

    print("\nHyperparameter tuning completed.")
    print(f"Best combined score: {best_score:.4f}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Plotting results
    x = range(len(all_avg_rewards))
    labels = [f"lr={p['learning_rate']},bs={p['batch_size']},disc={p['discount']:.2f}" for p in all_param_sets]

    plt.figure(figsize=(15, 8))

    plt.subplot(3, 1, 1)
    plt.bar(x, all_avg_rewards)
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.ylabel('Average Reward')
    plt.title('Hyperparameter Tuning: Average Reward')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.bar(x, all_avg_actor_losses, color='orange')
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.ylabel('Avg Actor Loss')
    plt.title('Hyperparameter Tuning: Average Actor Loss')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.bar(x, all_avg_critic_losses, color='green')
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.ylabel('Avg Critic Loss')
    plt.title('Hyperparameter Tuning: Average Critic Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hyperparameter_tuning()
