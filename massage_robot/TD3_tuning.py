import numpy as np
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from collections import deque

from TD3_train import MassageEnv, Actor, Critic, ReplayBuffer, TD3  # Assuming these are in TD3_train.py


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

    # Early stopping parameters
    min_avg_reward_threshold = 50.0  # stop if avg reward below this after min_episodes
    min_episodes_before_stop = 5  # minimum episodes before checking early stop

    # Adaptive threshold window size
    loss_window_size = 5

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
                                state_dim = len(env.get_state())
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

                                episodes = 10
                                episode_length = 5 * env.PointsInPath
                                start_timesteps = 1000
                                total_steps = 0
                                episode_rewards = []
                                actor_losses = []
                                critic_losses = []

                                min_bounds, max_bounds = env.get_action_bounds(margin=0.05)

                                early_stop = False

                                # Deques for adaptive thresholding
                                recent_critic_losses = deque(maxlen=loss_window_size)
                                recent_actor_losses = deque(maxlen=loss_window_size)

                                for episode in range(episodes):
                                    env.reset()
                                    state = env.get_state()
                                    episode_reward = 0

                                    for t in range(episode_length):
                                        if total_steps < start_timesteps:
                                            base_action = env.get_action()
                                            noise = np.random.normal(0, 0.1, size=base_action.shape)
                                            action = base_action + noise
                                        else:
                                            action = agent.select_action(state)
                                            action = action + np.random.normal(0, 0.1, size=action_dim)

                                        action = np.clip(action, min_bounds, max_bounds)

                                        env.step(action)
                                        next_state = env.get_state()
                                        reward = env.get_reward()
                                        done = (t == episode_length - 1)

                                        replay_buffer.add((state, action, reward, next_state, float(done)))

                                        state = next_state
                                        episode_reward += reward
                                        total_steps += 1

                                        if total_steps >= start_timesteps:
                                            actor_loss, critic_loss = agent.train(replay_buffer, batch_size)
                                            if actor_loss is not None:
                                                actor_losses.append(actor_loss)
                                                recent_actor_losses.append(actor_loss)
                                            critic_losses.append(critic_loss)
                                            recent_critic_losses.append(critic_loss)

                                            # Adaptive threshold check for critic loss
                                            if len(recent_critic_losses) == loss_window_size:
                                                mean_critic = np.mean(recent_critic_losses)
                                                std_critic = np.std(recent_critic_losses)
                                                upper_critic_threshold = mean_critic + 2 * std_critic
                                                if critic_loss > upper_critic_threshold:
                                                    print(f"Early stopping: Critic loss {critic_loss:.3f} exceeded adaptive threshold "
                                                          f"{upper_critic_threshold:.3f} at episode {episode+1}")
                                                    early_stop = True
                                                    break

                                            # Adaptive threshold check for actor loss (optional)
                                            if actor_loss is not None and len(recent_actor_losses) == loss_window_size:
                                                mean_actor = np.mean(recent_actor_losses)
                                                std_actor = np.std(recent_actor_losses)
                                                lower_actor_threshold = mean_actor - 2 * std_actor
                                                if actor_loss < lower_actor_threshold:
                                                    print(f"Warning: Actor loss {actor_loss:.3f} below adaptive lower threshold "
                                                          f"{lower_actor_threshold:.3f} at episode {episode+1}")
                                                    # Optional: early_stop = True

                                        if done or early_stop:
                                            break

                                    episode_rewards.append(episode_reward)
                                    print(f"Episode {episode + 1}, Reward: {episode_reward:.3f}")

                                    if episode + 1 >= min_episodes_before_stop:
                                        avg_reward_so_far = np.mean(episode_rewards)
                                        if avg_reward_so_far < min_avg_reward_threshold:
                                            print(f"Early stopping: Average reward {avg_reward_so_far:.3f} below threshold at episode {episode+1}")
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

    # Load and save best model weights again if needed (optional)
    # You can reload the best model here if you saved checkpoints during tuning

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