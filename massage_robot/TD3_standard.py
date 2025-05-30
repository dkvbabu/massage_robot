import os
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Queue

import torch
import pybullet as p
import pybullet_data
import gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from massage_env import MassageEnvGym  # Import your environment from separate module


# --- Stable Baselines3 callback for saving best model ---

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            env = self.training_env
            if hasattr(env, 'envs'):
                env = env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'get_episode_rewards'):
                episode_rewards = env.get_episode_rewards()
            else:
                episode_rewards = []
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True


# --- Reward logging callback with TensorBoard support ---

class RewardLoggingCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_rewards = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        self._current_rewards += self.locals.get('rewards', [0])[0] if 'rewards' in self.locals else 0
        self._current_length += 1

        done = self.locals.get('dones', [False])[0] if 'dones' in self.locals else False
        if done:
            self.episode_rewards.append(self._current_rewards)
            self.episode_lengths.append(self._current_length)
            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)} reward: {self._current_rewards:.3f}, length: {self._current_length}")
            # Log to TensorBoard
            self.logger.record('episode_reward', self._current_rewards)
            self.logger.record('episode_length', self._current_length)
            self.logger.dump(self.num_timesteps)
            self._current_rewards = 0.0
            self._current_length = 0
        return True

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, label='Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Episode Rewards')
            plt.legend()
            plt.grid(True)
            os.makedirs(self.log_dir, exist_ok=True)
            plt.savefig(os.path.join(self.log_dir, 'training_rewards.png'))
            plt.show()


# --- Info print callback (optional) ---

class InfoPrintCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        if infos:
            pass
            # print("Step info:", infos[0])  # Uncomment to debug step info
        return True


# --- Training function ---

import os
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList

def run_td3_training(env_params, hyperparams=None, total_timesteps=50000, log_dir='td3_logs'):
    os.makedirs(log_dir, exist_ok=True)

    # Create environment and wrap with Monitor and DummyVecEnv
    env = MassageEnvGym(**env_params)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Apply observation normalization if requested
    if hyperparams and hyperparams.get('obs_norm', False):
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    n_actions = env.action_space.shape[-1]

    # Action noise sigma from hyperparams or default
    noise_sigma = hyperparams.get('policy_noise', 0.05) if hyperparams else 0.05
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma * np.ones(n_actions))

    # Policy network architecture
    hidden_layers = hyperparams.get('hidden_layers', [256, 256]) if hyperparams else [256, 256]
    policy_kwargs = dict(net_arch=hidden_layers)

    # Device selection: use GPU if available and specified
    device = hyperparams.get('device', 'cpu') if hyperparams else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = 'cpu'

    # Learning rates
    actor_lr = hyperparams.get('actor_lr', 1e-3) if hyperparams else 1e-3
    critic_lr = hyperparams.get('critic_lr', 1e-3) if hyperparams else 1e-3

    # Create TD3 model
    model = TD3("MlpPolicy", env,
                action_noise=action_noise,
                verbose=1,
                tensorboard_log=log_dir,
                batch_size=hyperparams.get('batch_size', 128) if hyperparams else 128,
                gamma=hyperparams.get('gamma', 0.99) if hyperparams else 0.99,
                tau=hyperparams.get('tau', 0.005) if hyperparams else 0.005,
                learning_rate=actor_lr,
                policy_kwargs=policy_kwargs,
                device=device)

    # Setup callbacks
    reward_callback = RewardLoggingCallback(check_freq=1000, log_dir=log_dir)
    best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    info_print_callback = InfoPrintCallback()
    callback = CallbackList([reward_callback, best_model_callback, info_print_callback])

    import traceback

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except Exception:
        print("Error during training:")
        print(traceback.format_exc())
        raise
    finally:
        # Save VecNormalize statistics if used
        if hyperparams and hyperparams.get('obs_norm', False):
            env.save(os.path.join(log_dir, 'vecnormalize.pkl'))
        env.close()

    model.save(os.path.join(log_dir, 'final_model'))

    print(f"Training completed. Model saved to {os.path.join(log_dir, 'final_model.zip')}")

    return model

# --- Inference function with plots ---

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def draw_data(Forces, armparts, bodyparts):
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.title(f'Massage Pressure: Mean {np.mean(Forces):.2f}')
    plt.plot(Forces)
    plt.subplot(312)
    plt.title('Arm Part')
    plt.plot(armparts)
    plt.subplot(313)
    plt.title('Body Part')
    plt.plot(bodyparts)
    plt.tight_layout()
    plt.show()

def run_td3_inference_with_plots(env_params, model_path, render=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    env_params['render'] = render
    env = MassageEnvGym(**env_params)
    model = TD3.load(model_path, env=env)

    episode_rewards = []
    reward_components_history = []

    # Initialize lists to collect Forces, armparts, bodyparts for all episodes
    all_Forces = []
    all_armparts = []
    all_bodyparts = []

    max_episodes = 10
    max_steps_per_episode = env.max_steps

    try:
        for episode in range(max_episodes):
            obs, info = env.reset()  # Unpack 2 values from reset
            done = False
            episode_reward = 0
            episode_reward_components = []
            print('Episode ', episode)

            # Per episode data lists
            Forces = []
            armparts = []
            bodyparts = []

            for step in range(max_steps_per_episode):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)  # Unpack 5 values
                done = terminated or truncated
                episode_reward += reward
                episode_reward_components.append(info)

                # Collect Forces, armparts, bodyparts if present in info
                Forces.append(info.get('Forces', 0))
                armparts.append(info.get('armparts', 0))
                bodyparts.append(info.get('bodyparts', 0))

                if done:
                    break

            episode_rewards.append(episode_reward)
            reward_components_history.append(episode_reward_components)

            # Append episode data to all episodes lists
            all_Forces.extend(Forces)
            all_armparts.extend(armparts)
            all_bodyparts.extend(bodyparts)

    finally:
        env.close()

    avg_rewards_per_episode = [np.mean([step.get('total_reward', 0) for step in ep]) for ep in reward_components_history]

    # Existing plotting code for rewards, histograms, boxplots, etc.
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('TD3 Inference Episode Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig('td3_inference_episode_rewards.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards_per_episode, label='Avg Step Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Step Reward')
    plt.title('Average Step Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('td3_inference_avg_step_reward.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(episode_rewards, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Total Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Episode Rewards')
    plt.grid(True)
    plt.savefig('td3_inference_reward_distribution.png')
    plt.show()
    plt.close()

    cumulative_rewards = np.cumsum(episode_rewards)
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_rewards, marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Episodes')
    plt.grid(True)
    plt.savefig('td3_inference_cumulative_reward.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[[step.get('total_reward', 0) for step in ep] for ep in reward_components_history])
    plt.xlabel('Episode')
    plt.ylabel('Step Reward')
    plt.title('Boxplot of Step Rewards per Episode')
    plt.grid(True)
    plt.savefig('td3_inference_step_rewards_boxplot.png')
    plt.show()
    plt.close()

    # Flatten or aggregate Forces, armparts, bodyparts to 1D lists of scalars
    flat_Forces = [np.sum(f) if isinstance(f, (list, np.ndarray)) else f for f in all_Forces]
    flat_armparts = [np.mean(a) if isinstance(a, (list, np.ndarray)) else a for a in all_armparts]
    flat_bodyparts = [np.mean(b) if isinstance(b, (list, np.ndarray)) else b for b in all_bodyparts]

    # Call the new idraw_data function to plot Forces, armparts, bodyparts
    draw_data(flat_Forces, flat_armparts, flat_bodyparts)

    print("Inference plots saved.")

    #print('Total Rewards: ', episode_rewards)

    return episode_rewards

# --- Multiprocessing processes for training, tuning, inference ---

import traceback

def training_process(params, queue):
    try:
        print("Training process started")
        model = run_td3_training(params['env_params'], params.get('hyperparams'), params.get('total_timesteps', 50000), params.get('log_dir', 'td3_logs'))
        queue.put("Training completed")
    except Exception as e:
        tb_str = traceback.format_exc()
        queue.put(f"Training error:\n{tb_str}")


def tuning_process(env_params, queue):
    try:
        env_params['render'] = env_params.get('render', False)
        best_params, best_reward = random_search_td3(env_params)
        queue.put(f"Tuning completed. Best reward: {best_reward:.3f}")
        queue.put(f"Best params: {best_params}")
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        queue.put(f"Tuning error: {e}\n{tb_str}")


def inference_process(params, queue):
    try:
        render_flag = params.get('render', False)
        params['env_params']['render'] = render_flag
        try:
            total_reward = run_td3_inference_with_plots(params['env_params'], params['model_path'], render=render_flag)
        except Exception as e:            
            import traceback
            tb_str = traceback.format_exc()
            print(f"Error during inference:{e}\n{tb_str}")            
            raise
        #total_reward = run_td3_inference_with_plots(params['env_params'], params['model_path'], render=render_flag)
        queue.put(f"Inference completed")
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        queue.put(f"Inference error: {e}\n{tb_str}")


# --- GUI Control Panel ---

class TD3ControlApp:
    def __init__(self, root):
        self.root = root
        root.title("TD3 Control Panel")

        tk.Label(root, text="Frequency").grid(row=0, column=0, sticky="w")

        self.freq_var = tk.DoubleVar(master=root, value=3.0)
        self.freq_value_label = tk.Label(root, text=f"{self.freq_var.get():.2f}")
        self.freq_value_label.grid(row=0, column=2, sticky="w")

        def on_freq_change(event):
            self.freq_value_label.config(text=f"{self.freq_var.get():.2f}")

        freq_slider = ttk.Scale(root, from_=0.1, to=10.0, variable=self.freq_var, orient='horizontal', command=lambda e: on_freq_change(e))
        freq_slider.grid(row=0, column=1, sticky="ew")


        tk.Label(root, text="Amplitude").grid(row=1, column=0, sticky="w")

        self.amp_var = tk.DoubleVar(master=root, value=0.01)
        self.amp_value_label = tk.Label(root, text=f"{self.amp_var.get():.3f}")
        self.amp_value_label.grid(row=1, column=2, sticky="w")

        def on_amp_change(event):
            self.amp_value_label.config(text=f"{self.amp_var.get():.3f}")

        amp_slider = ttk.Scale(root, from_=0.0, to=0.05, variable=self.amp_var, orient='horizontal', command=lambda e: on_amp_change(e))
        amp_slider.grid(row=1, column=1, sticky="ew")
        self.render_var = tk.BooleanVar(master=root, value=False)
        tk.Checkbutton(root, text="Render Training (GUI)", variable=self.render_var).grid(row=2, column=0, columnspan=2)

        self.status_var = tk.StringVar(master=root, value="Idle")
        tk.Label(root, textvariable=self.status_var).grid(row=5, column=0, columnspan=2)

        ttk.Button(root, text="Start Training", command=self.start_training).grid(row=3, column=0, pady=10)
        ttk.Button(root, text="Start Inference", command=self.start_inference).grid(row=3, column=1, pady=10)
        ttk.Button(root, text="Start Hyperparam Tuning", command=self.start_tuning).grid(row=4, column=0, columnspan=2, pady=10)

        root.columnconfigure(1, weight=1)

        self.process = None
        self.queue = Queue()
        self.root.after(100, self.check_queue)

    def get_params(self):
        return {
            'traj_step': 100,
            'frequency': self.freq_var.get(),
            'amp': self.amp_var.get(),
            'x_offset': 0.1,
            'z_offset_lower': 0.01,
            'z_offset_upper': 0.1,
            'region': 'lower_back',
            'force_limit': 30,
            'traj_type': 'sine',
            'massage_technique': 'normal',
            'max_steps': 300,
            'render': self.render_var.get()
        }

    def start_training(self):
        print("Start training button clicked")
        if self.process is None or not self.process.is_alive():
            self.status_var.set("Training started...")
            log_dir = 'td3_logs'
            params = {
                'env_params': self.get_params(),
                'total_timesteps': 50000,
                'log_dir': log_dir
            }
            self.process = Process(target=training_process, args=(params, self.queue))
            self.process.start()
            print("Training process started")

            # Launch TensorBoard automatically
            try:
                self.tb_process = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", log_dir])
                print("TensorBoard started")
            except Exception as e:
                print(f"Failed to start TensorBoard: {e}")
        else:
            self.status_var.set("Process already running")
            print("Process already running")

    def start_inference(self):
        if self.process is None or not self.process.is_alive():
            self.status_var.set("Inference started...")
            params = self.get_params()
            params = {'env_params': params, 'model_path': 'td3_logs/final_model.zip', 'render': params.get('render', False)}
            self.process = Process(target=inference_process, args=(params, self.queue))
            self.process.start()
        else:
            self.status_var.set("Process already running")

    def start_tuning(self):
        if self.process is None or not self.process.is_alive():
            self.status_var.set("Hyperparameter tuning started...")
            env_params = self.get_params()
            self.process = Process(target=tuning_process, args=(env_params, self.queue))
            self.process.start()
        else:
            self.status_var.set("Process already running")

    def check_queue(self):
        while not self.queue.empty():
            msg = self.queue.get()
            self.status_var.set(str(msg))
        self.root.after(100, self.check_queue)


# --- Hyperparameter tuning function ---

def evaluate_model(env, model, num_episodes=5):
    episode_rewards = []
    for _ in range(num_episodes):
        obs, info = env.reset()  # Unpack 2 values
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)  # Unpack 5 values
            done = terminated or truncated
            total_reward += reward
        episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards)
    return avg_reward, episode_rewards

import csv
import os

import os
import csv
import random
import matplotlib.pyplot as plt

def random_search_td3(env_params, search_iters=10, total_timesteps=5000, reward_threshold=None):
    best_reward = -float('inf')
    best_params = None
    rewards_per_trial = []

    # Prepare CSV log file
    log_file = 'td3_hyperparam_tuning_log.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial', 'actor_lr', 'critic_lr', 'gamma', 'tau', 'policy_noise', 'noise_clip',
                             'policy_freq', 'batch_size', 'hidden_layers', 'obs_norm', 'avg_reward'])

    for i in range(search_iters):
        # Sample hyperparameters
        actor_lr = 10 ** random.uniform(-5, -3)
        critic_lr = 10 ** random.uniform(-5, -3)
        gamma = random.uniform(0.9, 0.999)
        tau = random.uniform(0.001, 0.01)
        policy_noise = random.uniform(0.05, 0.3)  # wider range
        noise_clip = random.uniform(0.1, 0.7)
        policy_freq = random.choice([1, 2, 3, 4])
        batch_size = random.choice([64, 100, 128, 256])
        # Hidden layers: choose number of layers and units per layer
        num_layers = random.choice([2, 3])
        units_per_layer = random.choice([128, 256, 512])
        hidden_layers = [units_per_layer] * num_layers
        obs_norm = random.choice([True, False])  # whether to normalize observations

        hyperparams = {
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'gamma': gamma,
            'tau': tau,
            'policy_noise': policy_noise,
            'noise_clip': noise_clip,
            'policy_freq': policy_freq,
            'batch_size': batch_size,
            'hidden_layers': hidden_layers,
            'obs_norm': obs_norm
        }

        print(f"TD3 Trial {i+1}/{search_iters} with params: {hyperparams}")

        try:
            # Do NOT add obs_norm to env_params
            env_params_tuned = env_params.copy()

            # Run training with these hyperparameters
            model = run_td3_training(env_params_tuned, hyperparams, total_timesteps=total_timesteps,
                                    log_dir=f'td3_tuning_trial_{i+1}')

            # Evaluate model
            env = MassageEnvGym(**env_params)
            avg_reward, _ = evaluate_model(env, model, num_episodes=5)
            env.close()

            print(f"Average reward for trial {i+1}: {avg_reward:.3f}")
            rewards_per_trial.append(avg_reward)

            # Log to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i+1, actor_lr, critic_lr, gamma, tau, policy_noise, noise_clip,
                                 policy_freq, batch_size, hidden_layers, obs_norm, avg_reward])

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_params = hyperparams

            # Early stopping if reward threshold reached
            if reward_threshold is not None and avg_reward >= reward_threshold:
                print(f"Early stopping: reached reward threshold {reward_threshold}")
                break

        except Exception as e:
            print(f"Trial {i+1} failed with error: {e}")
            rewards_per_trial.append(float('-inf'))  # mark failed trial

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards_per_trial) + 1), rewards_per_trial, marker='o')
    plt.xlabel('Trial')
    plt.ylabel('Average Reward')
    plt.title('Hyperparameter Tuning Results')
    plt.grid(True)
    plt.savefig('tuning_results.png')
    plt.show()

    print(f"Hyperparameter tuning completed. Best hyperparameters: {best_params} with reward {best_reward:.3f}")
    return best_params, best_reward

if __name__ == "__main__":
    root = tk.Tk()
    app = TD3ControlApp(root)
    root.mainloop()
