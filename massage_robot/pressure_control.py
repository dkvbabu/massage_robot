# File: massage_robot/pressure_control.py
"""
ForceController with training utilities for DQN and PPO integrating path generation.
"""
import numpy as np
import gym
import pybullet as p
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from massage_robot.gym_wrapper import MassageGymEnv

class ForceController:
    """
    Low-level PID-based force controller.
    """
    def __init__(self, kp=1.0, ki=0.0, kd=0.1, target_force=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target_force
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_force, dt=1e-3):
        error = self.target - current_force
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        tau = self.kp * error + self.ki * self.integral + self.kd * derivative
        return tau

    def reward(self, current_force):
        return - (current_force - self.target) ** 2


def train_dqn(total_timesteps=500_000,
              eval_freq=10000,
              n_eval_episodes=5,
              env_kwargs=None,
              model_kwargs=None):
    """
    Train and tune a DQN agent on the MassageGymEnv with given parameters.
    :param total_timesteps: number of training timesteps
    :param eval_freq: frequency for evaluation
    :param n_eval_episodes: number of episodes per evaluation
    :param env_kwargs: dict for env init kwargs (e.g. use_visual)
    :param model_kwargs: dict for DQN hyperparameters
    :return: trained model
    """
    env_kwargs = env_kwargs or {}
    model_kwargs = model_kwargs or {}

    # Create environments
    def make_env(): return Monitor(gym.make('MassageEnv-v1', **env_kwargs))
    env = make_env()
    eval_env = monitor = Monitor(gym.make('MassageEnv-v1', **env_kwargs))

    # Eval callback
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='models/dqn_best/',
                                 log_path='logs/dqn_eval/',
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes)

    # Initialize model
    model = DQN(
        policy='MlpPolicy',
        env=env,
        tensorboard_log='./runs/dqn/',
        **model_kwargs
    )
    # Train
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    # Save final
    model.save('models/dqn_final')
    return model


def train_ppo(total_timesteps=1_000_000,
              eval_freq=20000,
              n_eval_episodes=5,
              num_envs=4,
              env_kwargs=None,
              model_kwargs=None):
    """
    Train and tune a PPO agent on the MassageGymEnv with path generation.
    :param total_timesteps: number of training timesteps
    :param eval_freq: frequency for evaluation
    :param n_eval_episodes: number of episodes per evaluation
    :param num_envs: number of parallel envs
    :param env_kwargs: dict for env init kwargs
    :param model_kwargs: dict for PPO hyperparameters
    :return: trained model
    """
    env_kwargs = env_kwargs or {}
    model_kwargs = model_kwargs or {}

    # Create vectorized environments
    def make_env(): return Monitor(gym.make('MassageEnv-v1', **env_kwargs))
    vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
    eval_env = DummyVecEnv([make_env])

    # Eval callback
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='models/ppo_best/',
                                 log_path='logs/ppo_eval/',
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes)

    # Initialize model
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        tensorboard_log='./runs/ppo/',
        **model_kwargs
    )
    # Train
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    # Save final
    model.save('models/ppo_final')
    return model

# Example usage
if __name__ == '__main__':
    # DQN tuning
    dqn_model = train_dqn(
        total_timesteps=300_000,
        model_kwargs={'learning_rate':1e-3,'buffer_size':50000,'batch_size':64,'gamma':0.99}
    )
    # PPO tuning
    ppo_model = train_ppo(
        total_timesteps=500_000,
        num_envs=8,
        model_kwargs={'learning_rate':3e-4,'n_steps':1024,'gamma':0.98,'ent_coef':0.01}
    )
    print("Training completed: DQN & PPO models ready.")
