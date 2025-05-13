# File: massage_robot/ppo.py
"""
PPO training script for the MassageRobot environment.
- Uses vectorized environments for faster rollouts.
- Integrates visual inputs if enabled in the Gym wrapper.
- Logs metrics and evaluation to TensorBoard and saves the best model.
"""
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(use_visual=False, visual_resolution=(84,84)):
    """Create a monitored environment with optional visual input."""
    def _init():
        env = gym.make('MassageEnv-v1', use_visual=use_visual, visual_resolution=visual_resolution)
        return Monitor(env)
    return _init


if __name__ == '__main__':
    # Configuration for PPO
    use_visual = True        # simulate camera input
    visual_res = (84, 84)    # resolution for visual obs
    num_envs = 4             # parallel environments

    # Create vectorized environments
    vec_env = DummyVecEnv([make_env(use_visual, visual_res) for _ in range(num_envs)])
    eval_env = DummyVecEnv([make_env(use_visual, visual_res)])

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/ppo_massage/',
        log_path='logs/ppo_eval/',
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Instantiate PPO model
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log='./runs/ppo_training/',
    )

    # Train the model
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )

    # Save the best and final models
    model.save('models/ppo_massage_final')
    print('PPO training complete. Best model saved under models/ppo_massage/')

