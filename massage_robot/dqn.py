# File: massage_robot/dqn.py
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    # Create monitored environment
    env = Monitor(gym.make('MassageEnv-v1'))
    # Evaluation callback
    eval_env = Monitor(gym.make('MassageEnv-v1'))
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./models/',
                                 log_path='./logs/',
                                 eval_freq=10000,
                                 n_eval_episodes=5)
    # DQN hyperparameters tuning
    model = DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
    )
    # Train
    model.learn(total_timesteps=500_000, callback=eval_callback)
    # Save
    model.save('models/dqn_massage_best')
    print("DQN training complete.")
