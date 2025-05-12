import gym
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = gym.make('MassageEnv-v0')
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2_000_000)
    model.save("ppo_massage")
