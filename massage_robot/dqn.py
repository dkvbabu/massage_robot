import gym
from stable_baselines3 import DQN

if __name__ == "__main__":
    env = gym.make('MassageEnv-v0')
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("dqn_massage")
