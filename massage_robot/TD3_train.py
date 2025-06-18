import pybullet as p
import time, os
import pybullet_data
import numpy as np
import cv2

from utils import get_extrinsics, get_intrinsics, get_extrinsics2

import configparser

from human.human_creation import HumanCreation
from human import agent, human
from human.agent import Agent
from human.human import Human
from human.furniture import Furniture

import matplotlib.pyplot as plt

from generate_path import generate_trajectory

from test_viewer import load_scene, draw_data

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
from env import MassageEnv  # import the environment from env.py

# Updated Actor class
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)  # state_dim now includes 5 extra features
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a * self.max_action


# Updated Critic class
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.storage = deque(maxlen=max_size)

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )


class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)  # shape (1, state_dim)
        elif isinstance(state, torch.Tensor):
            if state.dim() == 1:
                state = state.unsqueeze(0)
        else:
            raise TypeError("State must be numpy array or torch tensor")
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.discount * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item() if actor_loss is not None else None, critic_loss.item()


import matplotlib.pyplot as plt

def get_action_bounds(env, margin=0):
    min_bounds = np.min(env.pntsAndReturn, axis=0) - margin
    max_bounds = np.max(env.pntsAndReturn, axis=0) + margin
    print('minbounds',min_bounds) 
    print('maxbounds',max_bounds) 
    # Ensure x-axis minimum bound is not negative
    # if min_bounds[0] < 0 and  min_bounds[0] < -0.3:
    #     min_bounds[0] = -0.3    
    
    # if min_bounds[2] > 0 and  min_bounds[2] > 1.06:
    #     min_bounds[2] = 1.06              

    return min_bounds, max_bounds

def local_reset(env):
    # Reset human position and orientation to fixed values
    fixed_pos = (-0.15, 0.2, 0.95)
    fixed_orn = (4.329780281177466e-17, 0.7071067811865476, 0.7071067811865475, -4.329780281177466e-17)
    p.resetBasePositionAndOrientation(env.human_inst.body, fixed_pos, fixed_orn)

    # Reset robot joints to initial angles
    initial_joint_angles = [-0.4, -0.9, 1, -2.0, -1.5, 0.0]
    for joint_index, angle in zip(env.controlledJoints, initial_joint_angles):
        p.resetJointState(env.armId, joint_index, angle, targetVelocity=0.0)

    # Reset tracking variables
    env.Forces = []
    env.bodyparts = []
    env.armparts = []

    env.old_path = []
    env.new_path = []
    env.actual_path = []

    # Do NOT call make_path() here to keep path fixed

def local_step(env, action):
    max_z = 1.14  # Set your max z bound here (adjust as needed)
    
    # Clip the z-axis coordinate of the target position
    clipped_action = list(action)
    clipped_action[2] = min(clipped_action[2], max_z)
    
    JointPoses = list(p.calculateInverseKinematics(env.armId, env.EndEfferctorId, action, env.EErot))
    
    p.setJointMotorControlArray(env.armId, jointIndices=env.controlledJoints, controlMode=p.POSITION_CONTROL,
                                targetPositions=[JointPoses[j - 1] for j in env.controlledJoints],
                                forces=50 * np.ones_like(env.controlledJoints))
    
    p.stepSimulation(physicsClientId=env.SimID)
    
    env.collect_stats()
    
    env.timestep += 1
    
    if (env.timestep % (env.PointsInPath * 2)) == 0:
        env.make_path()    

def print_end_effector_contacts(env):
    contacts = p.getContactPoints(bodyA=env.armId, bodyB=env.human_inst.body,
                                  linkIndexA=env.EndEfferctorId, physicsClientId=env.SimID)
    if contacts:
        print(f"Contacts on end effector (link {env.EndEfferctorId}):")
        for c in contacts:
            contact_pos = c[5]  # contact position on robot link
            contact_force = c[9]  # normal force
            print(f"  Contact point: {contact_pos}, Force: {contact_force:.4f}")
    else:
        print("No contacts on end effector.")

def verify_collision_filters(env):
    print("Collision filter status (0=disabled, 1=enabled) between robot links and human:")
    for link_idx in range(p.getNumJoints(env.armId)):
        status = p.getCollisionFilterPair(env.armId, env.human_inst.body, link_idx, -1, physicsClientId=env.SimID)
        print(f"  Link {link_idx}: {'Enabled' if status else 'Disabled'}")

def check_all_link_contacts(env):
    print("Checking contacts between robot links and human:")
    for link_idx in range(p.getNumJoints(env.armId)):
        contacts = p.getContactPoints(bodyA=env.armId, bodyB=env.human_inst.body,
                                      linkIndexA=link_idx, physicsClientId=env.SimID)
        if contacts:
            print(f"  Contacts on link {link_idx}:")
            for c in contacts:
                contact_pos = c[5]
                contact_force = c[9]
                print(f"    Contact point: {contact_pos}, Force: {contact_force:.4f}")        

import numpy as np

def train_td3():
    writer = SummaryWriter(log_dir="./runs/td3_training")
    env = MassageEnv(render=True)  # Use env.py's MassageEnv

    # Disable collisions for all arm links except end effector
    for link_idx in range(p.getNumJoints(env.armId)):
        if link_idx not in [env.EndEfferctorId, 7]:
            p.setCollisionFilterPair(env.armId, env.human_inst.body, link_idx, -1, enableCollision=0, physicsClientId=env.SimID)

    check_all_link_contacts(env)
    stats = env.collect_stats()
    state_dim = len(env.get_state(stats))
    print(f"Updated state dimension: {state_dim}")

    action_dim = 3
    max_action = 1.0
    agent = TD3(state_dim, action_dim, max_action)

    best_params = {
        'learning_rate': 0.0003,
        'batch_size': 256,
        'discount': 0.99,
        'tau': 0.01,
        'policy_noise': 0.1,
        'noise_clip': 0.5,
        'policy_freq': 2
    }

    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=best_params['learning_rate'])
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=best_params['learning_rate'])
    agent.discount = best_params['discount']
    agent.tau = best_params['tau']
    agent.policy_noise = best_params['policy_noise']
    agent.noise_clip = best_params['noise_clip']
    agent.policy_freq = best_params['policy_freq']

    replay_buffer = ReplayBuffer()

    episodes = 1000
    episode_length = 1440
    #episode_length = 360
    batch_size = best_params['batch_size']
    start_timesteps = 300

    total_steps = 0
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)

    episode_rewards = []
    actor_losses = []
    critic_losses = []

    alpha = 0.01  # smoothing factor for interpolation
    prev_action = np.zeros(action_dim)
    smooth_target_prev = None

    all_Forces = []
    all_bodyparts = []
    all_armparts = []
    all_old_path = []
    all_new_path = []
    all_actual_path = []

    for episode in range(episodes):
        local_reset(env)
        env.make_path()
        stats = env.collect_stats()
        state = env.get_state(stats)
        print(f"Initial state length: {len(state)}")

        state = state / np.linalg.norm(state) if np.linalg.norm(state) > 0 else state

        min_bounds, max_bounds = get_action_bounds(env, margin=0)
        print(f"TD3 action bounds: min {min_bounds}, max {max_bounds}")

        episode_reward = 0

        for t in range(episode_length):
            # Disable collisions for all arm links except end effector
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

            ee_pos, _ = p.getLinkState(env.armId, env.EndEfferctorId, physicsClientId=env.SimID)[:2]
            #print(f"Step {t}: End Effector Position - x: {ee_pos[0]:.3f}, y: {ee_pos[1]:.3f}, z: {ee_pos[2]:.3f}")
            stats = env.collect_stats()
            
            next_state = env.get_state(stats)
            next_state = next_state / np.linalg.norm(next_state) if np.linalg.norm(next_state) > 0 else next_state
            stats = env.collect_stats()
            
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
                writer.add_scalar('Loss/Critic', critic_loss, total_steps)

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward:.3f}")
        writer.add_scalar('Reward/Episode', episode_reward, episode)

        # Print mean and variance every 100 episodes
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            mean_return = np.mean(recent_rewards)
            var_return = np.var(recent_rewards)
            print(f"\nEpisodic Return Stats for episodes {episode - 99} to {episode + 1}:")
            print(f"Mean: {mean_return:.2f}")
            print(f"Variance: {var_return:.2f}")
            print(f"Mean ± Variance: {mean_return:.2f} ± {var_return:.2f}\n")

        all_Forces.extend(env.Forces)
        all_bodyparts.extend(env.bodyparts)
        all_armparts.extend(env.armparts)
        all_old_path.extend(env.old_path)
        all_new_path.extend(env.new_path)
        all_actual_path.extend(env.actual_path)

        if (episode + 1) % 200 == 0:
            draw_data(
                all_Forces,
                all_armparts,
                all_bodyparts,
                old_path=all_old_path,
                new_path=all_new_path,
                actual_path=all_actual_path
            )
            all_Forces.clear()
            all_bodyparts.clear()
            all_armparts.clear()
            all_old_path.clear()
            all_new_path.clear()
            all_actual_path.clear()

        if (episode + 1) % 200 == 0:
            actor_path = os.path.join(save_dir, f"actor_episode_{episode+1}.pth")
            critic_path = os.path.join(save_dir, f"critic_episode_{episode+1}.pth")
            torch.save(agent.actor.state_dict(), actor_path)
            torch.save(agent.critic.state_dict(), critic_path)
            print(f"Saved models at episode {episode + 1}")

        if (episode + 1) % 200 == 0:
            plt.figure(figsize=(12,5))

            plt.subplot(1, 3, 1)
            plt.plot(episode_rewards, label='Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Reward')
            plt.grid()
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(actor_losses, label='Actor Loss', color='orange')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Actor Loss')
            plt.grid()
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(critic_losses, label='Critic Loss', color='green')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Critic Loss')
            plt.grid()
            plt.legend()

            plt.tight_layout()
            plt.show()

            chunk_size = 10
            avg_rewards = [
                np.mean(episode_rewards[i:i + chunk_size])
                for i in range(0, len(episode_rewards), chunk_size)
            ]
            episode_indices = [i + chunk_size // 2 for i in range(0, len(episode_rewards), chunk_size)]

            plt.figure(figsize=(8, 4))
            plt.plot(episode_indices, avg_rewards, label=f'Average Reward (every {chunk_size} episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.title(f'Average Reward Per Every {chunk_size} Episodes')
            plt.grid()
            plt.legend()
            plt.show()

    # Summary table at the end of training
    print("\nSummary of Mean ± Variance of Episodic Returns (last 100 episodes):")
    print(f"{'TD3 Noisy':<30} {'Mean':>10} {'Variance':>15} {'Mean ± Variance':>20}")
    print("-" * 80)

    recent_rewards = episode_rewards[-100:]
    mean_return = np.mean(recent_rewards)
    var_return = np.var(recent_rewards)
    print(f"{'TD3 (Noisy Env)':<30} {mean_return:10.2f} {var_return:15.2f} {mean_return:.2f} ± {var_return:.2f}")

    torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_final.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic_final.pth"))
    print("Saved final models")

    writer.close()
    env.close()
    
if __name__ == "__main__":
    train_td3()