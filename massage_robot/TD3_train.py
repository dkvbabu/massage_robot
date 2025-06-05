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

class MassageEnv():
    def __init__(self, render=False):
        self.SimID = p.connect([p.DIRECT, p.GUI][render])
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -10)
        startPos = [-0.8, 0.1, 1.0]
        cubeStartingPose = [-1.3, 0.0, 0.5]
        self.EErot = p.getQuaternionFromEuler([0, 90, 0])
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        planeId = p.loadURDF("plane.urdf")
        cubeId = p.loadURDF("cube.urdf", cubeStartingPose, startOrientation)
        self.armId = p.loadURDF('urdf/ur5_robot.urdf', startPos, startOrientation)

        self.TimeStep = 1 / 24.0
        p.setTimeStep(self.TimeStep)

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.human_inst = load_scene(self.SimID)
        pos, orn = p.getBasePositionAndOrientation(self.human_inst.body)
        print(f"Human position: {pos}")
        print(f"Human orientation: {orn}")
        
        self.nArmJoints = p.getNumJoints(self.armId, physicsClientId=self.SimID)
        self.EndEfferctorId = self.nArmJoints - 3
        for link_idx in range(p.getNumJoints(self.armId)):
            if link_idx != self.EndEfferctorId:
                p.setCollisionFilterPair(self.armId, self.human_inst.body, link_idx, -1, enableCollision=0, physicsClientId=self.SimID)

        
        self.PointsInPath = 100
        self.controlledJoints = [1, 2, 3, 4, 5, 6]
        self.baseLevel = 0.02

        p.resetDebugVisualizerCamera(cameraYaw=92.4, cameraPitch=-41.8, cameraDistance=1.0,
                                     cameraTargetPosition=(-0.4828510, 0.12460, 0.6354567), )

        self.make_path()  # Generate path once here

        self.reset()
        self.timestep = 0

    def reset(self):
        # Reset human position and orientation to fixed values
        fixed_pos = (-0.15, 0.2, 0.95)
        fixed_orn = (4.329780281177466e-17, 0.7071067811865476, 0.7071067811865475, -4.329780281177466e-17)
        p.resetBasePositionAndOrientation(self.human_inst.body, fixed_pos, fixed_orn)

        # Reset robot joints to initial angles
        initial_joint_angles = [-0.4, -0.9, 1, -2.0, -1.5, 0.0]
        for joint_index, angle in zip(self.controlledJoints, initial_joint_angles):
            p.resetJointState(self.armId, joint_index, angle, targetVelocity=0.0)

        # Reset tracking variables
        self.Forces = []
        self.bodyparts = []
        self.armparts = []

        self.old_path = []
        self.new_path = []
        self.actual_path = []

        # Do NOT call make_path() here to keep path fixed

    def make_path(self):
        # Generate trajectory points along human back (x-axis)
        p1, p2 = p.getAABB(self.human_inst.body)
        pnts = generate_trajectory(np.array([p1[0], 0.3, p2[2] + self.baseLevel]),
                                   np.array([p2[0], 0.3, p2[2] + self.baseLevel]),
                                   numSamples=self.PointsInPath,
                                   frequency=6, amp=0.02)
        self.pntsAndReturn = np.vstack((pnts[::-1], pnts))

    def get_current_target(self):
        # Return current target point on path based on timestep
        return self.pntsAndReturn[(self.timestep % (2 * self.PointsInPath))]

    def get_action(self, change=0):
        # Heuristic action: target point on path
        return self.get_current_target()

    def step(self, action):
        
        JointPoses = list(p.calculateInverseKinematics(self.armId, self.EndEfferctorId, action, self.EErot))

        p.setJointMotorControlArray(self.armId, jointIndices=self.controlledJoints, controlMode=p.POSITION_CONTROL,
                                    targetPositions=[JointPoses[j - 1] for j in self.controlledJoints],
                                    forces=50 * np.ones_like(self.controlledJoints))
        
        # # Reset human position and orientation to fixed values
        # fixed_pos = (-0.15, 0.2, 0.95)
        # fixed_orn = (4.329780281177466e-17, 0.7071067811865476, 0.7071067811865475, -4.329780281177466e-17)
        # p.resetBasePositionAndOrientation(self.human_inst.body, fixed_pos, fixed_orn)


        p.stepSimulation(physicsClientId=self.SimID)

        self.collect_stats()

        self.timestep += 1

        if (self.timestep%(self.PointsInPath*2))==0:
            self.make_path()


    def collect_stats(self):
        out_1 = p.getContactPoints(self.armId, self.human_inst.body)

        if len(out_1):
            self.bodyparts.append(out_1[0][4])
            self.armparts.append(out_1[0][3])
            self.Forces.append(out_1[0][9])
        else:
            self.bodyparts.append(0)
            self.armparts.append(0)
            self.Forces.append(0)

        com_pose, com_orient, _, _, _, _ = p.getLinkState(self.armId, self.EndEfferctorId)
        self.actual_path.append(com_pose[-1])

        self.old_path.append(self.pntsAndReturn[(self.timestep % (2 * self.PointsInPath))][2])

    def close(self):
        p.disconnect()
        draw_data(self.Forces, self.armparts, self.bodyparts,
                  old_path=self.old_path, new_path=self.new_path, actual_path=self.actual_path)

    # Updated get_state method in MassageEnv class
    def get_state(self):
        # Raw joint angles
        joint_states = [p.getJointState(self.armId, j)[0] for j in self.controlledJoints]

        # Normalize joint angles assuming range ~[-pi, pi]
        normalized_joint_states = [angle / np.pi for angle in joint_states]

        # Raw end effector position
        ee_pos, _ = p.getLinkState(self.armId, self.EndEfferctorId)[:2]

        # Normalize end effector position assuming workspace ~[-1,1] meters
        normalized_ee_pos = [coord / 1.0 for coord in ee_pos]

        # Normalized timestep
        timestep_norm = self.timestep / 200.0

        # Current target position (assumed in same scale as ee_pos)
        target_pos = self.get_current_target()
        normalized_target_pos = [coord / 1.0 for coord in target_pos]

        # Contact info
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, physicsClientId=self.SimID)
        normal_forces = [cp[9] for cp in contact_points]
        normal_vectors = [cp[7] for cp in contact_points]
        total_normal_force = sum(normal_forces) if normal_forces else 0.0
        if normal_vectors:
            avg_normal_vector = np.mean(np.array(normal_vectors), axis=0)
        else:
            avg_normal_vector = np.array([0.0, 0.0, 0.0])
        num_contacts = len(contact_points)

        # Construct normalized state vector
        state = np.array(
            normalized_joint_states +
            normalized_ee_pos +
            [timestep_norm] +
            normalized_target_pos +
            [total_normal_force] +
            list(avg_normal_vector) +
            [num_contacts],
            dtype=np.float32
        )

        return state
        
    def get_reward(self):
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, physicsClientId=self.SimID)

        # Separate contacts by link index on the arm side
        end_effector_contacts = [cp for cp in contact_points if cp[3] == 7]  # linkIndexA == 7 is end effector
        other_contacts = [cp for cp in contact_points if cp[3] != 7]

        # Sum forces for end effector contacts only
        total_force = sum([cp[9] for cp in end_effector_contacts]) if end_effector_contacts else 0.0

        target_min, target_max = 15, 50

        reward_contact = 0.0
        reward_force = 0.0
        reward_penalty_force = 0.0
        reward_wrong_contact_penalty = 0.0
        reward_no_contact_penalty = 0.0
        
        # Penalize any contact from links other than end effector
        if other_contacts:
            reward_wrong_contact_penalty = -1.0  # Strong penalty for wrong contact

        if end_effector_contacts:
            reward_contact = 0.5

            if target_min <= total_force <= target_max:
                center = (target_min + target_max) / 2
                width = (target_max - target_min) / 2
                reward_force = 1.0 - abs(total_force - center) / width
            else:
                dist = min(abs(total_force - target_min), abs(total_force - target_max))
                reward_force = -dist / target_max * 0.5

            if total_force > target_max * 2:
                reward_penalty_force = -2.0
        else:
            reward_no_contact_penalty = -0.05

        # print('rewardcontact', reward_contact)
        # print('rewardforce', reward_force)
        # print('rewardpenaltyforce', reward_penalty_force)
        # print('rewardnocontact', reward_no_contact_penalty)
        # print('rewardwrongcontact', reward_wrong_contact_penalty)

        total_reward = (reward_contact + reward_force +
                        reward_penalty_force + reward_wrong_contact_penalty +
                        reward_no_contact_penalty)
        total_reward = max(min(total_reward, 1.0), -1.0)

        return total_reward

    
    def get_current_target(self):
        max_index = 2 * self.PointsInPath - 1
        idx = min(self.timestep, max_index)
        return self.pntsAndReturn[idx]
    
    def get_action_bounds(self, margin=0):
        min_bounds = np.min(self.pntsAndReturn, axis=0) - margin
        max_bounds = np.max(self.pntsAndReturn, axis=0) + margin
        # print('minbounds',min_bounds) 
        # print('maxbounds',max_bounds) 
        # Ensure x-axis minimum bound is not negative
        if min_bounds[0] < 0 and  min_bounds[0] < -0.3:
            min_bounds[0] = -0.3    
        
        if min_bounds[2] > 0 and  min_bounds[2] > 1.06:
            min_bounds[2] = 1.06              

        return min_bounds, max_bounds

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
        state = torch.FloatTensor(state.reshape(1, -1))
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

def train_td3():
    writer = SummaryWriter(log_dir="./runs/td3_training")
    # When initializing the agent, get the updated state_dim from get_state()
    env = MassageEnv(render=False)
    state_dim = len(env.get_state())  # This will now include the 5 new features
    action_dim = 3
    max_action = 1.0
    agent = TD3(state_dim, action_dim, max_action)

    # Best hyperparameters from tuning
    best_params = {
        'learning_rate': 0.001,
        'batch_size': 256,
        'discount': 0.999,
        'tau': 0.01,
        'policy_noise': 0.1,
        'noise_clip': 0.3,
        'policy_freq': 2
    }

    # Override default optimizers and parameters with best hyperparameters
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=best_params['learning_rate'])
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=best_params['learning_rate'])
    agent.discount = best_params['discount']
    agent.tau = best_params['tau']
    agent.policy_noise = best_params['policy_noise']
    agent.noise_clip = best_params['noise_clip']
    agent.policy_freq = best_params['policy_freq']

    replay_buffer = ReplayBuffer()

    episodes = 100
    episode_length = 1 * env.PointsInPath  # Full path length, no looping
    batch_size = best_params['batch_size']
    start_timesteps = 500  # Steps before training starts

    total_steps = 0

    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)

    episode_rewards = []
    actor_losses = []
    critic_losses = []

    for episode in range(episodes):
        env.reset()
        # Get midpoint of human back path
        mid_index = len(env.pntsAndReturn) // 2
        origin_pos = env.pntsAndReturn[mid_index]  # [x, y, z]

        axis_length = 0.2

        # Draw axes at origin_pos
        p.addUserDebugLine(origin_pos, origin_pos + [axis_length, 0, 0], [1, 0, 0], lineWidth=3)  # X axis red
        p.addUserDebugLine(origin_pos, origin_pos + [0, axis_length, 0], [0, 1, 0], lineWidth=3)  # Y axis green
        p.addUserDebugLine(origin_pos, origin_pos + [0, 0, axis_length], [0, 0, 1], lineWidth=3)  # Z axis blue

        state = env.get_state()
        episode_reward = 0

        min_bounds, max_bounds = env.get_action_bounds()
        #print("Action bounds:", min_bounds, max_bounds)

        for t in range(episode_length):
            action = agent.select_action(state)
            action = (action + np.random.normal(0, 0.1, size=action_dim))
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
                critic_losses.append(critic_loss)
                writer.add_scalar('Loss/Critic', critic_loss, total_steps)

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward:.3f}")
        writer.add_scalar('Reward/Episode', episode_reward, episode)

        # Save model every 50 episodes (adjust as needed)
        if (episode + 1) % 50 == 0:
            actor_path = os.path.join(save_dir, f"actor_episode_{episode+1}.pth")
            critic_path = os.path.join(save_dir, f"critic_episode_{episode+1}.pth")
            torch.save(agent.actor.state_dict(), actor_path)
            torch.save(agent.critic.state_dict(), critic_path)
            print(f"Saved models at episode {episode + 1}")

        # Plot every 100 episodes
        if (episode + 1) % 100 == 0:
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

            # Additional plot: average reward per every 10 episodes
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

    # Save final model
    torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_final.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic_final.pth"))
    print("Saved final models")
    
    writer.close()
    env.close()

if __name__ == "__main__":
    train_td3()