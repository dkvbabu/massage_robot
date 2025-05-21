import pybullet as p
import time, os
import pybullet_data
import numpy as np
import configparser
import logging
import threading
import tkinter as tk
from tkinter import messagebox

from human.human_creation import HumanCreation
from human import agent, human
from human.agent import Agent
from human.human import Human
from human.furniture import Furniture

import matplotlib.pyplot as plt

from robot_descriptions import ur5_description
from generate_path import generate_trajectory

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

logging.basicConfig(filename='simulation_errors.log', level=logging.ERROR)


def draw_data(Forces, armparts, bodyparts):
    plt.subplot(311)
    plt.title(f'Massage Pressure: Mean {np.mean(Forces):.2f}')
    plt.plot(Forces)
    plt.subplot(312)
    plt.title('Arm Part')
    plt.plot(armparts)
    plt.subplot(313)
    plt.title('Body Part')
    plt.plot(bodyparts)
    plt.show()


def load_scene(physicsClient):
    human_creation = HumanCreation(physicsClient, np_random=np.random)
    human_controllable_joint_indices = []
    human_inst = Human(human_controllable_joint_indices, controllable=False)

    configP = configparser.ConfigParser()
    configP.read(os.path.join((os.path.dirname(os.path.realpath(__file__))), './human/config.ini'))

    def config(tag, section=None):
        return float(configP['' if section is None else section][tag])

    human_inst.init(human_creation, None, True, 'random', 'random', config=config, id=physicsClient, np_random=np.random)

    furniture = Furniture()
    furniture.init("bed", human_creation.directory, id=physicsClient, np_random=np.random, wheelchair_mounted=False)
    furniture.set_friction(furniture.base, friction=5)

    joints_positions = []
    human_inst.setup_joints2(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
    human_inst.set_mass(human_inst.base, mass=100)
    human_inst.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
    human_inst.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi / 2, -np.pi, 0])

    return human_inst


def generate_sine_trajectory(start, end, numSamples, frequency, amp):
    x_vals = np.linspace(start[0], end[0], numSamples)
    y_vals = np.linspace(start[1], end[1], numSamples)
    z_base = np.linspace(start[2], end[2], numSamples)
    z_vals = z_base + amp * np.sin(2 * np.pi * frequency * np.linspace(0, 1, numSamples))
    return np.vstack((x_vals, y_vals, z_vals)).T


def generate_circular_trajectory(center, radius, numSamples, frequency, amp):
    t = np.linspace(0, 2 * np.pi, numSamples)
    x_vals = center[0] + radius * np.cos(t)
    y_vals = center[1] + radius * np.sin(t)
    z_vals = center[2] + amp * np.sin(frequency * t)
    return np.vstack((x_vals, y_vals, z_vals)).T


def generate_linear_trajectory(start, end, numSamples, frequency=None, amp=None):
    x_vals = np.linspace(start[0], end[0], numSamples)
    y_vals = np.linspace(start[1], end[1], numSamples)
    z_vals = np.linspace(start[2], end[2], numSamples)
    return np.vstack((x_vals, y_vals, z_vals)).T


def generate_kneading_trajectory(center, radius, numSamples, frequency, amp):
    t = np.linspace(0, 4 * np.pi, numSamples)  # multiple circles
    x_vals = center[0] + radius * np.cos(t)
    y_vals = center[1] + radius * np.sin(t)
    z_vals = center[2] + amp * np.sin(frequency * t)
    return np.vstack((x_vals, y_vals, z_vals)).T


def generate_pressure_trajectory(center, numSamples):
    # Static point repeated for pressure
    pnts = np.tile(center, (numSamples, 1))
    return pnts


def update_trajectory(human_body, traj_step, frequency, amp, x_offset, region, z_offset_lower, z_offset_upper, traj_type, massage_technique):
    p1, p2 = p.getAABB(human_body)
    if region == 'lower_back':
        z_pos = p1[2] + z_offset_lower
        y_pos = 0.03
    elif region == 'upper_back':
        z_pos = p2[2] - z_offset_upper
        y_pos = 0.4
    else:
        z_pos = (p1[2] + p2[2]) / 2
        y_pos = 0.3  # default

    start = np.array([(p1[0] + p2[0]) / 2 - x_offset, y_pos, z_pos])
    end = np.array([(p1[0] + p2[0]) / 2 + x_offset, y_pos, z_pos])
    center = np.array([(p1[0] + p2[0]) / 2, y_pos, z_pos])

    if massage_technique == 'kneading':
        radius = x_offset / 2  # smaller radius for kneading
        pnts = generate_kneading_trajectory(center, radius, traj_step, frequency, amp)
    elif massage_technique == 'pressure':
        pnts = generate_pressure_trajectory(center, traj_step)
    else:
        # Default to previous trajectory types
        if traj_type == 'sine':
            pnts = generate_sine_trajectory(start, end, traj_step, frequency, amp)
        elif traj_type == 'circular':
            radius = x_offset
            pnts = generate_circular_trajectory(center, radius, traj_step, frequency, amp)
        elif traj_type == 'linear':
            pnts = generate_linear_trajectory(start, end, traj_step)
        else:
            pnts = generate_linear_trajectory(start, end, traj_step)

    return np.vstack((pnts, pnts[::-1]))


def test_settings_single(physicsClient, human_inst, armId, nArmJoints, traj_step,
                         frequency, amp, x_offset, z_offset_lower, z_offset_upper, region, force, traj_type, massage_technique):
    params = {'frequency': frequency, 'amp': amp, 'x_offset': x_offset,
              'z_offset_lower': z_offset_lower, 'z_offset_upper': z_offset_upper,
              'region': region, 'force': force, 'traj_type': traj_type, 'massage_technique': massage_technique}
    print(f"Testing params: {params}")
    pntsAndReturn = update_trajectory(human_inst.body, traj_step, frequency, amp,
                                      x_offset, region, z_offset_lower, z_offset_upper, traj_type, massage_technique)
    Forces, bodyparts, armparts = [], [], []

    for j in range(1200):
        try:
            p.stepSimulation(physicsClientId=physicsClient)
            out_1 = p.getContactPoints(armId, human_inst.body)

            if len(out_1):
                bodyparts.append(out_1[0][4])
                armparts.append(out_1[0][3])
                Forces.append(out_1[0][9])
            else:
                bodyparts.append(0)
                armparts.append(0)
                Forces.append(0)

            out = p.getClosestPoints(armId, human_inst.body, 3, 5)
            if out:
                pntsAndReturn[j % (2 * traj_step), 2] += 2 * (out[0][6][2] - 0.005)
                pntsAndReturn[j % (2 * traj_step), 2] /= 3

            JointPoses = list(p.calculateInverseKinematics(armId, nArmJoints - 2, pntsAndReturn[j % (2 * traj_step)]))
            p.setJointMotorControlArray(armId, jointIndices=range(1, nArmJoints - 3), controlMode=p.POSITION_CONTROL,
                                        targetPositions=JointPoses, forces=force * np.ones_like(JointPoses))
            time.sleep(1 / 24.0)

            if j % int(2 / (1 / 24.0)) == 0:
                pntsAndReturn = update_trajectory(human_inst.body, traj_step, frequency, amp,
                                                  x_offset, region, z_offset_lower, z_offset_upper, traj_type, massage_technique)

        except Exception as e:
            logging.error(f"Error at step {j} with params {params}: {e}")
            break

    return [{'params': params, 'Forces': Forces, 'bodyparts': bodyparts, 'armparts': armparts}]


def report_results(results):
    for res in results:
        params = res['params']
        mean_force = np.mean(res['Forces']) if res['Forces'] else 0
        print(f"Frequency: {params['frequency']}, Amplitude: {params['amp']}, X Offset: {params['x_offset']}, "
              f"Z Offset Lower: {params['z_offset_lower']}, Z Offset Upper: {params['z_offset_upper']}, "
              f"Region: {params['region']}, Force: {params['force']}, Trajectory: {params['traj_type']}, "
              f"Massage Technique: {params['massage_technique']}, Mean Force: {mean_force:.2f}")
    if results:
        last = results[-1]
        draw_data(last['Forces'], last['armparts'], last['bodyparts'])


def run_simulation_with_params(frequency, amplitude, x_offset, z_offset_lower, z_offset_upper, region, force, traj_type, massage_technique):
    physicsClient = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    startPos = [-0.7, 0.1, 1.0]
    cubeStartingPose = [-1.3, 0.0, 0.5]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    planeId = p.loadURDF("plane.urdf")
    armId = p.loadURDF(ur5_description.URDF_PATH, startPos, startOrientation)
    cubeId = p.loadURDF("cube.urdf", cubeStartingPose, startOrientation)

    TimeStep = 1 / 24.0
    p.setTimeStep(TimeStep)

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    human_inst = load_scene(physicsClient)
    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)
    traj_step = 100

    results = test_settings_single(physicsClient, human_inst, armId, nArmJoints, traj_step,
                                   frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                                   region, force, traj_type, massage_technique)
    report_results(results)

    p.disconnect()  # Ensure PyBullet is disconnected in the main simulation function

def start_simulation():
    try:
        freq = float(freq_entry.get())
        amp = float(amp_entry.get())
        x_offset = float(x_offset_entry.get())
        z_offset_lower = float(z_offset_lower_entry.get())
        z_offset_upper = float(z_offset_upper_entry.get())
        region = region_var.get()
        force = float(force_entry.get())
        traj_type = traj_type_var.get()
        massage_technique = massage_technique_var.get()
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numbers for all fields.")
        return

    start_button.config(state=tk.DISABLED)
    threading.Thread(target=lambda: [run_simulation_with_params(freq, amp, x_offset,
                                                               z_offset_lower, z_offset_upper, region, force, traj_type, massage_technique),
                                    start_button.config(state=tk.NORMAL)]).start()


# GUI setup
root = tk.Tk()
root.title("Simulation Parameters")

tk.Label(root, text="Frequency:").grid(row=0, column=0, padx=5, pady=5)
freq_entry = tk.Entry(root)
freq_entry.insert(0, "6")
freq_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Amplitude:").grid(row=1, column=0, padx=5, pady=5)
amp_entry = tk.Entry(root)
amp_entry.insert(0, "0.035")
amp_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="X Offset:").grid(row=2, column=0, padx=5, pady=5)
x_offset_entry = tk.Entry(root)
x_offset_entry.insert(0, "0.1")
x_offset_entry.grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Z Offset Lower:").grid(row=3, column=0, padx=5, pady=5)
z_offset_lower_entry = tk.Entry(root)
z_offset_lower_entry.insert(0, "0.01")
z_offset_lower_entry.grid(row=3, column=1, padx=5, pady=5)

tk.Label(root, text="Z Offset Upper:").grid(row=4, column=0, padx=5, pady=5)
z_offset_upper_entry = tk.Entry(root)
z_offset_upper_entry.insert(0, "0.1")
z_offset_upper_entry.grid(row=4, column=1, padx=5, pady=5)

region_var = tk.StringVar(value='lower_back')
tk.Label(root, text="Target Region:").grid(row=5, column=0, padx=5, pady=5)
region_menu = tk.OptionMenu(root, region_var, 'lower_back', 'upper_back')
region_menu.grid(row=5, column=1, padx=5, pady=5)

tk.Label(root, text="Force:").grid(row=6, column=0, padx=5, pady=5)
force_entry = tk.Entry(root)
force_entry.insert(0, "100")
force_entry.grid(row=6, column=1, padx=5, pady=5)

traj_type_var = tk.StringVar(value='sine')
tk.Label(root, text="Trajectory Type:").grid(row=7, column=0, padx=5, pady=5)
traj_type_menu = tk.OptionMenu(root, traj_type_var, 'sine', 'circular', 'linear')
traj_type_menu.grid(row=7, column=1, padx=5, pady=5)

massage_technique_var = tk.StringVar(value='normal')
tk.Label(root, text="Massage Technique:").grid(row=8, column=0, padx=5, pady=5)
massage_technique_menu = tk.OptionMenu(root, massage_technique_var, 'normal', 'kneading', 'pressure')
massage_technique_menu.grid(row=8, column=1, padx=5, pady=5)

start_button = tk.Button(root, text="Start Simulation", command=start_simulation)
start_button.grid(row=9, column=0, columnspan=2, pady=10)


# === RL Training additions start here ===

class MassageEnv:
    def __init__(self, physicsClient, armId, human_inst, nArmJoints, traj_step=100,
                 frequency=6, amp=0.035, x_offset=0.1, z_offset_lower=0.01, z_offset_upper=0.1,
                 region='lower_back', force_limit=100, traj_type='sine', massage_technique='normal'):
        self.physicsClient = physicsClient
        self.armId = armId
        self.human_inst = human_inst
        self.nArmJoints = nArmJoints
        self.traj_step = traj_step

        self.frequency = frequency
        self.amp = amp
        self.x_offset = x_offset
        self.z_offset_lower = z_offset_lower
        self.z_offset_upper = z_offset_upper
        self.region = region
        self.force_limit = force_limit
        self.traj_type = traj_type
        self.massage_technique = massage_technique

        self.joint_indices = list(range(1, nArmJoints - 3))
        self.delta_angle = 0.05  # radians per action step

        from itertools import product
        self.action_space = list(product([-self.delta_angle, 0, self.delta_angle], repeat=len(self.joint_indices)))
        self.action_size = len(self.action_space)

        self.current_step = 0
        self.max_steps = 300  # Reduced for faster debugging

        self.trajectory_points = self._update_trajectory()
        self.target_joint_positions = [0.0] * len(self.joint_indices)

        # For smoothness reward
        self.prev_joint_velocities = np.zeros(len(self.joint_indices))

    def reset(self):
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physicsClient)

        p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        startPos = [-0.7, 0.1, 1.0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.armId = p.loadURDF(ur5_description.URDF_PATH, startPos, startOrientation, physicsClientId=self.physicsClient)
        self.nArmJoints = p.getNumJoints(self.armId, physicsClientId=self.physicsClient)
        self.joint_indices = list(range(1, self.nArmJoints - 3))

        self.human_inst = load_scene(self.physicsClient)

        self.target_joint_positions = [0.0] * len(self.joint_indices)
        self.current_step = 0
        self.trajectory_points = self._update_trajectory()

        self.prev_joint_velocities = np.zeros(len(self.joint_indices))

        return self._get_state()

    def _update_trajectory(self):
        return update_trajectory(self.human_inst.body, self.traj_step, self.frequency, self.amp,
                                 self.x_offset, self.region, self.z_offset_lower, self.z_offset_upper,
                                 self.traj_type, self.massage_technique)

    def _get_state(self):
        joint_states = p.getJointStates(self.armId, self.joint_indices, physicsClientId=self.physicsClient)
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, physicsClientId=self.physicsClient)
        total_force = sum([cp[9] for cp in contact_points]) if contact_points else 0.0
        state = np.concatenate([joint_positions, joint_velocities, [total_force]])
        return state.astype(np.float32)

    def step(self, action_idx):
        increments = self.action_space[action_idx]
        self.target_joint_positions = np.clip(
            np.array(self.target_joint_positions) + np.array(increments),
            -np.pi, np.pi
        )

        traj_point = self.trajectory_points[self.current_step % len(self.trajectory_points)]
        joint_poses = p.calculateInverseKinematics(self.armId, self.nArmJoints - 2, traj_point)

        p.setJointMotorControlArray(
            self.armId,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_joint_positions,
            forces=self.force_limit * np.ones(len(self.joint_indices)),
            physicsClientId=self.physicsClient
        )

        p.stepSimulation(physicsClientId=self.physicsClient)
        # Removed time.sleep for faster training

        self.current_step += 1
        next_state = self._get_state()
        reward = self._compute_reward()
        done = self.current_step >= self.max_steps
        return next_state, reward, done, {}

    def _compute_reward(self):
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, physicsClientId=self.physicsClient)
        total_force = sum([cp[9] for cp in contact_points]) if contact_points else 0.0
        target_min, target_max = 20, 40  # target force range for good massage pressure

        reward = 0.0  # Initialize reward

        # Contact Reward
        if contact_points:
            reward += 0.1

        # Force Reward (shaped)
        if target_min <= total_force <= target_max:
            center = (target_min + target_max) / 2
            width = (target_max - target_min) / 2
            reward += 1.0 - abs(total_force - center) / width
        else:
            dist = min(abs(total_force - target_min), abs(total_force - target_max))
            reward -= dist / target_max

        # Safety penalty for excessive force
        if total_force > target_max * 2:
            reward -= 2.0

        # Smoothness penalty (jerky movements)
        joint_states = p.getJointStates(self.armId, self.joint_indices, physicsClientId=self.physicsClient)
        joint_velocities = np.array([state[1] for state in joint_states])
        velocity_change = np.sum(np.abs(joint_velocities - self.prev_joint_velocities))
        reward -= velocity_change * 0.01  # scale factor for smoothness penalty
        self.prev_joint_velocities = joint_velocities

        # Clip reward to [-1, 1]
        reward = max(min(reward, 1.0), -1.0)
        return reward

class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0          # Start fully exploring
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995  # Slower decay for longer exploration
        self.batch_size = 64
        self.learning_rate = 1e-3

        self.policy_net = self._build_model().to(self.device)
        self.target_net = self._build_model().to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.loss_history = []

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.policy_net(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor([float(m[4]) for m in minibatch]).to(self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_rl_training(frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                    region, force, traj_type, massage_technique, max_rl_episodes=100):
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    startPos = [-0.7, 0.1, 1.0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("plane.urdf", physicsClientId=physicsClient)
    armId = p.loadURDF(ur5_description.URDF_PATH, startPos, startOrientation, physicsClientId=physicsClient)
    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)
    human_inst = load_scene(physicsClient)

    env = MassageEnv(physicsClient, armId, human_inst, nArmJoints,
                     traj_step=100,
                     frequency=frequency,
                     amp=amplitude,
                     x_offset=x_offset,
                     z_offset_lower=z_offset_lower,
                     z_offset_upper=z_offset_upper,
                     region=region,
                     force_limit=force,
                     traj_type=traj_type,
                     massage_technique=massage_technique)

    state_size = len(env._get_state())
    action_size = env.action_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_size, action_size, device=device)

    target_update_freq = 5

    episode_rewards = []
    episode_losses = []

    for e in range(max_rl_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            step += 1

        if e % target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)
        avg_loss = sum(agent.loss_history) / len(agent.loss_history) if agent.loss_history else 0
        episode_losses.append(avg_loss)
        agent.loss_history.clear()

        print(f"RL Episode {e+1}/{max_rl_episodes} finished with total reward: {total_reward:.2f}, avg loss: {avg_loss:.4f}")

    p.disconnect()

    # Plot rewards and losses
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title("Episode Rewards")
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1,2,2)
    plt.title("Episode Average Loss")
    plt.plot(episode_losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()

def start_rl_training():
    try:
        freq = float(freq_entry.get())
        amp = float(amp_entry.get())
        x_offset = float(x_offset_entry.get())
        z_offset_lower = float(z_offset_lower_entry.get())
        z_offset_upper = float(z_offset_upper_entry.get())
        region = region_var.get()
        force = float(force_entry.get())
        traj_type = traj_type_var.get()
        massage_technique = massage_technique_var.get()
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numbers for all fields.")
        return

    start_button.config(state=tk.DISABLED)
    rl_button.config(state=tk.DISABLED)

    def run_rl():
        try:
            run_rl_training(freq, amp, x_offset, z_offset_lower, z_offset_upper,
                            region, force, traj_type, massage_technique, max_rl_episodes=100)
        finally:
            # Update GUI buttons safely on main thread
            root.after(0, lambda: [start_button.config(state=tk.NORMAL), rl_button.config(state=tk.NORMAL)])
            # Optional: close GUI after training finishes
            root.after(0, root.destroy)

    threading.Thread(target=run_rl, daemon=True).start()

def start_rl_training_gui():
    try:
        freq = float(freq_entry.get())
        amp = float(amp_entry.get())
        x_offset = float(x_offset_entry.get())
        z_offset_lower = float(z_offset_lower_entry.get())
        z_offset_upper = float(z_offset_upper_entry.get())
        region = region_var.get()
        force = float(force_entry.get())
        traj_type = traj_type_var.get()
        massage_technique = massage_technique_var.get()
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numbers for all fields.")
        return

    start_button.config(state=tk.DISABLED)
    rl_button.config(state=tk.DISABLED)

    def run_rl():
        run_rl_training(freq, amp, x_offset, z_offset_lower, z_offset_upper,
                        region, force, traj_type, massage_technique, max_rl_episodes=100)
        root.after(0, lambda: [start_button.config(state=tk.NORMAL), rl_button.config(state=tk.NORMAL)])

    threading.Thread(target=run_rl, daemon=True).start()

import gym
from torch.distributions import Categorical

# === PPO Agent Implementation ===

class PPOAgent(nn.Module):
    def __init__(self, state_size, action_size, device='cpu', hidden_size=128, lr=3e-4, gamma=0.99, eps_clip=0.2):
        super(PPOAgent, self).__init__()
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

def run_ppo_training(frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                     region, force, traj_type, massage_technique, max_episodes=100, update_timestep=2000):
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    startPos = [-0.7, 0.1, 1.0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("plane.urdf", physicsClientId=physicsClient)
    armId = p.loadURDF(ur5_description.URDF_PATH, startPos, startOrientation, physicsClientId=physicsClient)
    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)
    human_inst = load_scene(physicsClient)

    env = MassageEnv(physicsClient, armId, human_inst, nArmJoints,
                     traj_step=100,
                     frequency=frequency,
                     amp=amplitude,
                     x_offset=x_offset,
                     z_offset_lower=z_offset_lower,
                     z_offset_upper=z_offset_upper,
                     region=region,
                     force_limit=force,
                     traj_type=traj_type,
                     massage_technique=massage_technique)

    state_size = len(env._get_state())
    action_size = env.action_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_agent = PPOAgent(state_size, action_size, device=device)
    ppo_agent.to(device)

    timestep = 0
    memory = {
        'states': [],
        'actions': [],
        'logprobs': [],
        'rewards': [],
        'is_terminals': []
    }

    def compute_returns(rewards, gamma):
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        return (returns - returns.mean()) / (returns.std() + 1e-5)

    max_timesteps = max_episodes * env.max_steps
    episode_rewards = []
    episode_reward = 0
    state = env.reset()

    for t in range(max_timesteps):
        timestep += 1
        state_tensor = torch.FloatTensor(state).to(device)
        action, logprob = ppo_agent.act(state)

        next_state, reward, done, _ = env.step(action)

        memory['states'].append(state_tensor)
        memory['actions'].append(torch.tensor(action).to(device))
        memory['logprobs'].append(logprob)
        memory['rewards'].append(reward)
        memory['is_terminals'].append(done)

        state = next_state
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            state = env.reset()

        # Update PPO agent
        if timestep % update_timestep == 0:
            # Convert lists to tensors
            states = torch.stack(memory['states'])
            actions = torch.stack(memory['actions'])
            old_logprobs = torch.stack(memory['logprobs'])
            rewards = memory['rewards']
            is_terminals = memory['is_terminals']

            returns = compute_returns(rewards, ppo_agent.gamma)

            # Optimize policy for K epochs
            K_epochs = 4
            for _ in range(K_epochs):
                logprobs, state_values, dist_entropy = ppo_agent.evaluate(states, actions)

                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = returns - state_values.detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - ppo_agent.eps_clip, 1 + ppo_agent.eps_clip) * advantages

                loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(state_values, returns) - 0.01 * dist_entropy.mean()

                ppo_agent.optimizer.zero_grad()
                loss.backward()
                ppo_agent.optimizer.step()

            # Clear memory
            memory = {k: [] for k in memory}
            print(f"PPO Update at timestep {t}, average reward: {np.mean(episode_rewards[-10:]) if episode_rewards else 0:.2f}")

    p.disconnect()

    # Plot rewards
    plt.figure()
    plt.title("PPO Episode Rewards")
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

def start_ppo_training_gui():
    try:
        freq = float(freq_entry.get())
        amp = float(amp_entry.get())
        x_offset = float(x_offset_entry.get())
        z_offset_lower = float(z_offset_lower_entry.get())
        z_offset_upper = float(z_offset_upper_entry.get())
        region = region_var.get()
        force = float(force_entry.get())
        traj_type = traj_type_var.get()
        massage_technique = massage_technique_var.get()
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numbers for all fields.")
        return

    start_button.config(state=tk.DISABLED)
    rl_button.config(state=tk.DISABLED)
    ppo_button.config(state=tk.DISABLED)

    def run_ppo():
        try:
            run_ppo_training(freq, amp, x_offset, z_offset_lower, z_offset_upper,
                             region, force, traj_type, massage_technique, max_episodes=100)
        finally:
            root.after(0, lambda: [start_button.config(state=tk.NORMAL),
                                  rl_button.config(state=tk.NORMAL),
                                  ppo_button.config(state=tk.NORMAL)])
            root.after(0, lambda: [start_button.config(state=tk.NORMAL), ppo_button.config(state=tk.NORMAL)])

    threading.Thread(target=run_ppo, daemon=True).start()

rl_button = tk.Button(root, text="Start RL Training(DQN)", command=start_rl_training_gui)
rl_button.grid(row=10, column=0, columnspan=2, pady=10)

ppo_button = tk.Button(root, text="Start PPO Training", command=start_ppo_training_gui)
ppo_button.grid(row=11, column=0, columnspan=2, pady=10)

root.mainloop()
