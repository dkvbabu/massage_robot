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

import gym
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter
import datetime

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
start_button.grid(row=9, column=5, columnspan=2, pady=10)


# === RL Training additions start here ===
def visualize_contact(armId, human_body, physicsClient):
    contact_points = p.getContactPoints(bodyA=armId, bodyB=human_body, physicsClientId=physicsClient)
    if contact_points:
        contacting_link_index = contact_points[0][3]
        p.changeVisualShape(objectUniqueId=armId, linkIndex=contacting_link_index, rgbaColor=[1, 0, 0, 1], physicsClientId=physicsClient)
    else:
        for link_index in range(-1, p.getNumJoints(armId, physicsClientId=physicsClient)):
            p.changeVisualShape(objectUniqueId=armId, linkIndex=link_index, rgbaColor=[1, 1, 1, 1], physicsClientId=physicsClient)

def visualize_contact_points(armId, human_body, physicsClient):
    contact_points = p.getContactPoints(bodyA=armId, bodyB=human_body, physicsClientId=physicsClient)
    for point in contact_points:
        position_on_A = point[5]
        position_on_B = point[6]
        p.addUserDebugLine(lineFromXYZ=position_on_A, lineToXYZ=position_on_B, lineColorRGB=[1, 0, 0], lifeTime=0.1, physicsClientId=physicsClient)

def visualize_contact_force(armId, human_body, physicsClient):
    contact_points = p.getContactPoints(bodyA=armId, bodyB=human_body, physicsClientId=physicsClient)
    if contact_points:
        contacting_link_index = contact_points[0][3]
        contact_force = contact_points[0][9]
        max_force = 100
        intensity = min(contact_force / max_force, 1.0)
        color = [1, 1 - intensity, 1 - intensity, 1]
        p.changeVisualShape(objectUniqueId=armId, linkIndex=contacting_link_index, rgbaColor=color, physicsClientId=physicsClient)
    else:
        for link_index in range(-1, p.getNumJoints(armId, physicsClientId=physicsClient)):
            p.changeVisualShape(objectUniqueId=armId, linkIndex=link_index, rgbaColor=[1, 1, 1, 1], physicsClientId=physicsClient)

def random_search_dqn(env_params, search_iters=10):
    best_reward = -float('inf')
    best_params = None
    for i in range(search_iters):
        hyperparams = {
            'gamma': random.uniform(0.9, 0.999),
            'epsilon': 1.0,
            'epsilon_min': random.uniform(0.01, 0.1),
            'epsilon_decay': random.uniform(0.95, 0.999),
            'batch_size': random.choice([64, 128, 256]),
            'learning_rate': 10 ** random.uniform(-5, -3),
            'target_update_freq': random.choice([5, 10, 20])
        }
        print(f"DQN Trial {i+1}/{search_iters} with params: {hyperparams}")
        rewards = run_rl_training(**env_params, max_rl_episodes=50, hyperparams=hyperparams, load_model=False)
        avg_reward = np.mean(rewards[-10:])
        print(f"Avg reward last 10 episodes: {avg_reward:.2f}")
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = hyperparams
            print(f"New best DQN params with avg reward {best_reward:.2f}")
    print("Best DQN hyperparameters found:", best_params)

def random_search_ppo(env_params, search_iters=10):
    best_reward = -float('inf')
    best_params = None
    for i in range(search_iters):
        hyperparams = {
            'hidden_size': random.choice([128, 256, 512]),
            'learning_rate': 10 ** random.uniform(-5, -3),
            'gamma': random.uniform(0.9, 0.999),
            'eps_clip': random.uniform(0.1, 0.3),
            'K_epochs': random.choice([3, 4, 5, 6])
        }
        print(f"PPO Trial {i+1}/{search_iters} with params: {hyperparams}")
        rewards = run_ppo_training(**env_params, max_episodes=50, hyperparams=hyperparams, load_model=False)
        avg_reward = np.mean(rewards[-10:])
        print(f"Avg reward last 10 episodes: {avg_reward:.2f}")
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = hyperparams
            print(f"New best PPO params with avg reward {best_reward:.2f}")
    print("Best PPO hyperparameters found:", best_params)

def random_search_td3(env_params, search_iters=10):
    best_reward = -float('inf')
    best_params = None
    for i in range(search_iters):
        hyperparams = {
            'actor_lr': 10 ** random.uniform(-5, -3),
            'critic_lr': 10 ** random.uniform(-5, -3),
            'gamma': random.uniform(0.9, 0.999),
            'tau': random.uniform(0.001, 0.01),
            'policy_noise': random.uniform(0.1, 0.3),
            'noise_clip': random.uniform(0.3, 0.7),
            'policy_freq': random.choice([1, 2, 3]),
            'batch_size': random.choice([64, 100, 128]),
            'hidden_size': random.choice([128, 256, 512])
        }
        print(f"TD3 Trial {i+1}/{search_iters} with params: {hyperparams}")
        rewards = run_td3_training(**env_params, max_episodes=50, hyperparams=hyperparams, load_model=False)
        avg_reward = np.mean(rewards[-10:])
        print(f"Avg reward last 10 episodes: {avg_reward:.2f}")
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = hyperparams
            print(f"New best TD3 params with avg reward {best_reward:.2f}")
    print("Best TD3 hyperparameters found:", best_params)

class MassageEnv:
    def __init__(self, physicsClient, armId, human_inst, nArmJoints, traj_step=100,
                 frequency=6, amp=0.035, x_offset=0.1, z_offset_lower=0.01, z_offset_upper=0.1,
                 region='lower_back', force_limit=100, traj_type='sine', massage_technique='normal',
                 max_steps=300):
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
        self.delta_angle = 0.05

        from itertools import product
        self.action_space = list(product([-self.delta_angle, 0, self.delta_angle], repeat=len(self.joint_indices)))
        self.action_size = len(self.action_space)

        self.current_step = 0
        self.max_steps = max_steps

        self.trajectory_vis_ids = []
        self.ee_vis_id = None

        self.trajectory_points = self._update_trajectory()
        self.target_joint_positions = [0.0] * len(self.joint_indices)

        self.prev_joint_velocities = np.zeros(len(self.joint_indices))
        p.configureDebugVisualizer(10, 1, physicsClientId=self.physicsClient)

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
        self.current_step = 0  # Reset step counter
        self.trajectory_points = self._update_trajectory()

        self.prev_joint_velocities = np.zeros(len(self.joint_indices))

        return self._get_state()

    def _update_trajectory(self):
        p1, p2 = p.getAABB(self.human_inst.body)
        if self.region == 'lower_back':
            z_pos = p1[2] + self.z_offset_lower
            y_pos = 0.03
        elif self.region == 'upper_back':
            z_pos = p2[2] - self.z_offset_upper
            y_pos = 0.4
        else:
            z_pos = (p1[2] + p2[2]) / 2
            y_pos = 0.3

        start = np.array([(p1[0] + p2[0]) / 2 - self.x_offset, y_pos, z_pos])
        end = np.array([(p1[0] + p2[0]) / 2 + self.x_offset, y_pos, z_pos])
        center = np.array([(p1[0] + p2[0]) / 2, y_pos, z_pos])

        if self.massage_technique == 'kneading':
            radius = self.x_offset / 2
            pnts = generate_kneading_trajectory(center, radius, self.traj_step, self.frequency, self.amp)
        elif self.massage_technique == 'pressure':
            pnts = generate_pressure_trajectory(center, self.traj_step)
        else:
            if self.traj_type == 'sine':
                pnts = generate_sine_trajectory(start, end, self.traj_step, self.frequency, self.amp)
            elif self.traj_type == 'circular':
                radius = self.x_offset
                pnts = generate_circular_trajectory(center, radius, self.traj_step, self.frequency, self.amp)
            elif self.traj_type == 'linear':
                pnts = generate_linear_trajectory(start, end, self.traj_step)
            else:
                pnts = generate_linear_trajectory(start, end, self.traj_step)

        trajectory_points = np.vstack((pnts, pnts[::-1]))
        self.trajectory_points = trajectory_points

        for vis_id in self.trajectory_vis_ids:
            try:
                p.getBodyInfo(vis_id, physicsClientId=self.physicsClient)
                p.removeBody(vis_id, physicsClientId=self.physicsClient)
            except:
                continue

        self.trajectory_vis_ids = []

        for point in trajectory_points:
            sphere_radius = 0.01
            sphere_color = [0, 1, 0, 1]
            sphere_vis_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=sphere_color, radius=sphere_radius)
            sphere_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_vis_id, basePosition=point)
            self.trajectory_vis_ids.append(sphere_body_id)

        return trajectory_points

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
        visualize_contact(self.armId, self.human_inst.body, self.physicsClient)
        visualize_contact_points(self.armId, self.human_inst.body, self.physicsClient)
        visualize_contact_force(self.armId, self.human_inst.body, self.physicsClient)

        if self.ee_vis_id is not None:
            try:
                p.getBodyInfo(self.ee_vis_id, physicsClientId=self.physicsClient)
                p.removeBody(self.ee_vis_id, physicsClientId=self.physicsClient)
            except:
                pass

        ee_link_state = p.getLinkState(self.armId, self.nArmJoints - 2, physicsClientId=self.physicsClient)
        ee_position = ee_link_state[0]

        sphere_radius = 0.02
        sphere_color = [1, 0, 0, 1]
        sphere_vis_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=sphere_color, radius=sphere_radius)
        self.ee_vis_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_vis_id, basePosition=ee_position)

        self.current_step += 1  # Increment step count

        next_state = self._get_state()
        reward = self._compute_reward()
        done = self.current_step >= self.max_steps
        return next_state, reward, done, {}

    def _compute_reward(self):
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, physicsClientId=self.physicsClient)
        total_force = sum([cp[9] for cp in contact_points]) if contact_points else 0.0
        target_min, target_max = 20, 40

        reward = 0.0

        if contact_points:
            reward += 0.2  # Encourage contact

        if target_min <= total_force <= target_max:
            center = (target_min + target_max) / 2
            width = (target_max - target_min) / 2
            reward += 1.0 - abs(total_force - center) / width
        else:
            dist = min(abs(total_force - target_min), abs(total_force - target_max))
            reward -= dist / target_max

        if total_force > target_max * 2:
            reward -= 3.0  # Strong penalty for excessive force

        joint_states = p.getJointStates(self.armId, self.joint_indices, physicsClientId=self.physicsClient)
        joint_velocities = np.array([state[1] for state in joint_states])
        velocity_change = np.sum(np.abs(joint_velocities - self.prev_joint_velocities))
        reward -= velocity_change * 0.02  # Penalize abrupt velocity changes
        self.prev_joint_velocities = joint_velocities

        # Progress bonus
        progress = self.current_step / self.max_steps
        reward += 0.5 * progress

        if not contact_points:
            reward -= 0.1  # Penalty for losing contact

        reward = max(min(reward, 1.0), -1.0)
        return reward
    def step_continuous(self, action):
        # action: numpy array of continuous increments for each joint
        increments = np.clip(action, -self.delta_angle, self.delta_angle)
        self.target_joint_positions = np.clip(
            np.array(self.target_joint_positions) + increments,
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
        visualize_contact(self.armId, self.human_inst.body, self.physicsClient)
        visualize_contact_points(self.armId, self.human_inst.body, self.physicsClient)
        visualize_contact_force(self.armId, self.human_inst.body, self.physicsClient)

        if self.ee_vis_id is not None:
            try:
                p.getBodyInfo(self.ee_vis_id, physicsClientId=self.physicsClient)
                p.removeBody(self.ee_vis_id, physicsClientId=self.physicsClient)
            except:
                pass

        ee_link_state = p.getLinkState(self.armId, self.nArmJoints - 2, physicsClientId=self.physicsClient)
        ee_position = ee_link_state[0]

        sphere_radius = 0.02
        sphere_color = [1, 0, 0, 1]
        sphere_vis_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=sphere_color, radius=sphere_radius)
        self.ee_vis_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_vis_id, basePosition=ee_position)

        self.current_step += 1

        next_state = self._get_state()
        reward = self._compute_reward()
        done = self.current_step >= self.max_steps
        return next_state, reward, done, {}    

class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
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

    def save(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"DQN model saved to {filepath}")

    def load(self, filepath):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            print(f"DQN model loaded from {filepath}")
        else:
            print(f"No DQN model found at {filepath}, starting fresh.")


class PPOAgent(nn.Module):
    def __init__(self, state_size, action_size, device='cpu', hidden_size=128, lr=3e-4, gamma=0.99, eps_clip=0.2):
        super(PPOAgent, self).__init__()
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

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

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"PPO model saved to {filepath}")

    def load(self, filepath):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"PPO model loaded from {filepath}")
        else:
            print(f"No PPO model found at {filepath}, starting fresh.")


# === TD3 Actor and Critic Networks ===

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

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


# === TD3 Agent ===

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device='cpu',
                 hidden_size=256, actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action, hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

        self.memory = deque(maxlen=100000)
        self.batch_size = 100

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        self.total_it += 1

        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"TD3 model saved to {filename}")

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            print(f"TD3 model loaded from {filename}")
        else:
            print(f"No TD3 model found at {filename}, starting fresh.")


# --- Safety check for inference ---
def check_safety_limits(env, state):
    joint_positions = state[:len(env.joint_indices)]
    max_angle = np.pi / 2
    if np.any(np.abs(joint_positions) > max_angle):
        print("Safety limit exceeded: joint angle too large.")
        return True

    contact_force = state[-1]
    max_force = 150
    if contact_force > max_force:
        print("Safety limit exceeded: contact force too high." ,contact_force)
        return True

    return False


# --- Real-time control loop for inference ---
def massage_control_loop(agent, env, num_steps=1000, sleep_time=0.01):
    state = env._get_state()
    for step in range(num_steps):
        if isinstance(agent, DQNAgent):
            action = agent.act(state)
        else:
            action, _ = agent.act(state)

        next_state, reward, done, _ = env.step(action)

        if check_safety_limits(env, next_state):
            print("Terminating control loop due to safety limits.")
            break

        if done:
            print(f"Episode finished at step {step}.")
            break

        state = next_state
        time.sleep(sleep_time)


# --- Inference runner ---
def run_inference_with_trained_model(model_type='dqn', model_path=None, num_steps=500):
    physicsClient = p.connect(p.GUI)
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
                     frequency=6,
                     amp=0.035,
                     x_offset=0.1,
                     z_offset_lower=0.01,
                     z_offset_upper=0.1,
                     region='lower_back',
                     force_limit=100,
                     traj_type='sine',
                     massage_technique='normal')

    state_size = len(env._get_state())
    action_size = env.action_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type.lower() == 'dqn':
        agent = DQNAgent(state_size, action_size, device=device)
        if model_path is not None:
            agent.load(model_path)
        agent.epsilon = 0.0
    elif model_type.lower() == 'ppo':
       # Use the hidden size that matches your saved model
        hidden_size = 128  # or 256, whichever you trained with
        agent = PPOAgent(state_size, action_size, device=device, hidden_size=hidden_size)
        if model_path is not None:
            agent.load(model_path)
    else:
        raise ValueError("Unsupported model_type. Use 'dqn' or 'ppo'.")

    massage_control_loop(agent, env, num_steps=num_steps)

    p.disconnect()


# --- GUI button callbacks for inference ---
def start_dqn_inference_gui():
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
    dqn_infer_button.config(state=tk.DISABLED)
    ppo_infer_button.config(state=tk.DISABLED)

    def run_infer():
        try:
            model_path = 'models/dqn/dqn_model.pth'
            run_inference_with_trained_model('dqn', model_path=model_path, num_steps=500)
        finally:
            root.after(0, lambda: [start_button.config(state=tk.NORMAL),
                                  rl_button.config(state=tk.NORMAL),
                                  ppo_button.config(state=tk.NORMAL),
                                  dqn_infer_button.config(state=tk.NORMAL),
                                  ppo_infer_button.config(state=tk.NORMAL)])

    threading.Thread(target=run_infer, daemon=True).start()


def run_ppo_training(frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                     region, force, traj_type, massage_technique,
                     max_episodes=100, update_timestep=2000,
                     save_dir='models/ppo', save_interval=10,
                     hyperparams=None, load_model=False, log_dir=None):
    if log_dir is None:
        log_dir = f"runs/ppo_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'ppo_model.pth')

    physicsClient = p.connect(p.GUI)
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

    if hyperparams is None:
        hyperparams = {}

    ppo_agent = PPOAgent(state_size, action_size, device=device,
                         hidden_size=hyperparams.get('hidden_size', 128),
                         lr=hyperparams.get('learning_rate', 3e-4),
                         gamma=hyperparams.get('gamma', 0.99),
                         eps_clip=hyperparams.get('eps_clip', 0.2))
    ppo_agent.to(device)

    if load_model:
        ppo_agent.load(model_path)

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
            writer.add_scalar('Reward', episode_reward, len(episode_rewards)-1)
            episode_reward = 0
            state = env.reset()

        if timestep % update_timestep == 0:
            states = torch.stack(memory['states'])
            actions = torch.stack(memory['actions'])
            old_logprobs = torch.stack(memory['logprobs'])
            rewards = memory['rewards']
            is_terminals = memory['is_terminals']

            returns = compute_returns(rewards, ppo_agent.gamma)

            K_epochs = hyperparams.get('K_epochs', 4)
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

            writer.add_scalar('Loss', loss.item(), timestep)
            memory = {k: [] for k in memory}
            print(f"PPO Update at timestep {t}, average reward: {np.mean(episode_rewards[-10:]) if episode_rewards else 0:.2f}")

        if len(episode_rewards) > 0 and len(episode_rewards) % save_interval == 0:
            ppo_agent.save(model_path)

    ppo_agent.save(model_path)
    writer.close()
    p.disconnect()

    plt.figure()
    plt.title("PPO Episode Rewards")
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

# === TD3 Training function ===

def run_td3_training(frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                     region, force, traj_type, massage_technique,
                     max_episodes=200, max_steps_per_episode=300,
                     save_dir='models/td3', save_interval=20,
                     hyperparams=None, load_model=False, log_dir=None):
    if log_dir is None:
        log_dir = f"runs/td3_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'td3_model.pth')

    physicsClient = p.connect(p.GUI)
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
                     massage_technique=massage_technique,
                     max_steps=max_steps_per_episode)

    state_size = len(env._get_state())
    action_size = len(env.joint_indices)
    max_action = env.delta_angle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hyperparams is None:
        hyperparams = {}

    agent = TD3Agent(state_size, action_size, max_action, device=device,
                     hidden_size=hyperparams.get('hidden_size', 256),
                     actor_lr=hyperparams.get('actor_lr', 3e-4),
                     critic_lr=hyperparams.get('critic_lr', 3e-4),
                     gamma=hyperparams.get('gamma', 0.99),
                     tau=hyperparams.get('tau', 0.005),
                     policy_noise=hyperparams.get('policy_noise', 0.2),
                     noise_clip=hyperparams.get('noise_clip', 0.5),
                     policy_freq=hyperparams.get('policy_freq', 2))

    if load_model:
        agent.load(model_path)

    episode_rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            noise = np.random.normal(0, max_action * 0.1, size=action_size)
            action = (action + noise).clip(-max_action, max_action)

            next_state, reward, done, _ = env.step_continuous(action)
            agent.add_experience(state, action, reward, next_state, float(done))

            agent.train()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)
        writer.add_scalar('Reward', episode_reward, episode)

        if (episode + 1) % save_interval == 0 or (episode + 1) == max_episodes:
            agent.save(model_path)

        print(f"TD3 Episode {episode + 1}/{max_episodes} Reward: {episode_reward:.3f}")

    writer.close()
    p.disconnect()

    plt.figure()
    plt.title("TD3 Episode Rewards")
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

def run_td3_inference(frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                      region, force, traj_type, massage_technique,
                      max_steps=300, model_path='models/td3/td3_model.pth'):
    physicsClient = p.connect(p.GUI)
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
                     massage_technique=massage_technique,
                     max_steps=max_steps)

    state_size = len(env._get_state())
    action_size = len(env.joint_indices)
    max_action = env.delta_angle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = 128  # Use the hidden size used during training
    agent = TD3Agent(state_size, action_size, max_action, device=device, hidden_size=hidden_size)
    agent.load(model_path)

    state = env.reset()
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step_continuous(action)
        state = next_state
        time.sleep(0.01)
        if done:
            print(f"Episode finished at step {step}")
            break

    p.disconnect()


# === Example GUI integration callbacks ===

def start_td3_training_gui():
    # Extract parameters from GUI entries (similar to your existing code)
    # For example:
    freq = float(freq_entry.get())
    amp = float(amp_entry.get())
    x_offset = float(x_offset_entry.get())
    z_offset_lower = float(z_offset_lower_entry.get())
    z_offset_upper = float(z_offset_upper_entry.get())
    region = region_var.get()
    force = float(force_entry.get())
    traj_type = traj_type_var.get()
    massage_technique = massage_technique_var.get()

    def run_training():
        run_td3_training(freq, amp, x_offset, z_offset_lower, z_offset_upper,
                         region, force, traj_type, massage_technique,
                         max_episodes=200, max_steps_per_episode=300,
                         save_dir='models/td3', save_interval=20,
                         hyperparams=None, load_model=True)

    threading.Thread(target=run_training, daemon=True).start()


def start_td3_inference_gui():
    freq = float(freq_entry.get())
    amp = float(amp_entry.get())
    x_offset = float(x_offset_entry.get())
    z_offset_lower = float(z_offset_lower_entry.get())
    z_offset_upper = float(z_offset_upper_entry.get())
    region = region_var.get()
    force = float(force_entry.get())
    traj_type = traj_type_var.get()
    massage_technique = massage_technique_var.get()

    def run_infer():
        run_td3_inference(freq, amp, x_offset, z_offset_lower, z_offset_upper,
                          region, force, traj_type, massage_technique,
                          max_steps=300, model_path='models/td3/td3_model.pth')

    threading.Thread(target=run_infer, daemon=True).start()

def start_ppo_inference_gui():
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
    dqn_infer_button.config(state=tk.DISABLED)
    ppo_infer_button.config(state=tk.DISABLED)

    def run_infer():
        try:
            model_path = 'models/ppo/ppo_model.pth'
            run_inference_with_trained_model('ppo', model_path=model_path, num_steps=500)
        finally:
            root.after(0, lambda: [start_button.config(state=tk.NORMAL),
                                  rl_button.config(state=tk.NORMAL),
                                  ppo_button.config(state=tk.NORMAL),
                                  dqn_infer_button.config(state=tk.NORMAL),
                                  ppo_infer_button.config(state=tk.NORMAL)])

    threading.Thread(target=run_infer, daemon=True).start()

def run_rl_training(frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                    region, force, traj_type, massage_technique,
                    max_rl_episodes=100, save_dir='models/dqn', save_interval=20,
                    hyperparams=None, load_model=False, log_dir=None):
    if log_dir is None:
        log_dir = f"runs/dqn_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'dqn_model.pth')

    physicsClient = p.connect(p.GUI)
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

    if hyperparams is None:
        hyperparams = {}

    agent = DQNAgent(state_size, action_size, device=device)
    if load_model:
        agent.load(model_path)

    # Override hyperparameters if provided
    agent.gamma = hyperparams.get('gamma', agent.gamma)
    agent.epsilon = hyperparams.get('epsilon', agent.epsilon)
    agent.epsilon_min = hyperparams.get('epsilon_min', agent.epsilon_min)
    agent.epsilon_decay = hyperparams.get('epsilon_decay', agent.epsilon_decay)
    agent.batch_size = hyperparams.get('batch_size', agent.batch_size)
    agent.learning_rate = hyperparams.get('learning_rate', agent.learning_rate)
    agent.optimizer = optim.Adam(agent.policy_net.parameters(), lr=agent.learning_rate)

    target_update_freq = hyperparams.get('target_update_freq', 5)

    episode_rewards = []
    episode_losses = []

    for e in range(max_rl_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        if e % target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)
        avg_loss = sum(agent.loss_history) / len(agent.loss_history) if agent.loss_history else 0
        episode_losses.append(avg_loss)
        agent.loss_history.clear()

        print(f"Episode {e+1} reward: {total_reward:.2f}, loss: {avg_loss:.4f}")

        # TensorBoard logging
        writer.add_scalar('Reward', total_reward, e)
        writer.add_scalar('Loss', avg_loss, e)

        if (e + 1) % save_interval == 0 or (e + 1) == max_rl_episodes:
            agent.save(model_path)

    writer.close()
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
# --- Existing RL training GUI buttons ---
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
        try:
            dqn_hyperparams = {
                'gamma': 0.95,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.99,
                'batch_size': 128,
                'learning_rate': 5e-4,
                'target_update_freq': 10
            }

            run_rl_training(freq, amp, x_offset, z_offset_lower, z_offset_upper,
                            region, force, traj_type, massage_technique,
                            max_rl_episodes=200, save_dir='models/dqn',
                            save_interval=25, hyperparams=dqn_hyperparams, load_model=True)
        finally:
            root.after(0, lambda: [start_button.config(state=tk.NORMAL), rl_button.config(state=tk.NORMAL)])

    threading.Thread(target=run_rl, daemon=True).start()


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
            ppo_hyperparams = {
                'hidden_size': 256,
                'learning_rate': 1e-4,
                'gamma': 0.98,
                'eps_clip': 0.15,
                'K_epochs': 5
            }

            run_ppo_training(freq, amp, x_offset, z_offset_lower, z_offset_upper,
                             region, force, traj_type, massage_technique,
                             max_episodes=150, update_timestep=1000,
                             save_dir='models/ppo', save_interval=10,
                             hyperparams=ppo_hyperparams, load_model=True)
        finally:
            root.after(0, lambda: [start_button.config(state=tk.NORMAL),
                                  rl_button.config(state=tk.NORMAL),
                                  ppo_button.config(state=tk.NORMAL)])

    threading.Thread(target=run_ppo, daemon=True).start()

def get_env_params_from_gui():
    return {
        'frequency': float(freq_entry.get()),
        'amplitude': float(amp_entry.get()),
        'x_offset': float(x_offset_entry.get()),
        'z_offset_lower': float(z_offset_lower_entry.get()),
        'z_offset_upper': float(z_offset_upper_entry.get()),
        'region': region_var.get(),
        'force': float(force_entry.get()),
        'traj_type': traj_type_var.get(),
        'massage_technique': massage_technique_var.get()
    }

def start_dqn_tuning():
    env_params = get_env_params_from_gui()
    threading.Thread(target=random_search_dqn, args=(env_params, 10), daemon=True).start()

def start_ppo_tuning():
    env_params = get_env_params_from_gui()
    threading.Thread(target=random_search_ppo, args=(env_params, 10), daemon=True).start()

def start_td3_tuning():
    env_params = get_env_params_from_gui()
    threading.Thread(target=random_search_td3, args=(env_params, 10), daemon=True).start()

dqn_infer_button = tk.Button(root, text="Run DQN Inference", command=start_dqn_inference_gui)
dqn_infer_button.grid(row=14, column=0, columnspan=2, pady=10)

ppo_infer_button = tk.Button(root, text="Run PPO Inference", command=start_ppo_inference_gui)
ppo_infer_button.grid(row=14, column=5, columnspan=2, pady=10)

rl_button = tk.Button(root, text="Start RL Training(DQN)", command=start_rl_training_gui)
rl_button.grid(row=11, column=0, columnspan=2, pady=10)

ppo_button = tk.Button(root, text="Start PPO Training", command=start_ppo_training_gui)
ppo_button.grid(row=12, column=0, columnspan=2, pady=10)

train_td3_button = tk.Button(root, text="Train TD3", command=start_td3_training_gui)
train_td3_button.grid(row=13, column=0, columnspan=2, pady=10)

infer_td3_button = tk.Button(root, text="Run TD3 Inference", command=start_td3_inference_gui)
infer_td3_button.grid(row=14, column=20, padx=10, pady=10)

dqn_tune_button = tk.Button(root, text="Tune DQN Hyperparams", command=start_dqn_tuning)
dqn_tune_button.grid(row=11, column=20)  

ppo_tune_button = tk.Button(root, text="Tune PPO Hyperparams", command=start_ppo_tuning)
ppo_tune_button.grid(row=12, column=20)  

td3_tune_button = tk.Button(root, text="Tune TD3 Hyperparams", command=start_td3_tuning)
td3_tune_button.grid(row=13, column=20, )  

root.mainloop()
