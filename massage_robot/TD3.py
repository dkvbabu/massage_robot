import os
import time
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import configparser

# --- Utility functions for trajectory generation ---

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
    pnts = np.tile(center, (numSamples, 1))
    return pnts

# --- Load human and furniture scene ---

def load_scene(physicsClient):
    from human.human_creation import HumanCreation
    from human.human import Human
    from human.furniture import Furniture

    human_creation = HumanCreation(physicsClient, np_random=np.random)
    human_controllable_joint_indices = []
    human_inst = Human(human_controllable_joint_indices, controllable=False)

    configP = configparser.ConfigParser()
    configP.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), './human/config.ini'))

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

# --- Visualization helpers ---

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
        max_force = 30
        intensity = min(contact_force / max_force, 1.0)
        color = [1, 1 - intensity, 1 - intensity, 1]
        p.changeVisualShape(objectUniqueId=armId, linkIndex=contacting_link_index, rgbaColor=color, physicsClientId=physicsClient)
    else:
        for link_index in range(-1, p.getNumJoints(armId, physicsClientId=physicsClient)):
            p.changeVisualShape(objectUniqueId=armId, linkIndex=link_index, rgbaColor=[1, 1, 1, 1], physicsClientId=physicsClient)

# --- MassageEnv class ---

class MassageEnv:
    def __init__(self, physicsClient, armId, human_inst, nArmJoints, traj_step=100,
                 frequency=3, amp=0.01, x_offset=0.1, z_offset_lower=0.01, z_offset_upper=0.1,
                 region='lower_back', force_limit=30, traj_type='sine', massage_technique='normal',
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

        self.joint_indices = list(range(1, nArmJoints - 1))
        self.delta_angle = 0.01

        from itertools import product
        self.action_space = list(product([-self.delta_angle, 0, self.delta_angle], repeat=len(self.joint_indices)))
        self.action_size = len(self.action_space)

        self.current_step = 0
        self.max_steps = max_steps

        self.trajectory_vis_ids = []
        self.ee_vis_id = None

        # Initialize trajectory points
        self.trajectory_points = self._update_trajectory()
        self.target_joint_positions = [0.0] * len(self.joint_indices)

        self.prev_joint_velocities = np.zeros(len(self.joint_indices))

        # New attributes for smooth descent and offsets
        self.step_count = 0
        self.steps_to_reach_back = 50  # Number of steps to move down to human back

        self.initial_height_offset = 0.1  # 10 cm above human back surface
        self.end_effector_offset_z = 0.02  # Contact offset height

        # End-effector link index (update if different)
        self.end_effector_link_index = 7

        # For collision safety and visualization
        self.previous_safe_joint_positions = None

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physicsClient)

        # Disable unwanted collisions
        self.disable_unwanted_collisions()

    def reset(self):
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physicsClient)
        p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)

        startPos = [-0.7, 0.1, 1.0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "urdf/ur5_robot_mod1.urdf")
        try:
            self.armId = p.loadURDF(urdf_path, startPos, startOrientation, physicsClientId=self.physicsClient)
        except Exception as e:
            print("Failed to load URDF:", e)
        self.nArmJoints = p.getNumJoints(self.armId, physicsClientId=self.physicsClient)
        self.joint_indices = list(range(1, self.nArmJoints - 1))

        self.human_inst = load_scene(self.physicsClient)

        self.target_joint_positions = [0.0] * len(self.joint_indices)
        self.current_step = 0

        self.trajectory_points = self._update_trajectory()

        self.prev_joint_velocities = np.zeros(len(self.joint_indices))

        initial_point = self.trajectory_points[0]
        initial_above_pos = [initial_point[0], initial_point[1], initial_point[2] + self.initial_height_offset]

        joint_positions = p.calculateInverseKinematics(self.armId, self.end_effector_link_index, initial_above_pos, physicsClientId=self.physicsClient)

        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.armId, joint_index, joint_positions[i], physicsClientId=self.physicsClient)

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physicsClient)

        self.step_count = 0
        self.steps_to_reach_back = 50

        self.target_joint_positions = list(joint_positions[:len(self.joint_indices)])
        self.previous_safe_joint_positions = self.target_joint_positions.copy()

        return self._get_state()

    def _update_trajectory(self):
        p1, p2 = p.getAABB(self.human_inst.body)
        inward_offset_y = 0.01
        inward_offset_x = 0.05
        clearance_z = 0.05

        if self.region == 'lower_back':
            z_pos = p2[2] + self.z_offset_lower + clearance_z
            y_pos = 0.03 - inward_offset_y
        elif self.region == 'upper_back':
            z_pos = p2[2] - self.z_offset_upper + clearance_z
            y_pos = 0.4 - inward_offset_y
        else:
            z_pos = (p1[2] + p2[2]) / 2 + clearance_z
            y_pos = 0.3 - inward_offset_y

        center_x = (p1[0] + p2[0]) / 2 - inward_offset_x

        start = np.array([center_x - self.x_offset, y_pos, z_pos])
        end = np.array([center_x + self.x_offset, y_pos, z_pos])
        center = np.array([center_x, y_pos, z_pos])

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

        if self.region == 'lower_back':
            orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        elif self.region == 'upper_back':
            orientation = p.getQuaternionFromEuler([np.pi, 0, np.pi / 2])
        else:
            orientation = p.getQuaternionFromEuler([0, np.pi, 0])

        joint_poses = p.calculateInverseKinematics(self.armId, 7, traj_point, orientation)

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

        ee_link_state = p.getLinkState(self.armId, 7, physicsClientId=self.physicsClient)
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

    def step_continuous(self, action):
        traj_point = self.trajectory_points[self.current_step % len(self.trajectory_points)]

        if self.region == 'lower_back':
            orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        elif self.region == 'upper_back':
            orientation = p.getQuaternionFromEuler([np.pi, 0, np.pi / 2])
        else:
            orientation = p.getQuaternionFromEuler([0, np.pi, 0])

        joint_poses = p.calculateInverseKinematics(
            self.armId,
            self.end_effector_link_index,
            traj_point,
            orientation,
            physicsClientId=self.physicsClient
        )

        p.setJointMotorControlArray(
            self.armId,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_poses[:len(self.joint_indices)],
            forces=self.force_limit * np.ones(len(self.joint_indices)),
            physicsClientId=self.physicsClient
        )

        p.stepSimulation(physicsClientId=self.physicsClient)

        contact_points = p.getContactPoints(
            bodyA=self.armId,
            bodyB=self.human_inst.body,
            linkIndexA=self.end_effector_link_index,
            physicsClientId=self.physicsClient
        )
        if contact_points:
            print(f"Contact detected at step {self.current_step}, points: {len(contact_points)}")
            for contact in contact_points:
                pos_ee = contact[5]
                pos_human = contact[6]
                #print(
                #    f"Force: {contact[9]:.3f}, "
                #    f"Pos on ee_link: ({pos_ee[0]:.3f}, {pos_ee[1]:.3f}, {pos_ee[2]:.3f}), "
                #    f"Pos on human: ({pos_human[0]:.3f}, {pos_human[1]:.3f}, {pos_human[2]:.3f})"
                #)
        else:
            pass
            #print(f"No contact at step {self.current_step}")

        visualize_contact(self.armId, self.human_inst.body, self.physicsClient)
        visualize_contact_points(self.armId, self.human_inst.body, self.physicsClient)
        visualize_contact_force(self.armId, self.human_inst.body, self.physicsClient)

        if self.ee_vis_id is not None:
            try:
                p.getBodyInfo(self.ee_vis_id, physicsClientId=self.physicsClient)
                p.removeBody(self.ee_vis_id, physicsClientId=self.physicsClient)
            except:
                pass

        ee_link_state = p.getLinkState(self.armId, self.end_effector_link_index, physicsClientId=self.physicsClient)
        ee_position = ee_link_state[0]

        sphere_radius = 0.02
        sphere_color = [1, 0, 0, 1]
        sphere_vis_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=sphere_color, radius=sphere_radius)
        self.ee_vis_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_vis_id, basePosition=ee_position, physicsClientId=self.physicsClient)

        self.current_step += 1

        next_state = self._get_state()
        total_reward, reward_components = self._compute_reward()
        done = self.current_step >= self.max_steps

        return next_state, total_reward, done, reward_components

    def _compute_reward(self):
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, linkIndexA=7, physicsClientId=self.physicsClient)

        total_force = sum([cp[9] for cp in contact_points]) if contact_points else 0.0

        target_min, target_max = 15, 50

        reward_contact = 0.5
        reward_force = 0.0
        reward_penalty_force = 0.0
        reward_velocity_penalty = 0.0
        reward_progress = 0.0
        reward_no_contact_penalty = 0.0

        if contact_points:
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

        joint_states = p.getJointStates(self.armId, self.joint_indices, physicsClientId=self.physicsClient)
        joint_velocities = np.array([state[1] for state in joint_states])
        velocity_change = np.sum(np.abs(joint_velocities - self.prev_joint_velocities))
        reward_velocity_penalty = -velocity_change * 0.01
        self.prev_joint_velocities = joint_velocities

        progress = self.current_step / self.max_steps
        reward_progress = 0.3 * progress

        total_reward = reward_contact + reward_force + reward_penalty_force + reward_velocity_penalty + reward_progress + reward_no_contact_penalty
        total_reward = max(min(total_reward, 1.0), -1.0)

        return total_reward, {
            'reward_contact': reward_contact,
            'reward_force': reward_force,
            'reward_penalty_force': reward_penalty_force,
            'reward_velocity_penalty': reward_velocity_penalty,
            'reward_progress': reward_progress,
            'reward_no_contact_penalty': reward_no_contact_penalty
        }

    def disable_unwanted_collisions(self):
        num_joints = p.getNumJoints(self.armId)
        for link_idx in range(-1, num_joints):
            if link_idx != self.end_effector_link_index:
                p.setCollisionFilterPair(
                    self.armId,
                    self.human_inst.body,
                    link_idx,
                    -1,
                    enableCollision=0,
                    physicsClientId=self.physicsClient
                )

# --- TD3 Actor and Critic networks ---

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
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

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

# --- TD3 Agent ---

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

        self.memory = deque(maxlen=10000)
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

# --- TD3 training function ---

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

    #physicsClient = p.connect(p.GUI)
    physicsClient = p.connect(p.DIRECT)
    if physicsClient < 0:
        raise RuntimeError("Failed to connect to PyBullet GUI")
    print(f"Connected to PyBullet with client id: {physicsClient}")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    startPos = [-0.7, 0.1, 1.0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("plane.urdf", physicsClientId=physicsClient)

    urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "urdf/ur5_robot_mod1.urdf")
    print(f"Loading robot URDF from: {urdf_path}")
    try:
        armId = p.loadURDF(urdf_path, startPos, startOrientation, physicsClientId=physicsClient)
        print("Robot URDF loaded successfully.")
    except Exception as e:
        print("Failed to load URDF:", e)
        p.disconnect()
        return

    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)
    human_inst = load_scene(physicsClient)
    print("Human scene loaded.")
    num_joints = p.getNumJoints(armId, physicsClientId=physicsClient)
    print(f"Number of joints: {num_joints}")

    for i in range(num_joints):
        info = p.getJointInfo(armId, i, physicsClientId=physicsClient)
        joint_index = info[0]
        joint_name = info[1].decode('utf-8')
        link_name = info[12].decode('utf-8')
        print(f"Joint index: {joint_index}, Joint name: {joint_name}, Link name: {link_name}")

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
    print("Environment initialized.")

    state = env.reset()
    print(f"Initial state shape: {state.shape}")

    state_size = len(state)
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
    try:
        for episode in range(max_episodes):
            print(f"Starting episode {episode + 1}/{max_episodes}")
            state = env.reset()
            episode_reward = 0
            for step in range(max_steps_per_episode):
                #print(f"Episode {episode + 1} Step {step + 1} - before step_continuous")
                action = agent.select_action(state)
                noise = np.random.normal(0, max_action * 0.1, size=action_size)
                action = (action + noise).clip(-max_action, max_action)

                next_state, reward, done, reward_components = env.step_continuous(action)
                #print(f"Episode {episode + 1} Step {step + 1} - after step_continuous")
                if isinstance(reward, (tuple, list, np.ndarray)):
                    print(f"Reward: {reward}, Done: {done}")
                else:
                    print(f"Reward: {reward:.3f}, Done: {done}")
                #print(f"Reward components: {reward_components}")

                agent.add_experience(state, action, reward, next_state, float(done))
                agent.train()

                state = next_state
                episode_reward += reward

                if done:
                    print(f"Episode {episode + 1} finished after {step + 1} steps.")
                    break

            episode_rewards.append(episode_reward)
            writer.add_scalar('Reward', episode_reward, episode)

            if (episode + 1) % save_interval == 0 or (episode + 1) == max_episodes:
                agent.save(model_path)

            print(f"TD3 Episode {episode + 1}/{max_episodes} Reward: {episode_reward:.3f}")

    except Exception as e:
        print(f"Exception in training loop: {e}")
    writer.close()
    p.disconnect()

    plt.figure()
    plt.title("TD3 Episode Rewards")
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    return episode_rewards

# --- Random search for hyperparameters ---

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
        rewards = run_td3_training(
            frequency=6,
            amplitude=0.01,
            x_offset=0.1,
            z_offset_lower=0.01,
            z_offset_upper=0.1,
            region='lower_back',
            force=30,
            traj_type='sine',
            massage_technique='normal',
            max_episodes=50,
            hyperparams=hyperparams,
            load_model=False
        )
        avg_reward = np.mean(rewards)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = hyperparams
    return best_params, best_reward

import csv
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
import torch
import pybullet as p
import pybullet_data
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns  # For enhanced plotting

def run_td3_inference(frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                      region, force, traj_type, massage_technique,
                      max_episodes=10, max_steps_per_episode=100,
                      save_dir='models/td3',
                      hyperparams=None, load_model=True, log_dir=None):
    print("Starting TD3 inference...")

    if log_dir is None:
        log_dir = f"runs/td3_inference_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'td3_model.pth')

    physicsClient = p.connect(p.DIRECT)
    print(f"Connected to PyBullet with client id: {physicsClient}")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    startPos = [-0.7, 0.1, 1.0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("plane.urdf", physicsClientId=physicsClient)
    urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "urdf/ur5_robot_mod1.urdf")
    print(f"Loading robot URDF from: {urdf_path}")
    armId = p.loadURDF(urdf_path, startPos, startOrientation, physicsClientId=physicsClient)
    print("Robot URDF loaded.")

    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)
    print(f"Number of arm joints: {nArmJoints}")

    human_inst = load_scene(physicsClient)
    print("Human scene loaded.")

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
    print("Environment initialized.")

    state_size = len(env._get_state())
    action_size = len(env.joint_indices)
    max_action = env.delta_angle
    print(f"State size: {state_size}, Action size: {action_size}, Max action: {max_action}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hidden_size = 256  # Use the hidden size used during training
    agent = TD3Agent(state_size, action_size, max_action, device=device, hidden_size=hidden_size)
    if load_model:
        print(f"Loading model from {model_path} ...")
        agent.load(model_path)

    episode_rewards = []
    reward_components_history = []
    max_episodes = 10

    try:
        for episode in range(max_episodes):
            print(f"Starting episode {episode + 1}/{max_episodes}")
            state = env.reset()
            episode_reward = 0
            episode_reward_components = []

            for step in range(max_steps_per_episode):
                action = agent.select_action(state)
                noise = np.random.normal(0, max_action * 0.1, size=action_size)
                action = (action + noise).clip(-max_action, max_action)

                next_state, reward, done, reward_components = env.step_continuous(action)

                if isinstance(reward, (tuple, list, np.ndarray)):
                    print(f"Reward: {reward}, Done: {done}")
                else:
                    print(f"Reward: {reward:.3f}, Done: {done}")

                state = next_state
                episode_reward += reward if not isinstance(reward, (tuple, list, np.ndarray)) else sum(reward)
                episode_reward_components.append(reward_components)

                if done:
                    print(f"Episode {episode + 1} finished after {step + 1} steps.")
                    break

            episode_rewards.append(episode_reward)
            reward_components_history.append(episode_reward_components)
            writer.add_scalar('Reward', episode_reward, episode)

            print(f"TD3 Episode {episode + 1}/{max_episodes} Reward: {episode_reward:.3f}")

    except Exception as e:
        print(f"Exception in inference loop: {e}")

    writer.close()
    p.disconnect()
    print("Disconnected from PyBullet. Inference finished.")

    # --- Debug prints before aggregation ---
    print("Starting aggregation and plotting...")
    print(f"Number of episodes recorded: {len(episode_rewards)}")
    print(f"Reward components history length: {len(reward_components_history)}")
    if len(reward_components_history) > 0:
        print(f"Sample reward components from first episode: {reward_components_history[0][:3]}")  # first 3 steps

    # Aggregate and plot average reward components per episode
    avg_components = {}
    for episode_components in reward_components_history:
        if not episode_components:
            continue
        keys = episode_components[0].keys()
        avg = {k: 0.0 for k in keys}
        for step_comp in episode_components:
            for k in keys:
                avg[k] += step_comp.get(k, 0.0)
        for k in keys:
            avg[k] /= len(episode_components)
        for k, v in avg.items():
            avg_components.setdefault(k, []).append(v)

    # Plot total episode rewards and save/show
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
    print("Saved and displayed plot: td3_inference_episode_rewards.png")

    # Plot average reward components and save/show
    plt.figure(figsize=(10, 6))
    for k, values in avg_components.items():
        plt.plot(values, label=f'Avg {k}')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward Component')
    plt.title('Average Reward Components per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('td3_inference_avg_reward_components.png')
    plt.show()
    plt.close()
    print("Saved and displayed plot: td3_inference_avg_reward_components.png")

    # Reward distribution histogram save/show
    plt.figure(figsize=(8, 5))
    plt.hist(episode_rewards, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Total Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Episode Rewards')
    plt.grid(True)
    plt.savefig('td3_inference_reward_distribution.png')
    plt.show()
    plt.close()
    print("Saved and displayed plot: td3_inference_reward_distribution.png")

    # Cumulative reward curve save/show
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
    print("Saved and displayed plot: td3_inference_cumulative_reward.png")

    # Boxplot of average reward components save/show
    plt.figure(figsize=(10, 6))
    data_for_boxplot = [avg_components[k] for k in avg_components.keys()]
    sns.boxplot(data=data_for_boxplot)
    plt.xticks(ticks=range(len(avg_components)), labels=list(avg_components.keys()), rotation=45)
    plt.ylabel('Average Reward Component Value')
    plt.title('Boxplot of Average Reward Components per Episode')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('td3_inference_reward_components_boxplot.png')
    plt.show()
    plt.close()
    print("Saved and displayed plot: td3_inference_reward_components_boxplot.png")

    # Save summary to CSV
    csv_file = 'td3_inference_summary.csv'
    with open(csv_file, mode='w', newline='') as f:
        writer_csv = csv.writer(f)
        header = ['Episode', 'TotalReward'] + list(avg_components.keys())
        writer_csv.writerow(header)
        for i in range(len(episode_rewards)):
            row = [i + 1, episode_rewards[i]]
            for k in avg_components.keys():
                row.append(avg_components[k][i] if i < len(avg_components[k]) else '')
            writer_csv.writerow(row)

    print(f"Inference summary saved to {csv_file}")

import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Queue
import time

# The TD3 functions are assumed to be in the same file (Part 1 above),
# so we can call them directly here.

def training_process(params, queue):
    try:
        rewards = run_td3_training(**params)
        queue.put("Training completed")
    except Exception as e:
        queue.put(f"Training error: {e}")

def inference_process(params, queue):
    try:
        # You can implement run_td3_inference similarly if needed
        # For now, just a placeholder call
        run_td3_inference(**params)
        queue.put("Inference completed")
    except Exception as e:
        queue.put(f"Inference error: {e}")

def tuning_process(env_params, queue):
    try:
        best_params, best_reward = random_search_td3(env_params)
        queue.put(f"Tuning completed. Best reward: {best_reward:.2f}")
        queue.put(f"Best params: {best_params}")
    except Exception as e:
        queue.put(f"Tuning error: {e}")

class TD3ControlApp:
    def __init__(self, root):
        self.root = root
        root.title("TD3 Control Panel")

        tk.Label(root, text="Frequency").grid(row=0, column=0, sticky="w")
        self.freq_var = tk.DoubleVar(value=6.0)
        freq_slider = ttk.Scale(root, from_=0.1, to=10.0, variable=self.freq_var, orient='horizontal')
        freq_slider.grid(row=0, column=1, sticky="ew")

        tk.Label(root, text="Amplitude").grid(row=1, column=0, sticky="w")
        self.amp_var = tk.DoubleVar(value=0.01)
        amp_slider = ttk.Scale(root, from_=0.0, to=0.05, variable=self.amp_var, orient='horizontal')
        amp_slider.grid(row=1, column=1, sticky="ew")

        self.status_var = tk.StringVar(value="Idle")
        status_label = tk.Label(root, textvariable=self.status_var)
        status_label.grid(row=4, column=0, columnspan=2)

        train_button = ttk.Button(root, text="Start Training", command=self.start_training)
        train_button.grid(row=2, column=0, pady=10)

        infer_button = ttk.Button(root, text="Start Inference", command=self.start_inference)
        infer_button.grid(row=2, column=1, pady=10)

        tune_button = ttk.Button(root, text="Start Hyperparam Tuning", command=self.start_tuning)
        tune_button.grid(row=3, column=0, columnspan=2, pady=10)

        root.columnconfigure(1, weight=1)

        self.process = None
        self.queue = Queue()
        self.root.after(100, self.check_queue)

    def get_params(self):
        return {
            'frequency': self.freq_var.get(),
            'amplitude': self.amp_var.get(),
            'x_offset': 0.1,
            'z_offset_lower': 0.01,
            'z_offset_upper': 0.1,
            'region': 'lower_back',
            'force': 30,
            'traj_type': 'sine',
            'massage_technique': 'normal',
            'max_episodes': 200,
            'max_steps_per_episode': 300,
            'load_model': False
        }

    def start_training(self):
        if self.process is None or not self.process.is_alive():
            self.status_var.set("Training started...")
            params = self.get_params()
            self.process = Process(target=training_process, args=(params, self.queue))
            self.process.start()
        else:
            self.status_var.set("Process already running")

    def start_inference(self):
        if self.process is None or not self.process.is_alive():
            self.status_var.set("Inference started...")
            params = self.get_params()
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
            self.status_var.set(msg)
        self.root.after(100, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = TD3ControlApp(root)
    root.mainloop()
