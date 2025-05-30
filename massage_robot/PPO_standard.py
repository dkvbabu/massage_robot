import os
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from multiprocessing import Process, Queue
import random
import seaborn as sns
import time

# -------------------------
# Trajectory generation functions
# -------------------------

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

# -------------------------
# Visualization helpers
# -------------------------

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

# -------------------------
# Load scene helper (simplified placeholder)
# -------------------------

def load_scene(physicsClient):
    from human.human_creation import HumanCreation
    from human.human import Human
    from human.furniture import Furniture

    human_creation = HumanCreation(physicsClient, np_random=np.random)
    human_controllable_joint_indices = []
    human_inst = Human(human_controllable_joint_indices, controllable=False)

    import configparser
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
# -------------------------
# Gym Environment
# -------------------------

class MassageEnvGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 traj_step=100,
                 frequency=3,
                 amp=0.01,
                 x_offset=0.1,
                 z_offset_lower=0.01,
                 z_offset_upper=0.1,
                 region='lower_back',
                 force_limit=30,
                 traj_type='sine',
                 massage_technique='normal',
                 max_steps=300,
                 render=False):
        super(MassageEnvGym, self).__init__()

        self.render_mode = render
        self.physicsClient = None
        self.armId = None
        self.human_inst = None
        self.nArmJoints = None

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
        self.max_steps = max_steps

        self.delta_angle = 0.001 # max joint increment per step

        self.current_step = 0

        self._init_simulation()

        self.joint_indices = list(range(1, self.nArmJoints - 1))

        self.trajectory_points = self._update_trajectory()

        self.target_joint_positions = np.zeros(len(self.joint_indices))

        self.prev_joint_velocities = np.zeros(len(self.joint_indices))

        self.ee_vis_id = None

        obs_dim = 2 * len(self.joint_indices) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.action_space = spaces.Box(low=-self.delta_angle, high=self.delta_angle, shape=(len(self.joint_indices),), dtype=np.float32)

    def _init_simulation(self):
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physicsClient)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.physicsClient)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.physicsClient)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.physicsClient)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        if self.physicsClient < 0:
            raise RuntimeError("Failed to connect to PyBullet")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10, physicsClientId=self.physicsClient)

        p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)

        startPos = [-0.7, 0.1, 1.0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "urdf/ur5_robot_mod1.urdf")
        if not os.path.exists(urdf_path):
            # Fallback to a simple robot URDF from pybullet_data if your URDF is missing
            urdf_path = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self.armId = p.loadURDF(urdf_path, startPos, startOrientation, physicsClientId=self.physicsClient)

        self.nArmJoints = p.getNumJoints(self.armId, physicsClientId=self.physicsClient)

        self.human_inst = load_scene(self.physicsClient)

        self.disable_unwanted_collisions()

    def _update_trajectory(self):
        p1, p2 = p.getAABB(self.human_inst.body, physicsClientId=self.physicsClient)
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
        return trajectory_points

    def reset(self):
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)

        startPos = [-0.7, 0.1, 1.0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "urdf/ur5_robot_mod1.urdf")
        if not os.path.exists(urdf_path):
            urdf_path = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self.armId = p.loadURDF(urdf_path, startPos, startOrientation, physicsClientId=self.physicsClient)
        self.nArmJoints = p.getNumJoints(self.armId, physicsClientId=self.physicsClient)
        self.joint_indices = list(range(1, self.nArmJoints - 1))

        self.human_inst = load_scene(self.physicsClient)

        self.trajectory_points = self._update_trajectory()
        self.current_step = 0
        self.target_joint_positions = np.zeros(len(self.joint_indices))
        self.prev_joint_velocities = np.zeros(len(self.joint_indices))

        initial_point = self.trajectory_points[0]
        initial_above_pos = [initial_point[0], initial_point[1], initial_point[2] + 0.1]

        joint_positions = p.calculateInverseKinematics(self.armId, 7, initial_above_pos, physicsClientId=self.physicsClient)
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(self.armId, joint_index, joint_positions[i], physicsClientId=self.physicsClient)

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physicsClient)

        self.target_joint_positions = np.array(joint_positions[:len(self.joint_indices)])

        self.disable_unwanted_collisions()

        return self._get_state()

    def _get_state(self):
        joint_states = p.getJointStates(self.armId, self.joint_indices, physicsClientId=self.physicsClient)
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, physicsClientId=self.physicsClient)
        total_force = sum([cp[9] for cp in contact_points]) if contact_points else 0.0
        state = np.concatenate([joint_positions, joint_velocities, [total_force]])
        return state.astype(np.float32)

    def step(self, action):
        increments = np.clip(action, -self.delta_angle, self.delta_angle)
        self.target_joint_positions = np.clip(self.target_joint_positions + increments, -np.pi, np.pi)

        traj_point = self.trajectory_points[self.current_step % len(self.trajectory_points)]

        if self.region == 'lower_back':
            orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        elif self.region == 'upper_back':
            orientation = p.getQuaternionFromEuler([np.pi, 0, np.pi / 2])
        else:
            orientation = p.getQuaternionFromEuler([0, np.pi, 0])

        joint_poses = p.calculateInverseKinematics(
            self.armId,
            7,
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

        # Collect Forces, armparts, bodyparts from contact points
        contact_points = p.getContactPoints(bodyA=self.armId, bodyB=self.human_inst.body, physicsClientId=self.physicsClient)
        Forces = []
        armparts = []
        bodyparts = []
        if contact_points:
            for cp in contact_points:
                Forces.append(cp[9])       # Normal force magnitude
                armparts.append(cp[3])     # Link index on arm
                bodyparts.append(cp[4])    # Link index on human body
        else:
            # No contact, append zeros
            Forces.append(0)
            armparts.append(0)
            bodyparts.append(0)

        if contact_points:
            print(f"Contact detected at step {self.current_step}, points: {len(contact_points)}")
            for contact in contact_points:
                pos_ee = contact[5]
                pos_human = contact[6]
                # Uncomment for detailed contact info print
                # print(
                #     f"Force: {contact[9]:.3f}, "
                #     f"Pos on ee_link: ({pos_ee[0]:.3f}, {pos_ee[1]:.3f}, {pos_ee[2]:.3f}), "
                #     f"Pos on human: ({pos_human[0]:.3f}, {pos_human[1]:.3f}, {pos_human[2]:.3f})"
                # )
        else:
            pass
            # print(f"No contact at step {self.current_step}")

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
        self.ee_vis_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_vis_id, basePosition=ee_position, physicsClientId=self.physicsClient)

        self.current_step += 1

        next_state = self._get_state()
        reward = self._compute_reward()
        done = self.current_step >= self.max_steps

        info = self.last_reward_components.copy() if hasattr(self, 'last_reward_components') else {}

        # Add collected contact info to info dict
        info['Forces'] = Forces
        info['armparts'] = armparts
        info['bodyparts'] = bodyparts

        time.sleep(0.02)

        return next_state, reward, done, info
    
    def _compute_reward(self):
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, linkIndexA=7, physicsClientId=self.physicsClient)
        total_force = sum([cp[9] for cp in contact_points]) if contact_points else 0.0

        target_min, target_max = 15, 50

        reward_contact = 0.0
        reward_force = 0.0
        reward_penalty_force = 0.0
        reward_velocity_penalty = 0.0
        reward_progress = 0.0
        reward_no_contact_penalty = 0.0

        if contact_points:
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

        joint_states = p.getJointStates(self.armId, self.joint_indices, physicsClientId=self.physicsClient)
        joint_velocities = np.array([state[1] for state in joint_states])
        velocity_change = np.sum(np.abs(joint_velocities - self.prev_joint_velocities))
        reward_velocity_penalty = -velocity_change * 0.01
        self.prev_joint_velocities = joint_velocities

        progress = self.current_step / self.max_steps
        reward_progress = 0.3 * progress

        total_reward = reward_contact + reward_force + reward_penalty_force + reward_velocity_penalty + reward_progress + reward_no_contact_penalty
        total_reward = max(min(total_reward, 1.0), -1.0)

        self.last_reward_components = {
            'reward_contact': reward_contact,
            'reward_force': reward_force,
            'reward_penalty_force': reward_penalty_force,
            'reward_velocity_penalty': reward_velocity_penalty,
            'reward_progress': reward_progress,
            'reward_no_contact_penalty': reward_no_contact_penalty,
            'total_reward': total_reward
        }

        return total_reward

    def close(self):
        if self.physicsClient is not None:
            p.disconnect(self.physicsClient)
            self.physicsClient = None

    def disable_unwanted_collisions(self):
        # Set the EE link index explicitly if not set
        if not hasattr(self, 'end_effector_link_index'):
            self.end_effector_link_index = 7  # Adjust if your EE link index differs

        num_joints = p.getNumJoints(self.armId, physicsClientId=self.physicsClient)

        # Disable collisions between all arm links except EE link and the human body
        for link_idx in range(-1, num_joints):
            if link_idx != self.end_effector_link_index:
                p.setCollisionFilterPair(
                    self.armId,
                    self.human_inst.body,
                    link_idx,
                    -1,  # -1 means all links of human body
                    enableCollision=0,
                    physicsClientId=self.physicsClient
                )
        print(f"Collision filtering applied: only EE link {self.end_effector_link_index} collides with human body.")

# -------------------------
# Callbacks for training
# -------------------------

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
            if hasattr(env, 'episode_rewards') and len(env.episode_rewards) > 0:
                mean_reward = np.mean(env.episode_rewards[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True

class RewardLoggingCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_rewards = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        self._current_rewards += self.locals.get('rewards', [0])[0] if 'rewards' in self.locals else 0
        self._current_length += 1
        done = self.locals.get('dones', [False])[0] if 'dones' in self.locals else False
        if done:
            self.episode_rewards.append(self._current_rewards)
            self.episode_lengths.append(self._current_length)
            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)} reward: {self._current_rewards:.3f}, length: {self._current_length}")
            self._current_rewards = 0.0
            self._current_length = 0
        return True

    def _on_training_end(self) -> None:
        if self.episode_rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, label='Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Episode Rewards')
            plt.legend()
            plt.grid(True)
            os.makedirs(self.log_dir, exist_ok=True)
            plt.savefig(os.path.join(self.log_dir, 'training_rewards.png'))
            plt.show()

# -------------------------
# Training, evaluation, tuning, inference functions
# -------------------------

def run_ppo_training(env_params, hyperparams=None, total_timesteps=10000, log_dir='ppo_logs'):
    os.makedirs(log_dir, exist_ok=True)
    env = DummyVecEnv([lambda: MassageEnvGym(**env_params)])

    if hyperparams is None:
        hyperparams = {}

    model = PPO("MlpPolicy", env,
                verbose=1,
                tensorboard_log=log_dir,
                batch_size=hyperparams.get('batch_size', 64),
                gamma=hyperparams.get('gamma', 0.99),
                n_steps=hyperparams.get('n_steps', 2048),
                ent_coef=hyperparams.get('ent_coef', 0.0),
                learning_rate=hyperparams.get('learning_rate', 3e-4),
                clip_range=hyperparams.get('clip_range', 0.2),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
                )

    reward_callback = RewardLoggingCallback(check_freq=1000, log_dir=log_dir)
    best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    callback = CallbackList([reward_callback, best_model_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(os.path.join(log_dir, 'final_model'))

    env.close()

    print(f"PPO Training completed. Model saved to {os.path.join(log_dir, 'final_model.zip')}")

    return model

def evaluate_model(env, model, num_episodes=5):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards)
    return avg_reward, episode_rewards

def random_search_ppo(env_params, search_iters=5):
    best_reward = -float('inf')
    best_params = None
    rewards_per_trial = []

    for i in range(search_iters):
        hyperparams = {
            'learning_rate': 10 ** random.uniform(-5, -3),
            'gamma': random.uniform(0.9, 0.999),
            'n_steps': random.choice([512, 1024, 2048]),
            'ent_coef': random.uniform(0.0, 0.01),
            'clip_range': random.uniform(0.1, 0.3),
            'gae_lambda': random.uniform(0.8, 1.0),
            'batch_size': random.choice([32, 64, 128]),
            'max_grad_norm': random.uniform(0.3, 1.0)
        }
        print(f"PPO Trial {i+1}/{search_iters} with params: {hyperparams}")
        model = run_ppo_training(env_params, hyperparams, total_timesteps=10000, log_dir=f'ppo_tuning_trial_{i+1}')

        env = MassageEnvGym(**env_params)
        avg_reward, _ = evaluate_model(env, model, num_episodes=5)
        env.close()

        print(f"Average reward for trial {i+1}: {avg_reward:.3f}")
        rewards_per_trial.append(avg_reward)
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = hyperparams

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, search_iters + 1), rewards_per_trial, marker='o')
    plt.xlabel('Trial')
    plt.ylabel('Average Reward')
    plt.title('Hyperparameter Tuning Results')
    plt.grid(True)
    plt.savefig('ppo_tuning_results.png')
    plt.show()

    print(f"Hyperparameter tuning completed. Best hyperparameters: {best_params} with reward {best_reward:.3f}")
    return best_params, best_reward

import matplotlib.pyplot as plt
import numpy as np

def draw_data(Forces, armparts, bodyparts):
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.title(f'Massage Pressure: Mean {np.mean(Forces):.2f}')
    plt.plot(Forces)
    plt.subplot(312)
    plt.title('Arm Part')
    plt.plot(armparts)
    plt.subplot(313)
    plt.title('Body Part')
    plt.plot(bodyparts)
    plt.tight_layout()
    plt.show()

def run_ppo_inference_with_plots(env_params, model_path, render=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    env_params['render'] = render
    env = MassageEnvGym(**env_params)
    model = PPO.load(model_path, env=env)

    episode_rewards = []
    reward_components_history = []

    all_Forces = []
    all_armparts = []
    all_bodyparts = []

    max_episodes = 10
    max_steps_per_episode = env.max_steps

    try:
        for episode in range(max_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_reward_components = []
            print('Episode ', episode)

            Forces = []
            armparts = []
            bodyparts = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_reward_components.append(info)

                Forces.append(info.get('Forces', 0))
                armparts.append(info.get('armparts', 0))
                bodyparts.append(info.get('bodyparts', 0))

            episode_rewards.append(episode_reward)
            reward_components_history.append(episode_reward_components)

            all_Forces.extend(Forces)
            all_armparts.extend(armparts)
            all_bodyparts.extend(bodyparts)

    finally:
        env.close()

    avg_rewards_per_episode = [np.mean([step.get('total_reward', 0) for step in ep]) for ep in reward_components_history]

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Inference Episode Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig('ppo_inference_episode_rewards.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards_per_episode, label='Avg Step Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Step Reward')
    plt.title('Average Step Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('ppo_inference_avg_step_reward.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(episode_rewards, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Total Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Episode Rewards')
    plt.grid(True)
    plt.savefig('ppo_inference_reward_distribution.png')
    plt.show()
    plt.close()

    cumulative_rewards = np.cumsum(episode_rewards)
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_rewards, marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Episodes')
    plt.grid(True)
    plt.savefig('ppo_inference_cumulative_reward.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[[step.get('total_reward', 0) for step in ep] for ep in reward_components_history])
    plt.xlabel('Episode')
    plt.ylabel('Step Reward')
    plt.title('Boxplot of Step Rewards per Episode')
    plt.grid(True)
    plt.savefig('ppo_inference_step_rewards_boxplot.png')
    plt.show()
    plt.close()

    # Flatten or aggregate Forces, armparts, bodyparts to 1D lists of scalars
    flat_Forces = [np.sum(f) if isinstance(f, (list, np.ndarray)) else f for f in all_Forces]
    flat_armparts = [np.mean(a) if isinstance(a, (list, np.ndarray)) else a for a in all_armparts]
    flat_bodyparts = [np.mean(b) if isinstance(b, (list, np.ndarray)) else b for b in all_bodyparts]

    # Call draw_data function to plot Forces, armparts, bodyparts
    draw_data(flat_Forces, flat_armparts, flat_bodyparts)

    print("Inference plots saved.")

    return episode_rewards

# -------------------------
# Multiprocessing wrappers for GUI
# -------------------------

def training_process(params, queue):
    try:
        print("PPO Training process started")
        model = run_ppo_training(params['env_params'], params.get('hyperparams'), params.get('total_timesteps', 10000), params.get('log_dir', 'ppo_logs'))
        queue.put("Training completed")
    except Exception as e:
        queue.put(f"Training error: {e}")

def tuning_process(env_params, queue):
    try:
        best_params, best_reward = random_search_ppo(env_params)
        queue.put(f"Tuning completed. Best reward: {best_reward:.3f}")
        queue.put(f"Best params: {best_params}")
    except Exception as e:
        queue.put(f"Tuning error: {e}")

def inference_process(params, queue):
    try:
        render_flag = params.get('render', False)
        params['env_params']['render'] = render_flag
        total_reward = run_ppo_inference_with_plots(params['env_params'], params['model_path'], render=render_flag)
        queue.put(f"Inference completed.")
    except Exception as e:
        queue.put(f"Inference error: {e}")

# -------------------------
# GUI Control Panel
# -------------------------

class PPOControlApp:
    def __init__(self, root):
        self.root = root
        root.title("PPO Control Panel")

        tk.Label(root, text="Frequency").grid(row=0, column=0, sticky="w")

        self.freq_var = tk.DoubleVar(master=root, value=3.0)
        self.freq_value_label = tk.Label(root, text=f"{self.freq_var.get():.2f}")
        self.freq_value_label.grid(row=0, column=2, sticky="w")

        def on_freq_change(event):
            self.freq_value_label.config(text=f"{self.freq_var.get():.2f}")

        freq_slider = ttk.Scale(root, from_=0.1, to=10.0, variable=self.freq_var, orient='horizontal', command=lambda e: on_freq_change(e))
        freq_slider.grid(row=0, column=1, sticky="ew")


        tk.Label(root, text="Amplitude").grid(row=1, column=0, sticky="w")

        self.amp_var = tk.DoubleVar(master=root, value=0.01)
        self.amp_value_label = tk.Label(root, text=f"{self.amp_var.get():.3f}")
        self.amp_value_label.grid(row=1, column=2, sticky="w")

        def on_amp_change(event):
            self.amp_value_label.config(text=f"{self.amp_var.get():.3f}")

        amp_slider = ttk.Scale(root, from_=0.0, to=0.05, variable=self.amp_var, orient='horizontal', command=lambda e: on_amp_change(e))
        amp_slider.grid(row=1, column=1, sticky="ew")

        self.render_var = tk.BooleanVar(master=root, value=False)
        tk.Checkbutton(root, text="Render Training (GUI)", variable=self.render_var).grid(row=2, column=0, columnspan=2)

        self.status_var = tk.StringVar(master=root, value="Idle")
        tk.Label(root, textvariable=self.status_var).grid(row=5, column=0, columnspan=2)

        ttk.Button(root, text="Start Training", command=self.start_training).grid(row=3, column=0, pady=10)
        ttk.Button(root, text="Start Inference", command=self.start_inference).grid(row=3, column=1, pady=10)
        ttk.Button(root, text="Start Hyperparam Tuning", command=self.start_tuning).grid(row=4, column=0, columnspan=2, pady=10)

        root.columnconfigure(1, weight=1)

        self.process = None
        self.queue = Queue()
        self.root.after(100, self.check_queue)

    def get_params(self):
        return {
            'traj_step': 100,
            'frequency': self.freq_var.get(),
            'amp': self.amp_var.get(),
            'x_offset': 0.1,
            'z_offset_lower': 0.01,
            'z_offset_upper': 0.1,
            'region': 'lower_back',
            'force_limit': 30,
            'traj_type': 'sine',
            'massage_technique': 'normal',
            'max_steps': 300,
            'render': self.render_var.get()
        }

    def start_training(self):
        if self.process is None or not self.process.is_alive():
            self.status_var.set("Training started...")
            params = {'env_params': self.get_params(), 'total_timesteps': 10000, 'log_dir': 'ppo_logs'}
            self.process = Process(target=training_process, args=(params, self.queue))
            self.process.start()
        else:
            self.status_var.set("Process already running")

    def start_inference(self):
        if self.process is None or not self.process.is_alive():
            self.status_var.set("Inference started...")
            params = self.get_params()
            params = {'env_params': params, 'model_path': 'ppo_logs/final_model.zip', 'render': params.get('render', False)}
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

# -------------------------
# Main entry point
# -------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = PPOControlApp(root)
    root.mainloop()
