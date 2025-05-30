import os
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import time

# Helper functions for trajectory generation (copy from your original script)
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

# Load scene helper function (simplified, adjust imports as needed)
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

# Visualization helpers (optional, copy from your original script)
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

        self.delta_angle = 0.01  # max joint increment per step

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

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            # Set other random seeds if needed

        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)

        startPos = [-0.7, 0.1, 1.0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "urdf/ur5_robot_mod1.urdf")
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

        observation = self._get_state()
        info = {}  # You can add useful info here if needed

        return observation, info

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
        print(f"Step {self.current_step}: action={action}, reward={reward}")
        done = self.current_step >= self.max_steps

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

        # Gymnasium API expects two booleans: terminated and truncated
        terminated = done
        truncated = False

        # Include these in info dictionary
        info = self.last_reward_components.copy() if hasattr(self, 'last_reward_components') else {}
        info.update({
            'Forces': Forces,
            'armparts': armparts,
            'bodyparts': bodyparts
        })
        time.sleep(0.02)
        return next_state, reward, terminated, truncated, info

    def _compute_reward(self):
        contact_points = p.getContactPoints(self.armId, self.human_inst.body, linkIndexA=7, physicsClientId=self.physicsClient)
        total_force = sum([cp[9] for cp in contact_points]) if contact_points else 0.0

        target_min, target_max = 15, 100

        reward_contact = 0.3
        reward_force = 0.0
        reward_penalty_force = 0.0
        reward_velocity_penalty = 0.0
        reward_no_contact_penalty = 0.0

        if contact_points:
            if target_min <= total_force <= target_max:
                center = (target_min + target_max) / 2
                width = (target_max - target_min) / 2
                reward_force = 1.0 - abs(total_force - center) / width
            else:
                dist = min(abs(total_force - target_min), abs(total_force - target_max))
                reward_force = -dist / target_max * 0.3

            if total_force > target_max * 2:
                reward_penalty_force = -1.0
        else:
            reward_no_contact_penalty = -0.02

        joint_states = p.getJointStates(self.armId, self.joint_indices, physicsClientId=self.physicsClient)
        joint_velocities = np.array([state[1] for state in joint_states])
        velocity_change = np.sum(np.abs(joint_velocities - self.prev_joint_velocities))
        reward_velocity_penalty = -velocity_change * 0.005
        self.prev_joint_velocities = joint_velocities

        total_reward = reward_contact + reward_force + reward_penalty_force + reward_velocity_penalty + reward_no_contact_penalty

        # Scale reward to amplify learning signal
        scale_factor = 10.0
        total_reward *= scale_factor

        # Optional: clip reward to avoid extreme values (comment out if unstable)
        total_reward = max(min(total_reward, 10.0), -10.0)

        self.last_reward_components = {
            'reward_contact': reward_contact,
            'reward_force': reward_force,
            'reward_penalty_force': reward_penalty_force,
            'reward_velocity_penalty': reward_velocity_penalty,
            'reward_no_contact_penalty': reward_no_contact_penalty,
            'total_reward': total_reward
        }

        return total_reward

    def close(self):
        if self.physicsClient is not None:
            p.disconnect(self.physicsClient)
            self.physicsClient = None

    def disable_unwanted_collisions(self):
        if not hasattr(self, 'end_effector_link_index'):
            self.end_effector_link_index = 7  # or your actual ee link index

        num_joints = p.getNumJoints(self.armId, physicsClientId=self.physicsClient)
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
