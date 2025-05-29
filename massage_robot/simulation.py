import tkinter as tk
from tkinter import messagebox
import threading
import pybullet as p
import time, os
import pybullet_data
import numpy as np
import configparser
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import datetime
import seaborn as sns  # For boxplot

from human.human_creation import HumanCreation
from human import agent, human
from human.agent import Agent
from human.human import Human
from human.furniture import Furniture

from robot_descriptions import ur5_description

logging.basicConfig(filename='simulation_errors.log', level=logging.ERROR)

# --- Simulation functions from simulation_1.py ---

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
    t = np.linspace(0, 4 * np.pi, numSamples)
    x_vals = center[0] + radius * np.cos(t)
    y_vals = center[1] + radius * np.sin(t)
    z_vals = center[2] + amp * np.sin(frequency * t)
    return np.vstack((x_vals, y_vals, z_vals)).T

def generate_pressure_trajectory(center, numSamples):
    pnts = np.tile(center, (numSamples, 1))
    return pnts

def update_trajectory(human_body, traj_step, frequency, amp, x_offset, region, z_offset_lower, z_offset_upper, traj_type, massage_technique):
    p1, p2 = p.getAABB(human_body)
    inward_offset_y = 0.02
    inward_offset_x = 0.1
    clearance_z = 0.15
    if region == 'lower_back':
        z_pos = p1[2] + z_offset_lower
        y_pos = 0.03 - inward_offset_y
    elif region == 'upper_back':
        z_pos = p2[2] - z_offset_upper
        y_pos = 0.4 - inward_offset_y
    else:
        z_pos = (p1[2] + p2[2]) / 2
        y_pos = 0.3 - inward_offset_y
    if region == 'upper_back':
        z_pos += clearance_z
    center_x = (p1[0] + p2[0]) / 2 - inward_offset_x
    start = np.array([center_x - x_offset, y_pos, z_pos])
    end = np.array([center_x + x_offset, y_pos, z_pos])
    center = np.array([center_x, y_pos, z_pos])
    if massage_technique == 'kneading':
        radius = inward_offset_x
        pnts = generate_kneading_trajectory(center, radius, traj_step, frequency, amp)
    elif massage_technique == 'pressure':
        center = center - 0.05
        pnts = generate_pressure_trajectory(center, traj_step)
    else:
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
                         frequency, amp, x_offset, z_offset_lower, z_offset_upper, region, force, traj_type, massage_technique,
                         sphereId, humanBodyId, end_effector_link_index):
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
            contact_points = p.getContactPoints(bodyA=armId, bodyB=humanBodyId, linkIndexA=end_effector_link_index)
            if contact_points:
                print(f"Contact detected at step {j}, points: {len(contact_points)}")
                for contact in contact_points:
                    print(f"Force: {contact[9]:.3f}, Pos on ee_link: {contact[5]}, Pos on human: {contact[6]}")
            else:
                print(f"No contact at step {j}")
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
            upper_back_orientation = p.getQuaternionFromEuler([np.pi, 0, np.pi/2])
            if region == 'lower_back':
                orientation = p.getQuaternionFromEuler([0, np.pi, 0])
            elif region == 'upper_back':
                orientation = upper_back_orientation
            else:
                orientation = p.getQuaternionFromEuler([0, np.pi, 0])
            JointPoses = list(p.calculateInverseKinematics(armId, nArmJoints - 1, pntsAndReturn[j % (2 * traj_step)], orientation))
            print('jointposes', JointPoses)
            print('jointindices', range(1, nArmJoints - 1))
            p.setJointMotorControlArray(armId, jointIndices=range(1, nArmJoints - 1), controlMode=p.POSITION_CONTROL,
                                        targetPositions=JointPoses, forces=force * np.ones_like(JointPoses))
            time.sleep(1 / 24.0)
            if j % int(2 / (1 / 24.0)) == 0:
                pntsAndReturn = update_trajectory(human_inst.body, traj_step, frequency, amp,
                                                  x_offset, region, z_offset_lower, z_offset_upper, traj_type, massage_technique)
        except Exception as e:
            logging.error(f"Error at step {j} with params {params}: {e}")
            print('errorinloop', e)
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
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    planeId = p.loadURDF("plane.urdf")
    urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "urdf/ur5_robot_mod1.urdf")
    try:
        armId = p.loadURDF(urdf_path, startPos, startOrientation)
    except Exception as e:
        print("Failed to load URDF:", e)
    cubeStartingPose = [-1.3, 0.0, 0.5]
    cubeId = p.loadURDF("cube.urdf", cubeStartingPose, startOrientation)
    TimeStep = 1 / 24.0
    p.setTimeStep(TimeStep)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    human_inst = load_scene(physicsClient)
    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)
    ee_link_index = None
    for i in range(nArmJoints):
        info = p.getJointInfo(armId, i)
        link_name = info[12].decode('utf-8')
        if link_name == 'ee_link':
            ee_link_index = i
            break
    if ee_link_index is None:
        raise ValueError("ee_link not found in robot model")
    humanBodyId = human_inst.base
    print("Human body ID:", humanBodyId)
    print('eelinkindex', ee_link_index)
    num_joints_arm = p.getNumJoints(armId, physicsClientId=physicsClient)
    for joint_idx in range(-1, num_joints_arm):
        p.setCollisionFilterPair(armId, planeId, joint_idx, -1, 0, physicsClientId=physicsClient)
        p.setCollisionFilterPair(armId, humanBodyId, joint_idx, -1, 0, physicsClientId=physicsClient)
    num_joints_human = p.getNumJoints(humanBodyId, physicsClientId=physicsClient)
    for joint_idx in range(-1, num_joints_human):
        p.setCollisionFilterPair(humanBodyId, planeId, joint_idx, -1, 0, physicsClientId=physicsClient)
    for link_idx in range(-1, nArmJoints):
        contact_info = p.getContactPoints(bodyA=armId, bodyB=humanBodyId, linkIndexA=link_idx)
        if contact_info:
            joint_name = p.getJointInfo(armId, link_idx)[12].decode() if link_idx != -1 else 'base'
            print(f"Link {link_idx} ({joint_name}) is colliding")
    traj_step = 100
    sphereId = None
    results = test_settings_single(physicsClient, human_inst, armId, nArmJoints, traj_step,
                                   frequency, amplitude, x_offset, z_offset_lower, z_offset_upper,
                                   region, force, traj_type, massage_technique, sphereId, humanBodyId, ee_link_index)
    report_results(results)
    p.disconnect()

# --- GUI code from simulation_gui.py ---

root = tk.Tk()
root.title("Simulation Parameters")

tk.Label(root, text="Frequency:").grid(row=0, column=0, sticky='w', padx=5, pady=4)
freq_entry = tk.Entry(root)
freq_entry.insert(0, "6")
freq_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=4)

tk.Label(root, text="Amplitude:").grid(row=1, column=0, sticky='w', padx=5, pady=4)
amp_entry = tk.Entry(root)
amp_entry.insert(0, "0.01")
amp_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=4)

tk.Label(root, text="X Offset:").grid(row=2, column=0, sticky='w', padx=5, pady=4)
x_offset_entry = tk.Entry(root)
x_offset_entry.insert(0, "0.1")
x_offset_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=4)

tk.Label(root, text="Z Offset Lower:").grid(row=3, column=0, sticky='w', padx=5, pady=4)
z_offset_lower_entry = tk.Entry(root)
z_offset_lower_entry.insert(0, "0.01")
z_offset_lower_entry.grid(row=3, column=1, sticky='ew', padx=5, pady=4)

tk.Label(root, text="Z Offset Upper:").grid(row=4, column=0, sticky='w', padx=5, pady=4)
z_offset_upper_entry = tk.Entry(root)
z_offset_upper_entry.insert(0, "0.1")
z_offset_upper_entry.grid(row=4, column=1, sticky='ew', padx=5, pady=4)

tk.Label(root, text="Target Region:").grid(row=5, column=0, sticky='w', padx=5, pady=4)
region_var = tk.StringVar(value='lower_back')
region_menu = tk.OptionMenu(root, region_var, 'lower_back', 'upper_back')
region_menu.grid(row=5, column=1, sticky='ew', padx=5, pady=4)

def validate_force_input(new_value):
    if new_value == "":
        return True
    try:
        val = float(new_value)
        return 0 <= val <= 100
    except ValueError:
        return False

vcmd = (root.register(validate_force_input), '%P')
tk.Label(root, text="Force:").grid(row=6, column=0, sticky='w', padx=5, pady=4)
force_entry = tk.Entry(root, validate='key', validatecommand=vcmd)
force_entry.insert(0, "100")
force_entry.grid(row=6, column=1, sticky='ew', padx=5, pady=4)

tk.Label(root, text="Trajectory Type:").grid(row=7, column=0, sticky='w', padx=5, pady=4)
traj_type_var = tk.StringVar(value='sine')
traj_type_menu = tk.OptionMenu(root, traj_type_var, 'sine', 'circular', 'linear')
traj_type_menu.grid(row=7, column=1, sticky='ew', padx=5, pady=4)

tk.Label(root, text="Massage Technique:").grid(row=8, column=0, sticky='w', padx=5, pady=4)
massage_technique_var = tk.StringVar(value='normal')
massage_technique_menu = tk.OptionMenu(root, massage_technique_var, 'normal', 'kneading', 'pressure')
massage_technique_menu.grid(row=8, column=1, sticky='ew', padx=5, pady=4)

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
    if not (0 <= force <= 100):
        messagebox.showerror("Invalid force", "Force must be between 0 and 100 inclusive.")
        return
    start_button.config(state=tk.DISABLED)
    threading.Thread(target=lambda: [run_simulation_with_params(freq, amp, x_offset,
                                                               z_offset_lower, z_offset_upper, region, force, traj_type, massage_technique),
                                    start_button.config(state=tk.NORMAL)]).start()

button_width = 20
button_padx = 5
button_pady = 5

start_button = tk.Button(root, text="Start Simulation", command=start_simulation, width=button_width)
start_button.grid(row=0, column=2, columnspan=2, padx=button_padx, pady=button_pady)

# You can add other buttons and their callbacks similarly, calling functions from this script

root.mainloop()
