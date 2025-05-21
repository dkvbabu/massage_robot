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

    p.disconnect()


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

root.mainloop()
