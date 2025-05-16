# File: massage_robot/test_viewer.py

import os
import time
import pybullet as p
import pybullet_data
import numpy as np
import configparser

from human.human_creation import HumanCreation
from human.furniture import Furniture

import robot_descriptions.ur5_description as ur5_desc
import robot_descriptions.robotiq_2f85_description as robotiq_desc
from generate_path import generate_trajectory

def main():
    # 1) Connect to PyBullet GUI
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)

    # 2) Setup URDF and mesh search paths
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # built-in assets
    pkg_dir = os.path.dirname(__file__)
    urdf_dir = os.path.join(pkg_dir, "urdf")
    meshes_dir = os.path.join(pkg_dir, "meshes")
    p.setAdditionalSearchPath(urdf_dir)    # for URDFs
    p.setAdditionalSearchPath(meshes_dir)  # for mesh files

    # 3) Define start poses
    startPos = [-0.7, 0.1, 1.0]
    cubeStartingPose = [-1.3, 0.0, 0.5]
    startOrn = p.getQuaternionFromEuler([0, 0, 0])

    # 4) Load plane and bench
    
    planeId = p.loadURDF(
    os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
    [0,0,0], [0,0,0,1],
    useFixedBase=True
    )
    benchId = p.loadURDF(
        os.path.join(pybullet_data.getDataPath(), "cube.urdf"),
        cubeStartingPose, startOrn,
        useFixedBase=True
    )


    # 5) Load UR5 robot
    armId = p.loadURDF(ur5_desc.URDF_PATH, startPos, startOrn, useFixedBase=True)

    # 6) Robotiq gripper as end-effector
    wrist_link = p.getNumJoints(armId) - 2
    link_state = p.getLinkState(armId, wrist_link)
    wrist_pos, wrist_orn = link_state[0], link_state[1]

    gripper_id = p.loadURDF(robotiq_desc.URDF_PATH, wrist_pos, wrist_orn, useFixedBase=False)
    p.createConstraint(parentBodyUniqueId=armId, parentLinkIndex=wrist_link,
                       childBodyUniqueId=gripper_id, childLinkIndex=-1,
                       jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                       parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])

    # 7) Massage ball attached to gripper
    ball_radius = 0.05
    ball_vis = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.6, 0.2, 0.2, 1])
    ball_id = p.createMultiBody(baseMass=0.2, baseCollisionShapeIndex=-1,
                                baseVisualShapeIndex=ball_vis, basePosition=wrist_pos, baseOrientation=wrist_orn)
    p.createConstraint(parentBodyUniqueId=gripper_id, parentLinkIndex=-1,
                       childBodyUniqueId=ball_id, childLinkIndex=-1,
                       jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                       parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])

    # 8) Visualizer settings
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # 9) Load realistic human phantom model
    phantom_start_pos = [-0.15, 0.2, 0.95]
    phantom_start_orn = p.getQuaternionFromEuler([np.deg2rad(10), 0, 0])  # 10° back raise
    
    human_phantom_path = os.path.join(urdf_dir, "human_phantom_realistic.urdf")
    human_phantom_id = p.loadURDF(
        human_phantom_path,
        phantom_start_pos,
        phantom_start_orn,
        useFixedBase=True
    )


    furniture = Furniture()
    furniture.init("bed", HumanCreation(physicsClient).directory, id=physicsClient, np_random=np.random, wheelchair_mounted=False)
    furniture.set_friction(furniture.base, friction=5)

    # 10) Simulation parameters
    TimeStep = 1/24.0
    p.setTimeStep(TimeStep)
    nArmJoints = p.getNumJoints(armId)

    traj_step = 100
    lower_start = np.array([-0.4, 0.3, 0.85])
    lower_end   = np.array([ 0.4, 0.3, 0.85])
    upper_start = np.array([-0.4, 0.3, 1.10])
    upper_end   = np.array([ 0.4, 0.3, 1.10])

    # generate trajectories
    pnts_sine_lower = generate_trajectory(lower_start, lower_end, numSamples=traj_step, frequency=6, amp=0.035)
    pnts_linear_lower = np.linspace(lower_start, lower_end, traj_step)
    theta = np.linspace(0, 2*np.pi, traj_step)
    radius = 0.2
    center_lower = (lower_start + lower_end) / 2
    pnts_circular_lower = np.array([center_lower + [radius*np.cos(t), 0, radius*np.sin(t)] for t in theta])

    pnts_sine_upper = generate_trajectory(upper_start, upper_end, numSamples=traj_step, frequency=6, amp=0.035)
    pnts_linear_upper = np.linspace(upper_start, upper_end, traj_step)
    center_upper = (upper_start + upper_end) / 2
    pnts_circular_upper = np.array([center_upper + [radius*np.cos(t), 0, radius*np.sin(t)] for t in theta])

    # print sizes
    print(f"Sine lower-back:    {len(pnts_sine_lower)} points")
    print(f"Linear lower-back:  {len(pnts_linear_lower)} points")
    print(f"Circular lower-back:{len(pnts_circular_lower)} points")
    print(f"Sine upper-back:    {len(pnts_sine_upper)} points")
    print(f"Linear upper-back:  {len(pnts_linear_upper)} points")
    print(f"Circular upper-back:{len(pnts_circular_upper)} points")

    segments = [
        ("lower-back", "sine", pnts_sine_lower),
        ("lower-back", "linear", pnts_linear_lower),
        ("lower-back", "circular", pnts_circular_lower),
        ("upper-back", "sine", pnts_sine_upper),
        ("upper-back", "linear", pnts_linear_upper),
        ("upper-back", "circular", pnts_circular_upper)
    ]
    labeled, boundaries, idx_acc = [], [], 0
    for label, pattern, seq in segments:
        closed = np.vstack((seq, seq[::-1]))
        labeled.append((label, pattern, closed))
        boundaries.append((idx_acc, idx_acc+len(closed), label, pattern))
        idx_acc += len(closed)
    pntsAndReturn = np.concatenate([c for _,_,c in labeled], axis=0)
    total_pts = pntsAndReturn.shape[0]

    # 11) Sim loop
    for step in range(5000):
        p.stepSimulation()
        idx = step % total_pts
        raw_target = pntsAndReturn[idx]

        # surface contact
        ray_start = [raw_target[0], raw_target[1], raw_target[2]+0.5]
        ray_end   = [raw_target[0], raw_target[1], raw_target[2]-0.5]
        hits = p.rayTest(ray_start, ray_end)
        surface_z = hits[0][3][2] if hits[0][0]==human_phantom_id else raw_target[2]

        target = raw_target.copy()
        target[2] = surface_z + ball_radius

        for start,end,label,pattern in boundaries:
            if idx==start:
                print(f"Executing '{pattern}' pattern for {label.replace('-',' ')} ({end-start} steps)")
                break

        JointPoses = p.calculateInverseKinematics(armId, wrist_link, target)
        p.setJointMotorControlArray(armId, list(range(1, nArmJoints-3)),
                                    p.POSITION_CONTROL, JointPoses)
        time.sleep(TimeStep)

    # 12) Cleanup
    armPos, armOrn = p.getBasePositionAndOrientation(armId)
    print("Final base pose:", armPos, armOrn)
    p.disconnect()

if __name__=="__main__":
    main()
