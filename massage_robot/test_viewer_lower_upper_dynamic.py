# File: massage_robot/test_viewer.py

import os
import time
import pybullet as p
import pybullet_data
import numpy as np
import configparser

from human.human_creation import HumanCreation
from human.human import Human
from human.furniture import Furniture

import robot_descriptions.ur5_description as ur5_desc
import robot_descriptions.robotiq_2f85_description as robotiq_desc
from generate_path import generate_trajectory


def main():
    # 1) Connect to PyBullet GUI
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)

    # 2) Setup URDF search paths
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    pkg_dir = os.path.dirname(__file__)
    urdf_dir = os.path.join(pkg_dir, "urdf")
    p.setAdditionalSearchPath(urdf_dir)

    # 3) Define start poses
    startPos = [-0.7, 0.1, 1.0]
    cubeStartingPose = [-1.3, 0.0, 0.5]
    startOrn = p.getQuaternionFromEuler([0, 0, 0])

    # 4) Load plane and bench
    planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    benchId = p.loadURDF(
        os.path.join(pybullet_data.getDataPath(), "cube.urdf"),
        cubeStartingPose, startOrn,
        useFixedBase=True
    )

    # 5) Load UR5 robot
    armId = p.loadURDF(
        ur5_desc.URDF_PATH,
        startPos, startOrn,
        useFixedBase=True
    )

    # 6) Robotiq 2F-85 gripper as end-effector
    wrist_link = p.getNumJoints(armId) - 2
    link_state = p.getLinkState(armId, wrist_link)
    wrist_pos, wrist_orn = link_state[0], link_state[1]

    gripper_id = p.loadURDF(
        robotiq_desc.URDF_PATH,
        wrist_pos, wrist_orn,
        useFixedBase=False
    )
    p.createConstraint(
        parentBodyUniqueId=armId,
        parentLinkIndex=wrist_link,
        childBodyUniqueId=gripper_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )

    # 7) Dynamic, non-colliding massage ball attached to gripper
    ball_vis = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.05,
        rgbaColor=[0.6, 0.2, 0.2, 1]
    )
    ball_id = p.createMultiBody(
        baseMass=0.2,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=ball_vis,
        basePosition=wrist_pos,
        baseOrientation=wrist_orn
    )
    p.createConstraint(
        parentBodyUniqueId=gripper_id,
        parentLinkIndex=-1,
        childBodyUniqueId=ball_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )

    # 8) Visualizer settings
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # 9) Create human and furniture
    before_bodies = p.getNumBodies()

    human_creation = HumanCreation(physicsClient, np_random=np.random)
    human_inst = Human([], controllable=False)

    configP = configparser.ConfigParser()
    configP.read(os.path.join(pkg_dir, 'human/config.ini'))
    def config(tag, section=None):
        return float(configP['' if section is None else section][tag])

    human_inst.init(
        human_creation, None, True,
        'random', 'random',
        config=config,
        id=physicsClient, np_random=np.random
    )

    furniture = Furniture()
    furniture.init(
        "bed", human_creation.directory,
        id=physicsClient, np_random=np.random,
        wheelchair_mounted=False
    )
    furniture.set_friction(furniture.base, friction=5)

    human_inst.setup_joints2(
        [], use_static_joints=True,
        reactive_force=None, reactive_gain=0.01
    )
    # give human some mass (will be zeroed out below)
    human_inst.set_mass(human_inst.base, mass=100)
    human_inst.set_base_velocity([0, 0, 0], [0, 0, 0])
    human_inst.set_base_pos_orient(
        [-0.15, 0.2, 0.95], [-np.pi/2, -np.pi, 0]
    )

    # identify the new human body ID
    after_bodies = p.getNumBodies()
    human_id = None
    for bid in range(before_bodies, after_bodies):
        info = p.getBodyInfo(bid)
        if b'human' in info[1].lower() or b'phantom' in info[1].lower():
            human_id = bid
            break
    if human_id is None:
        human_id = after_bodies - 1

    # 9a) Freeze the human: zero out mass on base and all links
    num_hj = p.getNumJoints(human_id)
    p.changeDynamics(human_id, -1, mass=0)
    for link_idx in range(num_hj):
        p.changeDynamics(human_id, link_idx, mass=0)

    # 10) Simulation parameters and trajectories
    TimeStep = 1/24.0
    p.setTimeStep(TimeStep)
    nArmJoints = p.getNumJoints(armId)

    traj_step = 100
    lower_start = np.array([-0.4, 0.3, 0.95])
    lower_end   = np.array([ 0.4, 0.3, 0.95])
    upper_start = np.array([-0.4, 0.3, 1.10])
    upper_end   = np.array([ 0.4, 0.3, 1.10])

    # Sine trajectories
    pnts_sine_lower = generate_trajectory(
        lower_start, lower_end,
        numSamples=traj_step, frequency=6, amp=0.035
    )
    pnts_sine_upper = generate_trajectory(
        upper_start, upper_end,
        numSamples=traj_step, frequency=6, amp=0.035
    )

    # Linear trajectories
    pnts_linear_lower = np.linspace(lower_start, lower_end, traj_step)
    pnts_linear_upper = np.linspace(upper_start, upper_end, traj_step)

    # Circular trajectories
    theta = np.linspace(0, 2*np.pi, traj_step)
    radius = 0.2
    center_lower = (lower_start + lower_end) / 2
    center_upper = (upper_start + upper_end) / 2
    pnts_circular_lower = np.array([
        center_lower + np.array([radius*np.cos(t), 0, radius*np.sin(t)])
        for t in theta
    ])
    pnts_circular_upper = np.array([
        center_upper + np.array([radius*np.cos(t), 0, radius*np.sin(t)])
        for t in theta
    ])

    # Print trajectory sizes
    print(f"Sine lower-back:    {len(pnts_sine_lower)} points")
    print(f"Linear lower-back:  {len(pnts_linear_lower)} points")
    print(f"Circular lower-back:{len(pnts_circular_lower)} points")
    print(f"Sine upper-back:    {len(pnts_sine_upper)} points")
    print(f"Linear upper-back:  {len(pnts_linear_upper)} points")
    print(f"Circular upper-back:{len(pnts_circular_upper)} points")

    # Combine with return paths
    all_trajs = []
    for seq in [
        pnts_sine_lower, pnts_linear_lower, pnts_circular_lower,
        pnts_sine_upper, pnts_linear_upper, pnts_circular_upper
    ]:
        closed = np.vstack((seq, seq[::-1]))
        all_trajs.append(closed)
    pntsAndReturn = np.concatenate(all_trajs, axis=0)

    # 11) Run simulation loop
    total_pts = pntsAndReturn.shape[0]
    for j in range(5000):
        p.stepSimulation()

        idx = j % total_pts
        target = pntsAndReturn[idx]
        
        JointPoses = p.calculateInverseKinematics(
            armId, wrist_link, target
        )
        p.setJointMotorControlArray(
            armId,
            jointIndices=list(range(1, nArmJoints - 3)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=JointPoses
        )
        time.sleep(TimeStep)

    # 12) Cleanup
    armPos, armOrn = p.getBasePositionAndOrientation(armId)
    print("Final base pose:", armPos, armOrn)
    p.disconnect()


if __name__ == "__main__":
    main()
