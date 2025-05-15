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
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # built-in assets
    pkg_dir = os.path.dirname(__file__)
    urdf_dir = os.path.join(pkg_dir, "urdf")
    p.setAdditionalSearchPath(urdf_dir)  # custom URDFs

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
    human_inst.set_mass(human_inst.base, mass=100)
    human_inst.set_base_velocity([0, 0, 0], [0, 0, 0])
    human_inst.set_base_pos_orient(
        [-0.15, 0.2, 0.95], [-np.pi/2, -np.pi, 0]
    )

    # 10) Simulation parameters and trajectory
    TimeStep = 1/24.0
    p.setTimeStep(TimeStep)
    nArmJoints = p.getNumJoints(armId)

    traj_step = 100
    pnts = generate_trajectory(
        np.array([-0.4, 0.3, 1.035]),
        np.array([0.4, 0.3, 1.035]),
        numSamples=traj_step, frequency=6, amp=0.035
    )
    pntsAndReturn = np.vstack((pnts, pnts[::-1]))

    # 11) Run simulation loop
    for j in range(5000):
        p.stepSimulation()
        target = pntsAndReturn[j % (2 * traj_step)]
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
