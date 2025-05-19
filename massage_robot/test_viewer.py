import pybullet as p
import time,os
import pybullet_data
import numpy as np

import configparser

from human.human_creation import HumanCreation
from human import agent, human
from human.agent import Agent
from human.human import Human
from human.furniture import Furniture

import matplotlib.pyplot as plt

from robot_descriptions import ur5_description#shadow_hand_mj_description,ur5e_mj_description,

from generate_path import generate_trajectory


def draw_data(Forces,armparts,bodyparts):

    # Show pressure Profile
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

    # Create Human

    human_creation = HumanCreation(physicsClient,np_random=np.random)
    human_controllable_joint_indices = []#(human.right_arm_joints)]
    human_inst = Human(human_controllable_joint_indices, controllable=False)

    configP = configparser.ConfigParser()
    configP.read(os.path.join((os.path.dirname(os.path.realpath(__file__))), './human/config.ini'))
    def config(tag, section=None):
        return float(configP['' if section is None else section][tag])

    human_inst.init(human_creation, None,True, 'random', 'random',config=config,id=physicsClient,np_random=np.random)

    # Add bed
    furniture = Furniture()
    furniture.init("bed", human_creation.directory,id=physicsClient, np_random=np.random, wheelchair_mounted= False)
    furniture.set_friction(furniture.base, friction=5)


    # Setup Human
    # Lock human joints and set velocities to 0
    joints_positions = []
    human_inst.setup_joints2(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
    human_inst.set_mass(human_inst.base, mass=100)
    human_inst.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
    #human_inst.reset_joints()
    # get controllable joints only in arm
    # y is head(+)-foot (-)
    # x is body width

    joints_positions = []#[(human_inst.j_right_shoulder_x, 30)]
    #human_inst.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
    human_inst.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2, -np.pi, 0])

    return human_inst



def main():

    physicsClient = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)

    startPos = [-0.7,0.1,1.0]
    cubeStartingPose = [-1.3,0.0,0.5]
    startOrientation = p.getQuaternionFromEuler([0,0,0])

    planeId = p.loadURDF("plane.urdf")
    armId = p.loadURDF(ur5_description.URDF_PATH,startPos, startOrientation)
    cubeId = p.loadURDF("cube.urdf",cubeStartingPose, startOrientation)

    TimeStep = 1/24.0
    p.setTimeStep(TimeStep)

    # Load End effector TODO
    #hand_spec = mujoco.MjSpec.from_file(shadow_hand_mj_description.MJCF_PATH_RIGHT)
    #arm_spec = mujoco.MjSpec.from_file(ur5_description.MJCF_PATH)
    #p.loadMJCF()

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    human_inst = load_scene(physicsClient)

    # Setup Path
    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)

    JointPoses = p.calculateInverseKinematics(armId, nArmJoints-2, [-0.4, 0.3, 1.05]) 
    traj_step = 100
    pnts = generate_trajectory(np.array([-0.175, 0.3, 1.035]),np.array([0.05, 0.3, 1.035]),numSamples=traj_step,frequency=6,amp=0.035)

    pntsAndReturn = np.vstack((pnts,pnts[::-1]))
    print(f'Number of DOFs: {nArmJoints})')

    #breakpoint()
    #print(p.getAABB(human_inst.body))

    Forces = []
    bodyparts = []
    armparts = []

    for j in range(1200):

        p.stepSimulation(physicsClientId=physicsClient)
        out = p.getClosestPoints(armId,human_inst.body,3,5)
        out_1 = p.getContactPoints(armId,human_inst.body)

        if len(out_1):

            bodyparts.append(out_1[0][4])
            armparts.append(out_1[0][3])
            Forces.append(out_1[0][9]) # normal force

        else:
            bodyparts.append(0)
            armparts.append(0)
            Forces.append(0)

        # Augment Path
        pntsAndReturn[j%(2*traj_step),2] += 2*(out[0][6][2]-0.005)
        pntsAndReturn[j%(2*traj_step),2] /= 3

        JointPoses = list(p.calculateInverseKinematics(armId, nArmJoints-2, pntsAndReturn[j%(2*traj_step)])) 

        p.setJointMotorControlArray(armId, jointIndices=range(1,nArmJoints-3), controlMode=p.POSITION_CONTROL, 
                                    targetPositions=JointPoses,forces=100*np.ones_like(JointPoses))
        time.sleep(TimeStep)

        if j%int(2/TimeStep):
            # Update Path
            p1,p2 = p.getAABB(human_inst.body)
            pnts = generate_trajectory(np.array([p1[0]+0.125, 0.3, p2[2]-0.04]),np.array([p2[0]+0.1, 0.3, p2[2]-0.04]),numSamples=traj_step,frequency=6,amp=0.035)
            pntsAndReturn = np.vstack((pnts,pnts[::-1]))



    armPos, armOrn = p.getBasePositionAndOrientation(armId)

    print(armPos,armOrn)

    p.disconnect()

    draw_data(Forces,armparts,bodyparts)


if __name__ == "__main__":

    main()



