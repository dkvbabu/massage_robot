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


from robot_descriptions import ur5_description#shadow_hand_mj_description,ur5e_mj_description,

from generate_path import generate_trajectory


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

    # End effector
    #hand_spec = mujoco.MjSpec.from_file(shadow_hand_mj_description.MJCF_PATH_RIGHT)
    #arm_spec = mujoco.MjSpec.from_file(ur5_description.MJCF_PATH)
    #p.loadMJCF()

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


    # Create Human
    human_creation = HumanCreation(physicsClient,np_random=np.random)
    human_controllable_joint_indices = []#(human.right_arm_joints)]
    human_inst = Human(human_controllable_joint_indices, controllable=False)

    configP = configparser.ConfigParser()
    configP.read(os.path.join((os.path.dirname(os.path.realpath(__file__))), './human/config.ini'))
    def config(tag, section=None):
        return float(configP['' if section is None else section][tag])

    human_inst.init(human_creation, None,True, 'random', 'random',config=config,id=physicsClient,np_random=np.random)

    furniture = Furniture()
    furniture.init("bed", human_creation.directory,id=physicsClient, np_random=np.random, wheelchair_mounted= False)
    furniture.set_friction(furniture.base, friction=5)


    # Lock human joints and set velocities to 0
    joints_positions = []
    human_inst.setup_joints2(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
    human_inst.set_mass(human_inst.base, mass=100)
    human_inst.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
    #human_inst.reset_joints()


    #get controllable joints only in arm

    # y is head(+)-foot (-)
    # x is body width



    joints_positions = []#[(human_inst.j_right_shoulder_x, 30)]
    #human_inst.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
    human_inst.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2, -np.pi, 0])

    # Set time step
    TimeStep = 1/24.0
    p.setTimeStep(TimeStep)


    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)

    JointPoses = p.calculateInverseKinematics(armId, nArmJoints-2, [-0.4, 0.3, 1.05])#, 
                                #lowerLimits=[-np.pi]*(nArmJoints-1), upperLimits=[np.pi]*(nArmJoints-1))
    #breakpoint()


    traj_step = 100
    #pnts = generate_trajectory(np.array([-0.4, 0.3, 1.035]),np.array([0.4, 0.3, 1.035]),numSamples=traj_step,frequency=6,amp=0.035)
    pnts = generate_trajectory(np.array([-0.175, 0.3, 1.035]),np.array([0.05, 0.3, 1.035]),numSamples=traj_step,frequency=6,amp=0.035)

    pntsAndReturn = np.vstack((pnts,pnts[::-1]))
    #pntsAndReturn = np.clip(pntsAndReturn,a_max=10,a_min=1.01)

    print(f'Number of DOFs: {nArmJoints})')

    #breakpoint()
    #print(p.getAABB(human_inst.body))
    #((-0.30500000000000005, 0.07299999999999995, 0.823), (0.00500000000000006, 0.32700000000000007, 1.077))

    #JointPosesAll = p.calculateInverseKinematics2(armId, [nArmJoints-2]* traj_step,pnts.tolist())#, 

    #breakpoint()
    poses,vels,reactforces_6_val,torques =0,0,0,0

    out = list(p.getJointStates(armId,[i for i in range(nArmJoints)], physicsClientId=physicsClient))

    print(f'Arm poses:{[link[0] for link in out]})')

    #breakpoint()
    for j in range(5000):
        p.stepSimulation(physicsClientId=physicsClient)

        out = p.getClosestPoints(armId,human_inst.body,3,6,2)
        #pntsAndReturn[j%(2*traj_step),2] = max(out[0][6][2]-0.005,pntsAndReturn[j%(2*traj_step),2])
        pntsAndReturn[j%(2*traj_step),2] += (out[0][6][2]-0.005)
        pntsAndReturn[j%(2*traj_step),2] /= 2
        JointPoses = list(p.calculateInverseKinematics(armId, nArmJoints-2, pntsAndReturn[j%(2*traj_step)]))#, 

        #p.setJointMotorControlArray(armId, jointIndices=range(nArmJoints), controlMode=p.POSITION_CONTROL, targetPositions=[1,0,0,0,0,0,0,1,1,1])
        p.setJointMotorControlArray(armId, jointIndices=range(1,nArmJoints-3), controlMode=p.POSITION_CONTROL, targetPositions=JointPoses)
        time.sleep(TimeStep)

        #out = list(p.getJointStates(armId,[i for i in range(nArmJoints)], physicsClientId=physicsClient))

        #print(f'Arm poses:{[link[0] for link in out]})')

        if j%int(2/TimeStep):
            p1,p2 = p.getAABB(human_inst.body)
            pnts = generate_trajectory(np.array([p1[0]+0.125, 0.3, p2[2]-0.04]),np.array([p2[0]+0.1, 0.3, p2[2]-0.04]),numSamples=traj_step,frequency=6,amp=0.035)
            pntsAndReturn = np.vstack((pnts,pnts[::-1]))


            #breakpoint()


    armPos, armOrn = p.getBasePositionAndOrientation(armId)

    print(armPos,armOrn)

    p.disconnect()

    # 

    # Useful commands:
    # p.saveState
    # p.restoreState
    # p.setTimestep


    # p.getEulerFromQuaternion(quat) -->  [roll, pitch, yaw]

    # p.getJointInfo()

    # p.setJointMotorControl2() # make joint fixed max_force=0

    # p.setJointMotorControlMultiDof()

    # p.getJointStates() --> [pos, vel, forces, torque]
    # p.getLinkStates()

    # p.getDynamicsInfo() --> [mass, center of mass,friction]


    # p.startStateLogging(), p.stopStateLogging()


    # p.getBasePositionAndOrientation(bodyUniqueId)


    # p.computeViewMatrix()
    # p.computeProjectionMatrix()

    # p.getCameraImage() --> [rgb, depth]

    # p.getContactPoints()
    # p.getClosestPoints()

    # Lidar:
    # p.rayTest()
    # p.rayTestBatch()

    # self-collosion (URDF_USE_SELF_COLLOSION) in p.loadURDF

    # IK
    # p.calculateInverseDynamics() --> forces
    # p.calculateInverseKinematics() -->  joint angles

    # p.calculateInverseKinematics2() --> list of angles


    # Deformable

    # pybullet.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    # in .urdf file: <deformable name="y"> </deformable>





    # TODO
    # Timestep issue

if __name__ == "__main__":

    main()



