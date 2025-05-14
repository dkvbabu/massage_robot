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



def main():

    physicsClient = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

    p.setGravity(0,0,-10)

    startPos = [-1,0.1,1.0]
    cubeStartingPose = [-1.3,0.0,0.5]

    startOrientation = p.getQuaternionFromEuler([0,0,0])


    planeId = p.loadURDF("plane.urdf")
    armId = p.loadURDF(ur5_description.URDF_PATH,startPos, startOrientation)
    cubeId = p.loadURDF("cube.urdf",cubeStartingPose, startOrientation)
    # Make cube completely static
    p.changeDynamics(cubeId, -1, mass=0)
    p.changeDynamics(cubeId, -1, lateralFriction=5.0)
    p.changeDynamics(cubeId, -1, rollingFriction=5.0)
    p.changeDynamics(cubeId, -1, spinningFriction=5.0)
    p.changeDynamics(cubeId, -1, linearDamping=1.0)
    p.changeDynamics(cubeId, -1, angularDamping=1.0)

    # End effector
    #hand_spec = mujoco.MjSpec.from_file(shadow_hand_mj_description.MJCF_PATH_RIGHT)
    #arm_spec = mujoco.MjSpec.from_file(ur5_description.MJCF_PATH)
    #p.loadMJCF()

    # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


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
    # Make furniture completely static
    p.changeDynamics(furniture.base, -1, mass=0)
    furniture.set_friction(furniture.base, friction=5)
    # Lock all joints of the furniture
    for joint in range(p.getNumJoints(furniture.base)):
        p.setJointMotorControl2(furniture.base, joint, p.POSITION_CONTROL, targetPosition=0, force=0)
        p.changeDynamics(furniture.base, joint, jointLowerLimit=0, jointUpperLimit=0)


    # Lock human joints and set velocities to 0
    joints_positions = []
    human_inst.setup_joints2(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
    human_inst.set_mass(human_inst.base, mass=0)
    human_inst.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
    #human_inst.reset_joints()


    joints_positions = []#[(human_inst.j_right_shoulder_x, 30)]
    #human_inst.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
    human_inst.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2, -np.pi, 0])

    # Set time step
    TimeStep = 1/24.0
    p.setTimeStep(TimeStep)

    # --- Add sliders for robot base position and orientation ---
    pos_sliders = []
    ori_sliders = []
    pos_labels = ['X', 'Y', 'Z']
    ori_labels = ['Yaw', 'Pitch', 'Roll']
    pos_defaults = [-1.0, 0.1, 1.0]
    ori_defaults = [0.0, 0.0, 0.0]
    for i, label in enumerate(pos_labels):
        slider = p.addUserDebugParameter(f'{label} Position', -2, 2, pos_defaults[i])
        pos_sliders.append(slider)
    for i, label in enumerate(ori_labels):
        slider = p.addUserDebugParameter(f'{label}', -3.14, 3.14, ori_defaults[i])
        ori_sliders.append(slider)

    # --- Add sliders for robot arm joints ---
    num_joints = p.getNumJoints(armId)
    joint_sliders = []
    for i in range(num_joints):
        info = p.getJointInfo(armId, i)
        joint_name = info[1].decode('utf-8')
        lower = info[8] if info[8] < info[9] else -3.14
        upper = info[9] if info[9] > info[8] else 3.14
        slider = p.addUserDebugParameter(f'{joint_name}', lower, upper, 0)
        joint_sliders.append(slider)

    for _ in range(5000):
        # Read base position/orientation sliders
        pos = [p.readUserDebugParameter(s) for s in pos_sliders]
        yaw, pitch, roll = [p.readUserDebugParameter(s) for s in ori_sliders]
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        quat = [qx, qy, qz, qw]
        p.resetBasePositionAndOrientation(armId, pos, quat)
        # Read joint sliders and set joint positions
        for i, slider in enumerate(joint_sliders):
            target = p.readUserDebugParameter(slider)
            p.setJointMotorControl2(armId, i, p.POSITION_CONTROL, targetPosition=target)
        p.stepSimulation(physicsClientId=physicsClient)
        time.sleep(TimeStep)

    cubePos, cubeOrn = p.getBasePositionAndOrientation(planeId)

    print(cubePos,cubeOrn)

    p.disconnect()


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

    # Make human static
    human_inst.set_mass(human_inst.base, mass=0)
    # Make table (furniture) static
    p.changeDynamics(furniture.base, -1, mass=0)

    # After loading the table, make it static if present
    # Find all bodies and set mass to zero for the table (excluding bed and human)
    for body_id in range(p.getNumBodies()):
        body_name = p.getBodyInfo(body_id)[0].decode('utf-8')
        if 'table' in body_name.lower():
            p.changeDynamics(body_id, -1, mass=0)

    # Make the cube static
    p.changeDynamics(cubeId, -1, mass=0)

if __name__ == "__main__":

    main()


