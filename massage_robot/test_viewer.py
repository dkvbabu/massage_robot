import pybullet as p
import time,os
import pybullet_data
import numpy as np
import cv2

from utils import get_extrinsics, get_intrinsics, get_extrinsics2

import configparser

from human.human_creation import HumanCreation
from human import agent, human
from human.agent import Agent
from human.human import Human
from human.furniture import Furniture

import matplotlib.pyplot as plt

# from massage_robot.robot_descriptions import ur5_description

from generate_path import generate_trajectory
from scipy.spatial.transform import Rotation




def draw_data(Forces,armparts,bodyparts,old_path,new_path,actual_path,eeId=7):

    # Show pressure Profile
    plt.subplot(411)
    plt.title(f'Massage Pressure: Mean {np.mean(Forces):.2f}')
    plt.ylabel('Newton')
    plt.plot(Forces)
    plt.subplot(412)
    plt.title(f'Arm Part: {np.round(np.mean([armid==eeId for armid in armparts]),2)*100}% End Effector')
    plt.plot(armparts)
    plt.subplot(413)
    plt.title(f'Body Part:  {np.round(np.mean([b==(-1) for b in bodyparts]),2)}')
    plt.plot(bodyparts)
    plt.subplot(414)
    plt.title('Massage Paths')
    if len(old_path): plt.plot(old_path,label='sinusoidal commands')
    if len(new_path): plt.plot(new_path,label='surface level')
    if len(actual_path): plt.plot(actual_path,label='modified path')
    plt.ylabel('meters')
    plt.legend()
    plt.show()

def arm_camera(projection_matrix,armId=2):

    # Center of mass position and orientation (of link-6)
    com_p, com_o, _, _, _, _ = p.getLinkState(armId, 4)
    rot_matrix = p.getMatrixFromQuaternion(com_o)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (0, 0, 1) # z-axis
    init_up_vector = (0, 1, 0) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(1000, 1000, view_matrix, projection_matrix)

    return np.array(img[2]).reshape(1000,1000,-1)



def load_scene(physicsClient):

    # Create Human

    human_creation = HumanCreation(physicsClient,np_random=np.random)
    human_controllable_joint_indices = []#(human.right_arm_joints)]
    human_inst = Human(human_controllable_joint_indices, controllable=False)


    gender = 'male'
    human_inst.init(human_creation, None,True, 'none', gender,config=None,id=physicsClient,
                    np_random=np.random,mass=90,radius_scale=1.2, height_scale=1.1)

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

    startPos = [-0.8,0.1,1.0]
    cubeStartingPose = [-1.3,0.0,0.5]
    startOrientation = p.getQuaternionFromEuler([0,0,0])

    planeId = p.loadURDF("plane.urdf")
    #armId = p.loadURDF(ur5_description.URDF_PATH,startPos, startOrientation)
    armId = p.loadURDF('urdf/ur5_robot.urdf',startPos, startOrientation)
    cubeId = p.loadURDF("cube.urdf",cubeStartingPose, startOrientation)

    p.resetJointState(armId,1,-0.4)
    p.resetJointState(armId,2,-0.9)
    p.resetJointState(armId,3,1)
    p.resetJointState(armId,4,-2.0)
    p.resetJointState(armId,5,-1.5)
    #p.resetJointState(armId,6,-1.7)
    #p.resetJointState(armId,2,-1)
    #p.resetJointState(armId,3,0.2)
    TimeStep = 1/24.0
    p.setTimeStep(TimeStep)

    # Load End effector TODO
    #hand_spec = mujoco.MjSpec.from_file(shadow_hand_mj_description.MJCF_PATH_RIGHT)
    #arm_spec = mujoco.MjSpec.from_file(ur5_description.MJCF_PATH)
    #p.loadMJCF()

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


    # Good Cam 1
    #(1024, 768, (-0.9937682151794434, -0.07253935188055038, 0.08463338017463684, 0.0, 0.1114664301276207, -0.6467175483703613, 0.754540741443634, 0.0, 0.0, 0.7592723965644836, 0.6507730484008789, 0.0, -0.18283233046531677, -0.3637703061103821, -1.4271166324615479, 1.0), (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (-0.08463338017463684, -0.754540741443634, -0.6507730484008789), (-26500.486328125, 2972.43798828125, 0.0), (-1450.786865234375, -12934.3505859375, 15185.4462890625), 173.6001434326172, -40.599910736083984, 1.200002670288086, (-0.18885917961597443, -0.04351018741726875, 0.42400023341178894))

    # Good Cam 2
    #(1024, 768, (-0.041878849267959595, -0.6659466028213501, 0.7448229789733887, 0.0, 0.9991226196289062, -0.027913566678762436, 0.0312197208404541, 0.0, 0.0, 0.7454769015312195, 0.6665313839912415, 0.0, -0.14472033083438873, -0.7917931079864502, -1.0678036212921143, 1.0), (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (-0.7448229789733887, -0.0312197208404541, -0.6665313839912415), (-1116.7694091796875, 26643.2734375, 0.0), (-13318.9326171875, -558.2713623046875, 14909.5400390625), 92.40018463134766, -41.799922943115234, 1.0, (-0.4828510880470276, 0.12460841238498688, 0.6354567408561707))

    # Up Cam 3
    #(1024, 768, (-0.0017075804062187672, -0.9999969005584717, 0.001745292916893959, 0.0, 0.9999985098838806, -0.0017075776122510433, 2.9802324661432067e-06, 0.0, 0.0, 0.0017452954780310392, 0.9999983906745911, 0.0, -0.1250828057527542, -0.2837464511394501, -1.9349620342254639, 1.0), (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (-0.001745292916893959, -2.9802324661432067e-06, -0.9999983906745911), (-45.53548049926758, 26666.62890625, 0.0), (-19999.939453125, -34.15155792236328, 34.90591049194336), 90.0999984741211, -89.9000015258789, 1.2999999523162842, (-0.2828510105609894, 0.12460000067949295, 0.6354566812515259))



    human_inst = load_scene(physicsClient)

    # Reset the camera
    p.resetDebugVisualizerCamera(
        cameraYaw= 173.6001434326172,
        cameraPitch=-40.599910736083984,
        cameraDistance=1.200002670288086,
        cameraTargetPosition=(-0.18885917961597443, -0.04351018741726875, 0.42400023341178894),
    )
    #p.resetDebugVisualizerCamera(cameraYaw= 92.4,cameraPitch=-41.8,cameraDistance=1.0,cameraTargetPosition=(-0.4828510, 0.12460, 0.6354567),)

    # Up view
    p.resetDebugVisualizerCamera(cameraYaw= 90.1,cameraPitch=-89.9,cameraDistance=1.3,cameraTargetPosition=(-0.2828510, 0.12460, 0.6354567),)


    cam1_vmat = (-0.9937682151794434, -0.07253935188055038, 0.08463338017463684, 0.0, 0.1114664301276207, -0.6467175483703613, 0.754540741443634, 0.0, 0.0, 0.7592723965644836, 0.6507730484008789, 0.0, -0.18283233046531677, -0.3637703061103821, -1.4271166324615479, 1.0)

    cam1_pmat = (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0)

    cam_upview = (-0.0017075804062187672, -0.9999969005584717, 0.001745292916893959, 0.0, 0.9999985098838806, -0.0017075776122510433, 2.9802324661432067e-06, 0.0, 0.0, 0.0017452954780310392, 0.9999983906745911, 0.0, -0.1250828057527542, -0.2837464511394501, -1.9349620342254639, 1.0)

    cam_upproj = (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0)

    Width = 1024
    Height = 768

    K = get_intrinsics(np.array(cam_upproj).reshape(4,4),Width, Height)
    RT = get_extrinsics2(np.array(cam_upview).reshape(4,4))

    PMat = (K@RT[:3,:])
    # Setup Path
    nArmJoints = p.getNumJoints(armId, physicsClientId=physicsClient)

    JointPoses = p.calculateInverseKinematics(armId, nArmJoints-2, [-0.4, 0.3, 1.05]) 
    traj_step = 100
    pnts = generate_trajectory(np.array([-0.13, 0.3, 1.035]),np.array([0.05, 0.3, 1.035]),numSamples=traj_step,frequency=6,amp=0.01)

    pntsAndReturn = np.vstack((pnts[::-1],pnts))
    print(f'Number of DOFs: {nArmJoints})')

    last_dm = np.zeros((Height,Width))
    #breakpoint()
    #print(p.getAABB(human_inst.body))

    Forces = []
    bodyparts = []
    armparts = []

    old_path = []
    new_path = []
    actual_path = []

    far_ = 1000
    near_ = 0.01

    rot = Rotation.from_euler('xyz', [0, 90, 0], degrees=True)

    rot_quat = rot.as_quat()

    EndEfferctorId = nArmJoints-3
    for j in range(600):

        p.stepSimulation(physicsClientId=physicsClient)
        out = p.getClosestPoints(armId,human_inst.body,10,EndEfferctorId)#5
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
        #pntsAndReturn[j%(2*traj_step),2] += (1*(out[0][6][2]-0.00))
        #pntsAndReturn[j%(2*traj_step),2] /= 2

        #pntsAndReturn[j%(2*traj_step),2] = max(pntsAndReturn[j%(2*traj_step),2],out[0][6][2]-0.005) + (5*Forces[-1]/10e4)
        #time.sleep(TimeStep)
        #print(f'Pnt: {pntsAndReturn[j%(2*traj_step)]}')
        ImagArmPnt = (PMat@(np.array(list(pntsAndReturn[j%(2*traj_step)])+[1])[:,None]))
        ImagArmPnt /= ImagArmPnt[-1]
        #breakpoint()
        # Get the current camera
        cam = p.getDebugVisualizerCamera()

        com_pose, com_orient, _, _, _, _ = p.getLinkState(armId, EndEfferctorId)
        ArmLinkPnt = (PMat@(np.array(list(com_pose)+[1])[:,None]))
        actual_path.append(com_pose[-1])
        ArmLinkPnt /= ArmLinkPnt[-1]

        # try hide arm with  changeVisualShape
        #print(cam)
        # every two seconds
        if j%int(100/TimeStep):
            img_out = p.getCameraImage(Width, Height, cam_upview, cam_upproj)
            #img = np.array(img_out[2]).reshape(Height, Width,-1)[:,:,:3].astype(np.uint8)# RGBA

            depth = np.array(img_out[3]).reshape(Height, Width)#.astype(np.uint8)# depth
            nDepth = (far_*near_/(far_-(far_-near_)*depth))

            nDepth -= nDepth.max()
            nDepth = -1*nDepth

            #depth -= depth.min()
            #depth /= depth.max()
            #img = arm_camera(projection_matrix,armId=armId)

            mask = np.array(img_out[4]).reshape(Height, Width)#.astype(np.uint8)# depth

            Depth2Show = nDepth * (mask==human_inst.body)
            last_dm[(mask==human_inst.body)] = Depth2Show[(mask==human_inst.body)]
        if False:
            Depth2Show = last_dm.copy()

            Z_surface = Depth2Show[int(ImagArmPnt[1]),int(ImagArmPnt[0])]

            #print(Z_surface)
            #Depth2Show /= Depth2Show.max()
            
            #img = np.array(img_out[2]).reshape(Height, Width,-1)[:,:,:3].astype(np.uint8)# RGBA

            #cv2.imshow('img',((mask==human_inst.body)*255).astype(np.uint8))
            # convert to RGB
            if j:
                Depth2Show[Depth2Show>0] -= (Depth2Show[Depth2Show>0]).min()
                Depth2Show /= Depth2Show.max()

            img = (Depth2Show*255).astype(np.uint8).astype(np.uint8)
            img = np.dstack([np.zeros_like(img),np.zeros_like(img),img])
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            #print(ImagArmPnt[:2].flatten().astype(np.int32))
            #print(f'Old Z: {pntsAndReturn[j%(2*traj_step)][2]}, New Z: {Z_surface}: diff {abs(pntsAndReturn[j%(2*traj_step)][2]-Z_surface)}')

            old_path.append(pntsAndReturn[j%(2*traj_step)][2])
            new_path.append(Z_surface)

            if Z_surface:
                pass
                #pntsAndReturn[j%(2*traj_step)][2] += Z_surface
                #pntsAndReturn[j%(2*traj_step)][2] /= 2

                #pntsAndReturn[j%(2*traj_step)][2] = np.clip(pntsAndReturn[j%(2*traj_step)][2],Z_surface-0.01,Z_surface+0.01) # 1 cm
                #min(Z_surface,pntsAndReturn[j%(2*traj_step)][2])

            else:
                new_path[-1] = pntsAndReturn[j%(2*traj_step)][2]


            #print((PMat@np.array((-0.2828510, 0.12460, 0.6354567,1))))

            #breakpoint()

            #print(ImagArmPnt.T[0,:2],ArmLinkPnt.T[0,:2])
            #print(rot_quat)
            #cv2.circle(img,ImagArmPnt[:2].flatten().astype(np.int32),radius=4,color=(255,0,0),thickness=-1)
            #cv2.circle(img,ArmLinkPnt[:2].flatten().astype(np.int32),radius=4,color=(0,255,0),thickness=-1)

            #cv2.imshow('Depth Map',img)
            #cv2.waitKey(1)

            #states = p.getJointStates(armId,[0,1,2,3,4,5,6,7,8])
            #print([state[0] for state in states])

        # act
        # TODO taget orientation # we need only 3
        jointIndx = [1,2,3,4,5,6]
        if True:
            JointPoses = list(p.calculateInverseKinematics(armId, EndEfferctorId, pntsAndReturn[j%(2*traj_step)],rot_quat.tolist())) 
        else:
            JointPoses = list(p.calculateInverseKinematics(armId, EndEfferctorId, pntsAndReturn[j%(2*traj_step)]))
                                                       
        p.setJointMotorControlArray(armId, jointIndices=jointIndx, controlMode=p.POSITION_CONTROL, 
                                    targetPositions=[JointPoses[j-1] for j in jointIndx],forces=100*np.ones_like(jointIndx))
        #print(JointPoses)
        if j%int(2/TimeStep):
            # Update Path
            p1,p2 = p.getAABB(human_inst.body)
            #pnts = generate_trajectory(np.array([p1[0]+0.125, 0.3, p2[2]-0.04]),np.array([p2[0]+0.1, 0.3, p2[2]-0.04]),numSamples=traj_step,frequency=6,amp=0.035)
            pnts = generate_trajectory(np.array([p1[0]+0.0, 0.3, p2[2]+0.03]),np.array([p2[0]+0.05, 0.3, p2[2]+0.03]),numSamples=traj_step,frequency=6,amp=0.02)
            pntsAndReturn = np.vstack((pnts[::-1],pnts))


    cv2.destroyAllWindows()

    armPos, armOrn = p.getBasePositionAndOrientation(armId)

    print(armPos,armOrn)

    p.disconnect()

    draw_data(Forces,armparts,bodyparts,old_path=old_path,new_path=new_path,actual_path=actual_path)


if __name__ == "__main__":

    main()



