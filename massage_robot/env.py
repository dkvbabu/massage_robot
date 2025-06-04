import pybullet as p
import time,os
import pybullet_data
import numpy as np
import cv2

from utils import get_extrinsics,get_intrinsics,get_extrinsics2

import configparser

from human.human_creation import HumanCreation
from human import agent, human
from human.agent import Agent
from human.human import Human
from human.furniture import Furniture

import matplotlib.pyplot as plt

from generate_path import generate_trajectory

from test_viewer import load_scene,draw_data



class MassageEnv():

    def __init__(self,render=False,auto_reset=True):

        self.SimID = p.connect([p.DIRECT,p.GUI][render])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0,0,-10)
        startPos = [-0.8,0.1,1.0]
        cubeStartingPose = [-1.3,0.0,0.5]
        self.EErot = p.getQuaternionFromEuler([0, 90, 0])
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        self.single_observation_space = (27,)
        self.single_action_space = (3,)

        planeId = p.loadURDF("plane.urdf")
        cubeId = p.loadURDF("cube.urdf",cubeStartingPose, startOrientation)
        self.armId = p.loadURDF('urdf/ur5_robot.urdf',startPos, startOrientation)

        self.TimeStep = 1/24.0
        p.setTimeStep(self.TimeStep)
        self.episode_length = int(15/self.TimeStep)

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.human_inst = load_scene(self.SimID)

        self.nArmJoints =  p.getNumJoints(self.armId, physicsClientId=self.SimID)
        self.EndEfferctorId = self.nArmJoints-3

        self.PointsInPath = 100
        self.controlledJoints = [1,2,3,4,5,6]
        self.baseLevel = 0.02

        self.maxPressure = 50
        self.auto_reset = auto_reset

        p.resetDebugVisualizerCamera(cameraYaw= 92.4,cameraPitch=-41.8,cameraDistance=1.0,
                                     cameraTargetPosition=(-0.4828510, 0.12460, 0.6354567),)
        self.reset()




    def reset(self,seed=0):

        for i,rot in enumerate([-0.4,-0.9,1,-2.0,-1.5]):
            p.resetJointState(self.armId,i+1,rot)

        self.timestep = 0
        self.epReturn = 0
        self.Forces = []
        self.bodyparts = []
        self.armparts = []

        self.old_path = []
        self.new_path = []
        self.actual_path = []

        self.make_path()

        stats = self.collect_stats()

        self.human_inst.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2, -np.pi, 0])

        return self.get_state(stats)


    def make_path(self):

        # Update Path
        p1,p2 = p.getAABB(self.human_inst.body)
        pnts = generate_trajectory(np.array([p1[0], 0.3, p2[2]+self.baseLevel]),
                                   np.array([p2[0], 0.3, p2[2]+self.baseLevel]),numSamples=self.PointsInPath,frequency=3,amp=0.02)
        self.pntsAndReturn = np.vstack((pnts[::-1],pnts))


    def get_action(self,change=0):
        # change between 

        #out = p.getClosestPoints(self.armId,self.human_inst.body,10,self.EndEfferctorId)
        MoveTo = self.pntsAndReturn[(self.timestep%(2*self.PointsInPath))].copy()# + (change/10)
        MoveTo += (change/10)
        #MoveTo[2] += (1*(out[0][6][2]))
        #MoveTo[2] /= 2
        return MoveTo

    def step(self, change=0):

        action = self.get_action((1/(1 + np.exp(-change)))-0.5)

        JointPoses = list(p.calculateInverseKinematics(self.armId, self.EndEfferctorId, action, self.EErot))

        p.setJointMotorControlArray(self.armId, jointIndices=self.controlledJoints, controlMode=p.POSITION_CONTROL, 
                                    targetPositions=[JointPoses[j-1] for j in self.controlledJoints],forces=100*np.ones_like(self.controlledJoints))

        p.stepSimulation(physicsClientId=self.SimID)

        stats = self.collect_stats()

        self.timestep += 1

        if (self.timestep%(self.PointsInPath*2))==0:
            self.make_path()


        done = False
        info = {}
        reward = self.get_reward(stats)
        AllStates = self.get_state(stats)
        if (self.timestep==self.episode_length) and self.auto_reset:

            done = True
            # save episodic return and length
            info.update({'episode':{'r':self.epReturn,'l':self.timestep}})
            AllStates = self.reset()

        return AllStates,reward,done,info

    def collect_stats(self):

        bodypart, armpart, Force, NormalDist, NormalDirect, friction, frictionDir,pnt = self.get_contact_points()
        self.bodyparts.append(bodypart)
        self.armparts.append(armpart)
        self.Forces.append(Force) 
        com_pose, com_orient, _, _, _, _ = p.getLinkState(self.armId, self.EndEfferctorId)
        self.actual_path.append(com_pose[-1])

        self.old_path.append(self.pntsAndReturn[(self.timestep%(2*self.PointsInPath))][2])
        #new_path.append(Z_surface)

        return bodypart, armpart, Force, NormalDist, NormalDirect, friction, frictionDir, com_pose, com_orient,pnt

    def get_contact_points(self):

        # Closest points and norms
        out = p.getClosestPoints(self.armId,self.human_inst.body,10,self.EndEfferctorId)

        # Contact points and forces + frictions
        out_1 = p.getContactPoints(self.armId,self.human_inst.body)

        if len(out_1):
            # there's contact
            bodypart = (out_1[0][4])
            armpart = (out_1[0][3])
            Force = (out_1[0][9]) # normal force
            pnt = out_1[0][6]

            NormalDist = (out_1[0][8]) # normal distance

            NormalDirect = (out_1[0][7]) # normal direction
            friction = (out_1[0][10]) # friction
            frictionDir = (out_1[0][11]) # friction direction
        else:
            bodypart = -2
            armpart = -2
            Force = (out[0][9]) # normal force
            pnt = out[0][6]

            NormalDist = (out[0][8]) # normal distance

            NormalDirect = (out[0][7]) # normal direction
            friction = (out[0][10]) # friction
            frictionDir = (out[0][11]) # friction direction

        return bodypart, armpart, Force, NormalDist, NormalDirect, friction, frictionDir,pnt

    def close(self):


        p.disconnect()
        draw_data(self.Forces,self.armparts,self.bodyparts,
                  old_path=self.old_path,new_path=self.new_path,actual_path=self.actual_path)



    def get_reward(self,stats):

        bodypart, armpart, Force, NormalDist, NormalDirect, friction, frictionDir, com_pose, com_orient,pnt = stats

        NoContact = (armpart!=self.EndEfferctorId)

        HighPressure = (Force>self.maxPressure)

        WrongContact = (armpart not in [self.EndEfferctorId,-2])

        reward = ([2*Force,2*(self.maxPressure-Force)][HighPressure]) - (WrongContact*10)

        self.epReturn += reward

        return reward

    def get_state(self,stats):

        # 1 Joint states (position, velocity)
        Jstates = p.getJointStates(self.armId, self.controlledJoints)
        ArmStates = np.array([[j[0],j[1]] for j in Jstates]).flatten()#12

        # 2 ee position

        # 3 Next N step in path
        current_path_indx = (self.timestep%(2*self.PointsInPath))

        next_path = np.zeros((int(3/self.TimeStep),3),dtype=np.float64)
        nextIdx = (current_path_indx+int(3/self.TimeStep))
        next_path = self.pntsAndReturn[current_path_indx:nextIdx:5] #(N,3)
        if next_path.shape[0]< int(3/self.TimeStep)//5:
            diffRows = int(3/self.TimeStep)-next_path.shape[0]
            next_path = np.vstack((next_path,np.zeros((diffRows,3))))
        # 4 Closest points and norms
        # 5 Contact points and forces + frictions
        bodypart, armpart, Force, NormalDist, NormalDirect, friction, frictionDir, com_pose, com_orient,pnt = stats

        contactStates = list(pnt)+[Force, NormalDist] + list(NormalDirect)+ list(com_pose)+ list(com_orient) # 12

        return np.array(contactStates+ArmStates.tolist()) #, next_path


def main():

    env = MassageEnv(render=True)

    for i in range(env.episode_length*20):

        state,reward,done,info = env.step()
        if done:
            env.reset()
            print(f'Episodic rewards: {info["episode"]['r']}')

        time.sleep(0.001)
    env.close()




if __name__ == '__main__':

    main()


























