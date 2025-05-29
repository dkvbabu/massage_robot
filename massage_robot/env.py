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

    def __init__(self,render=False):

        self.SimID = p.connect([p.DIRECT,p.GUI][render])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0,0,-10)
        startPos = [-0.8,0.1,1.0]
        cubeStartingPose = [-1.3,0.0,0.5]
        self.EErot = p.getQuaternionFromEuler([0, 90, 0])
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        planeId = p.loadURDF("plane.urdf")
        cubeId = p.loadURDF("cube.urdf",cubeStartingPose, startOrientation)
        self.armId = p.loadURDF('urdf/ur5_robot.urdf',startPos, startOrientation)

        self.TimeStep = 1/24.0
        p.setTimeStep(self.TimeStep)

        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.human_inst = load_scene(self.SimID)

        self.nArmJoints =  p.getNumJoints(self.armId, physicsClientId=self.SimID)
        self.EndEfferctorId = self.nArmJoints-3

        self.PointsInPath = 100
        self.controlledJoints = [1,2,3,4,5,6]
        self.baseLevel = 0.02

        p.resetDebugVisualizerCamera(cameraYaw= 92.4,cameraPitch=-41.8,cameraDistance=1.0,
                                     cameraTargetPosition=(-0.4828510, 0.12460, 0.6354567),)
        self.reset()




    def  reset(self):

        for i,rot in enumerate([-0.4,-0.9,1,-2.0,-1.5]):
            p.resetJointState(self.armId,i+1,rot)

        self.timestep = 0

        self.Forces = []
        self.bodyparts = []
        self.armparts = []

        self.old_path = []
        self.new_path = []
        self.actual_path = []

        self.make_path()



    def make_path(self):

        # Update Path
        p1,p2 = p.getAABB(self.human_inst.body)
        pnts = generate_trajectory(np.array([p1[0], 0.3, p2[2]+self.baseLevel]),
                                   np.array([p2[0], 0.3, p2[2]+self.baseLevel]),numSamples=self.PointsInPath,frequency=6,amp=0.02)
        self.pntsAndReturn = np.vstack((pnts[::-1],pnts))


    def get_action(self,change=0):

        out = p.getClosestPoints(self.armId,self.human_inst.body,10,self.EndEfferctorId)
    
        MoveTo = self.pntsAndReturn[(self.timestep%(2*self.PointsInPath))]

        #MoveTo[2] += (1*(out[0][6][2]))
        #MoveTo[2] /= 2
        
        return MoveTo

    def step(self, action):

        JointPoses = list(p.calculateInverseKinematics(self.armId, self.EndEfferctorId, action,self.EErot))

        p.setJointMotorControlArray(self.armId, jointIndices=self.controlledJoints, controlMode=p.POSITION_CONTROL, 
                                    targetPositions=[JointPoses[j-1] for j in self.controlledJoints],forces=100*np.ones_like(self.controlledJoints))

        p.stepSimulation(physicsClientId=self.SimID)

        self.collect_stats()

        self.timestep += 1

        if (self.timestep%(self.PointsInPath*2))==0:
            self.make_path()


    def collect_stats(self):

        out_1 = p.getContactPoints(self.armId,self.human_inst.body)

        if len(out_1):

            self.bodyparts.append(out_1[0][4])
            self.armparts.append(out_1[0][3])
            self.Forces.append(out_1[0][9]) 

        else:
            self.bodyparts.append(0)
            self.armparts.append(0)
            self.Forces.append(0)

        com_pose, com_orient, _, _, _, _ = p.getLinkState(self.armId, self.EndEfferctorId)
        self.actual_path.append(com_pose[-1])

        self.old_path.append(self.pntsAndReturn[(self.timestep%(2*self.PointsInPath))][2])
        #new_path.append(Z_surface)

    def close(self):


        p.disconnect()
        draw_data(self.Forces,self.armparts,self.bodyparts,
                  old_path=self.old_path,new_path=self.new_path,actual_path=self.actual_path)



    def get_reward(self):
        pass

    def get_state(self):
        pass


def main():

    env = MassageEnv(render=True)

    for i in range(3000):

        action = env.get_action()
        env.step(action)

    env.close()




if __name__ == '__main__':

    main()


























