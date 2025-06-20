import pybullet as p
import time,os
import pybullet_data
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter

from massage_robot.utils import get_extrinsics,get_intrinsics,get_extrinsics2

import configparser

from massage_robot.human.human_creation import HumanCreation
from massage_robot.human import agent, human
from massage_robot.human.agent import Agent
from massage_robot.human.human import Human
from massage_robot.human.furniture import Furniture

import matplotlib.pyplot as plt

from massage_robot.generate_path import generate_trajectory, PathGenerationProgram

from massage_robot.test_viewer import load_scene,draw_data



class MassageEnv():

    def __init__(self, render=False, pattern='sine', amp=0.02, freq=2.0, approach_samples=50, main_samples=200, retract_samples=50, force=20, speed=1.0, log_to_tb=True):

        self.SimID = p.connect([p.DIRECT,p.GUI][render])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0,0,-10)
        startPos = [-0.8,0.1,1.0]
        cubeStartingPose = [-1.3,0.0,0.5]
        self.EErot = p.getQuaternionFromEuler([0, 90, 0])
        startOrientation = p.getQuaternionFromEuler([0,0,0])

        planeId = p.loadURDF("plane.urdf")
        cubeId = p.loadURDF("cube.urdf",cubeStartingPose, startOrientation)
        urdf_path = os.path.join(os.path.dirname(__file__), "urdf", "ur5_robot.urdf")
        self.armId = p.loadURDF(urdf_path, startPos, startOrientation)

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

        self.pattern = pattern
        self.amp = amp
        self.freq = freq
        self.approach_samples = approach_samples
        self.main_samples = main_samples
        self.retract_samples = retract_samples
        self.force = force
        self.speed = speed

        p.resetDebugVisualizerCamera(cameraYaw= 92.4,cameraPitch=-41.8,cameraDistance=1.0,
                                     cameraTargetPosition=(-0.4828510, 0.12460, 0.6354567),)
        self.reset()

        self.log_to_tb = log_to_tb
        self.writer = SummaryWriter(log_dir="runs/sim_params") if log_to_tb else None




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
        p1, p2 = p.getAABB(self.human_inst.body)
        waypoints = [
            [p1[0], 0.3, p2[2] + self.baseLevel],
            [p2[0], 0.3, p2[2] + self.baseLevel]
        ]
        pg = PathGenerationProgram(waypoints)
        traj = pg.generate(
            main_samples=self.main_samples,
            approach_samples=self.approach_samples,
            retract_samples=self.retract_samples,
            pattern=self.pattern,
            amp=self.amp,
            freq=self.freq
        )
        self.pntsAndReturn = np.array([t['position'] for t in traj])


    def get_action(self,change=0):

        out = p.getClosestPoints(self.armId,self.human_inst.body,10,self.EndEfferctorId)
    
        MoveTo = self.pntsAndReturn[(self.timestep%(2*self.PointsInPath))]

        #MoveTo[2] += (1*(out[0][6][2]))
        #MoveTo[2] /= 2
        
        return MoveTo

    def step(self, action, force=None, speed=None):
        force = force if force is not None else self.force
        speed = speed if speed is not None else self.speed
        # Debug print for parameter tracking
        print(f"Step: force={force}, speed={speed}, pattern={self.pattern}, amp={self.amp}, freq={self.freq}, approach={self.approach_samples}, main={self.main_samples}, retract={self.retract_samples}")
        JointPoses = list(p.calculateInverseKinematics(self.armId, self.EndEfferctorId, action, self.EErot))
        p.setJointMotorControlArray(self.armId, jointIndices=self.controlledJoints, controlMode=p.POSITION_CONTROL, 
                                    targetPositions=[JointPoses[j-1] for j in self.controlledJoints],forces=force*np.ones_like(self.controlledJoints))
        p.stepSimulation(physicsClientId=self.SimID)
        self.collect_stats()
        self.timestep += 1
        if (self.timestep%(self.PointsInPath*2))==0:
            self.make_path()
        if speed > 0:
            time.sleep(1.0 / (24.0 * speed))
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar("params/force", force, self.timestep)
            self.writer.add_scalar("params/speed", speed, self.timestep)
            self.writer.add_scalar("params/amp", self.amp, self.timestep)
            self.writer.add_scalar("params/freq", self.freq, self.timestep)
            self.writer.add_scalar("params/approach_samples", self.approach_samples, self.timestep)
            self.writer.add_scalar("params/main_samples", self.main_samples, self.timestep)
            self.writer.add_scalar("params/retract_samples", self.retract_samples, self.timestep)
            self.writer.add_text("params/pattern", str(self.pattern), self.timestep)
            if self.Forces:
                self.writer.add_scalar("stats/force", self.Forces[-1], self.timestep)

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

    def set_camera(self, yaw, pitch):
        p.resetDebugVisualizerCamera(cameraYaw=yaw, cameraPitch=pitch, cameraDistance=1.2, cameraTargetPosition=(-0.18, -0.04, 0.42))

    def get_stats(self):
        return {
            'Forces': self.Forces,
            'armparts': self.armparts,
            'bodyparts': self.bodyparts,
            'old_path': self.old_path,
            'new_path': self.new_path,
            'actual_path': self.actual_path
        }

    def run_until_exit(self):
        """Run the simulation until the user closes the PyBullet window or presses ESC."""
        while p.isConnected(self.SimID):
            action = self.get_action()
            self.step(action, force=self.force, speed=self.speed)
            keys = p.getKeyboardEvents()
            if 27 in keys:  # ESC key
                break
            time.sleep(self.TimeStep)
        self.close()

    def update_parameters(self, **kwargs):
        """Update simulation parameters in real time."""
        path_params_changed = False
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ['pattern', 'amp', 'freq', 'approach_samples', 'main_samples', 'retract_samples']:
                    path_params_changed = True
        if path_params_changed:
            print(f"Path parameters changed: pattern={self.pattern}, amp={self.amp}, freq={self.freq}, approach={self.approach_samples}, main={self.main_samples}, retract={self.retract_samples}")
            self.make_path()
        else:
            print(f"Control parameters changed: force={self.force}, speed={self.speed}")

    def update_camera(self, yaw=None, pitch=None):
        if yaw is not None or pitch is not None:
            current = p.getDebugVisualizerCamera()
            new_yaw = yaw if yaw is not None else current[8]
            new_pitch = pitch if pitch is not None else current[9]
            self.set_camera(new_yaw, new_pitch)

def main():

    env = MassageEnv(render=True)

    for i in range(3000):

        action = env.get_action()
        env.step(action)

    env.close()




if __name__ == '__main__':

    main()


























