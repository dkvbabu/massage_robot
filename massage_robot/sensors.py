import numpy as np
import pybullet as p

class SensorSimulator:
    """
    Simulates RGB-D camera and LiDAR scans from arbitrary viewpoints.
    """
    def __init__(self, env, depth_noise=0.01):
        self.env = env
        self.noise = depth_noise

    def get_rgbd(self, cam_pose, fov=60, resolution=(640,480)):
        width, height = resolution
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_pose['target'],
            distance=cam_pose['distance'],
            yaw=cam_pose['yaw'], pitch=cam_pose['pitch'], roll=cam_pose['roll'],
            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov, width/height, 0.1, 10)
        rgb, depth, seg = p.getCameraImage(
            width, height, view_matrix, proj_matrix)
        depth = np.array(depth)*self.noise
        return np.array(rgb), depth

    def get_lidar_scan(self, origin, angles, max_range=2.0):
        ranges = []
        for theta in angles:
            dir = [np.cos(theta), np.sin(theta), 0]
            hit = p.rayTest(origin, [origin[i] + max_range*dir[i] for i in range(3)])[0]
            ranges.append(hit[2] if hit[0] != -1 else max_range)
        return np.array(ranges)
