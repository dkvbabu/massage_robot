import numpy as np
from scipy.spatial.transform import Rotation as R

class CoordinateTransforms:
    """
    Utility functions for transforming between sensor, world, and robot frames.
    """
    @staticmethod
    def pose_to_matrix(position, quaternion):
        rot = R.from_quat(quaternion).as_matrix()
        mat = np.eye(4)
        mat[:3,:3] = rot
        mat[:3,3] = position
        return mat

    @staticmethod
    def transform_points(points, matrix):
        homo = np.concatenate([points, np.ones((len(points),1))], axis=1)
        trans = (matrix @ homo.T).T
        return trans[:,:3]
