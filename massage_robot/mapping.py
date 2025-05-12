import numpy as np
from scipy.spatial import Delaunay

class SurfaceMapper:
    """
    Builds a 3D mesh of the human phantom surface from depth or point clouds.
    """
    def __init__(self):
        self.points = []

    def add_scan(self, depth_image, intrinsics):
        h, w = depth_image.shape
        fx, fy, cx, cy = intrinsics
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        zs = depth_image
        xs = (xs - cx) * zs / fx
        ys = (ys - cy) * zs / fy
        pts = np.stack([xs, ys, zs], axis=-1).reshape(-1,3)
        self.points.append(pts)

    def reconstruct(self):
        all_pts = np.vstack(self.points)
        tri = Delaunay(all_pts[:10000])
        return all_pts, tri.simplices
