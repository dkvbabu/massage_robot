import numpy as np
from massage_robot.mapping import SurfaceMapper

def test_add_and_reconstruct():
    sm = SurfaceMapper()
    depth = np.ones((10,10))
    intr = (1,1,5,5)
    sm.add_scan(depth, intr)
    pts, faces = sm.reconstruct()
    assert pts.ndim == 2
    assert faces.ndim == 2
