import pytest
from massage_robot.sensors import SensorSimulator, get_camera_image
from massage_robot.env import MassageEnv

def test_rgbd_shape():
    class DummyEnv:
        def __init__(self):
            self.SimID = 0  # Dummy SimID for testing
    env = DummyEnv()
    s = SensorSimulator(env)
    rgb, depth = s.get_rgbd({'target':[0,0,0],'distance':1,'yaw':0,'pitch':-30,'roll':0})
    assert rgb.shape[0] == depth.shape[0]

def test_camera_image():
    sim = MassageEnv(render=False)
    img = get_camera_image(sim)
    assert img.shape == (64, 64, 3)
    sim.close()
