import pytest
from massage_robot.sensors import SensorSimulator

def test_rgbd_shape():
    class E: pass
    env = E(); s = SensorSimulator(env)
    rgb, depth = s.get_rgbd({'target':[0,0,0],'distance':1,'yaw':0,'pitch':-30,'roll':0})
    assert rgb.shape[0] == depth.shape[0]
