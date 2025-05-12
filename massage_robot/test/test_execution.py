import pytest
from massage_robot.execution import GridExecutor

class DummyEnv:
    def read_force(self): return 0.0
    def step(self, wp, tau): return {'contact_forces':{0:0.0}, 'joint_states':[(0,0)]}

def test_execute_grid():
    class Planner:
        def plan_waypoints(self, r): return [[0,0,0]]
    exec = GridExecutor(Planner(), None, type('S', (), {'check':lambda self,f,v:(True,'')}))
    result = exec.execute_grid([[1]], DummyEnv())
    assert result[(0,0)] == 'completed'
