import pytest
from massage_robot.force_control import ForceController

def test_compute_torque_zero_error():
    ctrl = ForceController(kp=1.0, ki=0.0, kd=0.0, target_force=10.0)
    torque = ctrl.compute_torque(current_force=10.0, dt=0.1)
    assert torque == pytest.approx(0.0)

def test_compute_torque_pid_behavior():
    ctrl = ForceController(kp=2.0, ki=0.0, kd=0.0, target_force=5.0)
    tau = ctrl.compute_torque(current_force=3.0, dt=0.1)
    # error = 2.0 → torque = kp * error = 4.0
    assert tau == pytest.approx(4.0)

def test_reward_quadratic():
    ctrl = ForceController(target_force=8.0)
    assert ctrl.reward(6.0) == pytest.approx(- (6.0 - 8.0)**2)
