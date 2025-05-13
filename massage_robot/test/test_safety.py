import pytest
from massage_robot.safety import SafetyModule

def test_verify_safe():
    safety = SafetyModule(max_force=5.0, max_velocity=1.0)
    ok, msg = safety.verify([1.0, 2.0], [0.5, 0.2])
    assert ok is True
    assert msg == 'Safe'

def test_verify_force_violation():
    safety = SafetyModule(max_force=5.0, max_velocity=1.0)
    ok, msg = safety.verify([6.0, 2.0], [0.1, 0.1])
    assert ok is False
    assert 'Force limit' in msg

def test_verify_velocity_violation():
    safety = SafetyModule(max_force=10.0, max_velocity=1.0)
    ok, msg = safety.verify([1.0, 0.5], [1.5, 0.2])
    assert ok is False
    assert 'Velocity limit' in msg
