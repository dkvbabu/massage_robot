import numpy as np

class PressureController:
    """
    Computes torque commands to track a target pressure profile and rewards.
    """
    def __init__(self, kp=1.0, kd=0.1, target_force=10.0):
        self.kp = kp
        self.kd = kd
        self.target_force = target_force
        self.prev_error = 0.0

    def compute_action(self, current_force, dt=1e-3):
        error = self.target_force - current_force
        de = (error - self.prev_error) / dt
        tau = self.kp * error + self.kd * de
        self.prev_error = error
        return tau, error

    def reward(self, current_force):
        return - (current_force - self.target_force)**2
