import numpy as np

class ForceController:
    """
    Implements PID-based force/pressure control.
    """
    def __init__(self, kp=1.0, ki=0.0, kd=0.1, target_force=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target = target_force
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_force, dt=1e-3):
        error = self.target - current_force
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        tau = self.kp*error + self.ki*self.integral + self.kd*derivative
        return tau

    def reward(self, current_force):
        return - (current_force - self.target)**2
