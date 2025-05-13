## massage_robot/force_control.py
import numpy as np

class ForceController:
    """
    PID-based force control, with methods to compute torque commands and rewards.
    """
    def __init__(self, kp=1.0, ki=0.01, kd=0.05, target_force=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target = target_force
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_torque(self, current_force, dt):
        error = self.target - current_force
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        torque = self.kp*error + self.ki*self.integral + self.kd*derivative
        return torque

    def reward(self, current_force):
        # Negative quadratic cost around target
        return - (current_force - self.target)**2
