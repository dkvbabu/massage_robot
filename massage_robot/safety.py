class SafetyModule:
    """
    Monitors force and velocity limits; triggers emergency stop if exceeded.
    """
    def __init__(self, max_force=20.0, max_velocity=1.0):
        self.max_force = max_force
        self.max_velocity = max_velocity

    def check(self, forces, velocities):
        if any(f > self.max_force for f in forces):
            return False, 'Force limit exceeded'
        if any(abs(v) > self.max_velocity for v in velocities):
            return False, 'Velocity limit exceeded'
        return True, 'OK'
