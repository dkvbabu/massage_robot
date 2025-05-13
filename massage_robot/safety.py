## massage_robot/safety.py
class SafetyModule:
    """
    Checks force and velocity limits and triggers emergency stops.
    """
    def __init__(self, max_force=25.0, max_velocity=1.5):
        self.max_force = max_force
        self.max_velocity = max_velocity

    def verify(self, forces, velocities):
        for f in forces:
            if f > self.max_force:
                return False, 'Force limit violated'
        for v in velocities:
            if abs(v) > self.max_velocity:
                return False, 'Velocity limit violated'
        return True, 'Safe'
