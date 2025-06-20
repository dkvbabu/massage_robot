"""Safety module for monitoring forces and velocities."""

class SafetyModule:
    def __init__(self, max_force=5.0, max_velocity=1.0):
        self.max_force = max_force
        self.max_velocity = max_velocity

    def verify(self, forces, velocities):
        """Verify if forces and velocities are within safety limits."""
        # Check force limits
        for force in forces:
            if abs(force) > self.max_force:
                return False, f"Force limit exceeded: {force} > {self.max_force}"
        
        # Check velocity limits
        for velocity in velocities:
            if abs(velocity) > self.max_velocity:
                return False, f"Velocity limit exceeded: {velocity} > {self.max_velocity}"
        
        return True, "Safe"
