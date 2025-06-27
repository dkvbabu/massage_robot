current_pressure = 20

def set_pressure(value):
    global current_pressure
    current_pressure = value

def get_pressure():
    return current_pressure

# This module is ready for future extension (PID, RL, etc.)
