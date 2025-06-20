def kneading_lower_back(force=20, speed=1.0, duration=60):
    """Simulate kneading massage on lower back."""
    return {
        'technique': 'kneading',
        'region': 'lower_back',
        'force': force,
        'speed': speed,
        'duration': duration,
        'trajectory': [(0,0,0), (0.1,0,0), (0.2,0,0)]
    }

def kneading_upper_back(force=20, speed=1.0, duration=60):
    return {'technique': 'kneading', 'region': 'upper_back', 'force': force, 'speed': speed, 'duration': duration}

def kneading_shoulders(force=20, speed=1.0, duration=60):
    return {'technique': 'kneading', 'region': 'shoulders', 'force': force, 'speed': speed, 'duration': duration}

def kneading_neck(force=20, speed=1.0, duration=60):
    return {'technique': 'kneading', 'region': 'neck', 'force': force, 'speed': speed, 'duration': duration}

def pressure_lower_back(force=20, speed=1.0, duration=60):
    return {'technique': 'pressure', 'region': 'lower_back', 'force': force, 'speed': speed, 'duration': duration}

def pressure_upper_back(force=20, speed=1.0, duration=60):
    """Simulate pressure massage on upper back."""
    return {
        'technique': 'pressure',
        'region': 'upper_back',
        'force': force,
        'speed': speed,
        'duration': duration,
        'trajectory': [(0.2,0,0.1), (0.3,0,0.1), (0.4,0,0.1)]
    }

def pressure_shoulders(force=20, speed=1.0, duration=60):
    return {'technique': 'pressure', 'region': 'shoulders', 'force': force, 'speed': speed, 'duration': duration}

def pressure_neck(force=20, speed=1.0, duration=60):
    return {'technique': 'pressure', 'region': 'neck', 'force': force, 'speed': speed, 'duration': duration}

def tapping_lower_back(force=20, speed=1.0, duration=60):
    return {'technique': 'tapping', 'region': 'lower_back', 'force': force, 'speed': speed, 'duration': duration}

def tapping_upper_back(force=20, speed=1.0, duration=60):
    return {'technique': 'tapping', 'region': 'upper_back', 'force': force, 'speed': speed, 'duration': duration}

def tapping_shoulders(force=20, speed=1.0, duration=60):
    return {'technique': 'tapping', 'region': 'shoulders', 'force': force, 'speed': speed, 'duration': duration}

def tapping_neck(force=20, speed=1.0, duration=60):
    return {'technique': 'tapping', 'region': 'neck', 'force': force, 'speed': speed, 'duration': duration}

def rolling_lower_back(force=20, speed=1.0, duration=60):
    return {'technique': 'rolling', 'region': 'lower_back', 'force': force, 'speed': speed, 'duration': duration}

def rolling_upper_back(force=20, speed=1.0, duration=60):
    return {'technique': 'rolling', 'region': 'upper_back', 'force': force, 'speed': speed, 'duration': duration}

def rolling_shoulders(force=20, speed=1.0, duration=60):
    return {'technique': 'rolling', 'region': 'shoulders', 'force': force, 'speed': speed, 'duration': duration}

def rolling_neck(force=20, speed=1.0, duration=60):
    return {'technique': 'rolling', 'region': 'neck', 'force': force, 'speed': speed, 'duration': duration}
