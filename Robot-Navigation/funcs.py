from numpy import pi


def normalize_angle(angle):
    if angle > 2*pi:
        angle %= 2*pi
    elif angle < -2*pi:
        angle = -abs(angle) % 2*pi
        
    if angle > pi:
        return angle - 2*pi
    elif angle < -pi:
        return angle + 2*pi
    else:
        return angle


