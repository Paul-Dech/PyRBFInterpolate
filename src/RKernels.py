import numpy as np

def linear(r):
    return -r

def thin_plate_spline(r):
    return r**2 * np.log(r)

def cubic(r):
    return r**3

def gaussian(r):
    return np.exp(-r**2)

def multiquadric(r):
    return np.sqrt(1 + r**2)

def inverse_multiquadric(r):
    return 1 / np.sqrt(1 + r**2)

def quintic(r):
    return r**5