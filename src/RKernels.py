import numpy as np

def linear(r):
    """Linear kernel function f(r) = -r

    Parameters
    ----------
    r : array_like or float
        Distance between two points

    Returns
    -------
    array_like or float
        Kernel value
    """
    return -r

def thin_plate_spline(r):
    """Thin plate spline kernel function f(r) = r^2 * log(r)

    Parameters
    ----------
    r : array_like or float
        Distance between two points

    Returns
    -------
    array_like or float
        Kernel value
    """
    return r**2 * np.log(r)

def cubic(r):
    """Cubic kernel function f(r) = r^3
    
    Parameters
    ----------
    r : array_like or float
        Distance between two points
        
    Returns
    -------
    array_like or float
        Kernel value
    """
    return r**3

def gaussian(r):
    """Gaussian kernel function f(r) = exp(-r^2)
    
    Parameters
    ----------
    r : array_like or float
        Distance between two points
    
    Returns
    -------
    array_like or float
        Kernel value
    """
    return np.exp(-r**2)

def multiquadric(r):
    """Multiquadric kernel function f(r) = sqrt(1 + r^2)
    
    Parameters
    ----------
    r : array_like or float
        Distance between two points
    
    Returns
    -------
    array_like or float
        Kernel value
    """
    return np.sqrt(1 + r**2)

def inverse_multiquadric(r):
    """Inverse multiquadric kernel function f(r) = 1 / sqrt(1 + r^2)

    Parameters
    ----------
    r : array_like or float
        Distance between two points
    
    Returns
    -------
    array_like or float
        Kernel value
    """
    return 1 / np.sqrt(1 + r**2)

def quintic(r):
    """Quintic kernel function f(r) = r^5
    
    Parameters
    ----------
    r : array_like or float
        Distance between two points
    
    Returns
    -------
    array_like or float
        Kernel value
    """
    return r**5