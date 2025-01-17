import numpy as np

class linear_kernel:
    """Linear kernel function f(r) = -r
    """
    def __init__(self):
        pass
    def eval(self, x0, x1=None):
        """Evaluate the value of the kernel

        Parameters
        ----------
        x0 : array_like or float
            First point or distance between two points
        x1 : array_like or float, optional
            Second point (if x0 is the first point)

        Returns
        -------
        float
            Kernel value
        """
        return -x0 if x1 is None else -np.linalg.norm(x0 - x1)

    def eval_grad(self, x0, x1, _axis=1):
        """Gradient of the kernel with respect to x0

        Parameters
        ----------
        x0 : array_like or float
            First point
        x1 : array_like or float
            Second point
        
        Returns
        -------
        tulpe of array_like or float
            Gradients value with respect to x0 and x1
        """
        dist = np.linalg.norm(x0 - x1, axis=_axis)[:, None]
        grad = (x0 - x1) / dist
        return (-grad, grad)

class thin_plate_spline_kernel:
    """Thin plate spline kernel function f(r) = r^2 * log(r)
    """
    def __init__(self):
        pass
    def eval(self, x0, x1=None):
        """Evaluate the value of the kernel

        Parameters
        ----------
        x0 : array_like or float
            First point or distance between two points
        x1 : array_like or float, optional
            Second point (if x0 is the first point)

        Returns
        -------
        float
            Kernel value
        """
        if x1 is None:
            r = x0
        else:
            r = np.linalg.norm(x0 - x1)
        return r**2 * np.log(r) if r > 0 else 0
    def eval_grad(self, x0, x1, _axis=1):
        raise NotImplementedError("Gradient not implemented for this kernel")

class cubic_kernel:
    """Cubic kernel function f(r) = r^3
    """
    def __init__(self):
        pass
    def eval(self, x0, x1=None):
        """Evaluate the value of the kernel

        x1 : array_like or float
            First point or distance between two points
        x1 : array_like or float, optional
            Second point (if x0 is the first point)

        Returns
        -------
        float
            Kernel value
        """
        if x1 is None:
            r = x0
        else:
            r = np.linalg.norm(x0 - x1)
        return r**3

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

def cubic_grad(r):
    return 3 * r**2

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