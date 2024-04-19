# RBFInterpolator class

# @authors: Paul Dechamps, Adrien Crovato
# @date: 2024

import numpy as np
from scipy.spatial import KDTree
import importlib

_AVAILABLE_KERNELS = {
    'linear',
    'thin_plate_spline',
    'cubic',
    'gaussian',
    'multiquadric',
    'inverse_multiquadric',
    'quintic'
}

class RBFInterpolator:
    """Radial basis function (RBF) interpolation
    
    Parameters
    ----------
    y : array_like
        Data points, shape (n, d)
    x : array_like
        Interpolation points, shape (m, d)
    _neighbors : int, optional
        Number of neighbors to consider for each interpolation point, by default None
    _kernel : str, optional
        Kernel function to use, by default 'linear'
        Kernel can be one of the following:
        - 'linear' : f(r) = -r
        - 'thin_plate_spline' : f(r) = r^2 * log(r)
        - 'cubic' : f(r) = r^3
        - 'gaussian' : f(r) = exp(-r^2)
        - 'multiquadric' : f(r) = sqrt(1 + r^2)
        - 'inverse_multiquadric' : f(r) = 1 / sqrt(1 + r^2)
        - 'quintic' : f(r) = r^5
    
    Infos
    -----
    n : int Number of data points
    m : int Number of interpolation points
    d : int Number of dimensions
    
    Raises
    ------
    ValueError
        If an unknown kernel is provided
    """
    def __init__(self, y, x, _neighbors = None, _kernel='linear') -> None:

        if _kernel not in _AVAILABLE_KERNELS:
            raise ValueError(f"Unknown kernel: {_kernel}")
        
        RKernels = importlib.import_module("src.RKernels")
        self.__rbfKernel = getattr(RKernels, _kernel)

        self.x = x
        self.y = y
        if _neighbors is None:
            sz = y.shape[0]
        else:
            sz = _neighbors
        self.neighbors = _neighbors

        self.tree = KDTree(y)

        ydist, tmp = self.tree.query(y, k=sz)
        if self.neighbors == 1:
            tmp = tmp[:, None]

        self.Ay = np.zeros((y.shape[0], y.shape[0]))
        for i, idx in enumerate(tmp):
            for j in range(len(idx)):
                self.Ay[i, idx[j]] = self.__rbfKernel(ydist[i][j])

        xdist, _yindices = self.tree.query(x, k=sz)
        if self.neighbors == 1:
            _yindices = _yindices[:, None]

        self.Ax = np.zeros((x.shape[0], y.shape[0]))
        for i in range(xdist.shape[0]):
            for j in range(xdist.shape[1]):
                self.Ax[i, _yindices[i][j]] = self.__rbfKernel(xdist[i][j])
            
        if self.neighbors is not None:
            _yindices = np.sort(_yindices, axis=1)
            _yindices, inv = np.unique(_yindices, return_inverse=True, axis=0)
            _xindices = [[] for _ in range(len(_yindices))]
            for i, j in enumerate(inv):
                _xindices[j].append(i)
            self.yindices = _yindices
            self.xindices = _xindices

            self.Aytest = np.empty((self.y.shape[0], self.y.shape[0]))
            self.Axtest = np.empty((self.x.shape[0], self.y.shape[0]))
            for xidx, yidx in zip(self.xindices, self.yindices):
                xnbr = self.x[xidx]
                ynbr = self.y[yidx]
                for i in range(ynbr.shape[0]):
                    for j in range(i+1):
                        self.Aytest[yidx[i], yidx[j]] = self.__rbfKernel(np.linalg.norm(ynbr[i] - ynbr[j]))
                for i in range(xnbr.shape[0]):
                    for j in range(ynbr.shape[0]):
                        self.Axtest[xidx[i], yidx[j]] = self.__rbfKernel(np.linalg.norm(xnbr[i] - ynbr[j]))

    def interpolate(self, val):
        """Interpolate the given values at the interpolation points

        Parameters
        ----------
        val : array_like
            Values to interpolate, shape (n, d)

        Returns
        -------
        array_like
            Interpolated values, shape (m, d)
        """
        if val.ndim == 1:
            val = val[:, None]
        if self.neighbors is None:
            wk = np.linalg.solve(self.Ay, val)
            return self.Ax @ wk
        else:
            out = np.empty((self.x.shape[0], val.shape[1]), dtype=float)
            for xidx, yidx in zip(self.xindices, self.yindices):
                n = self.y[yidx].shape[0]
                m = self.x[xidx].shape[0]

                lhs = np.empty((n, n), dtype=float)
                for i in range(n):
                    for j in range(i+1):
                        lhs[i, j] = self.Aytest[yidx[i], yidx[j]]
                        lhs[j, i] = lhs[i, j]
                lhs = lhs.T
                rhs = val[yidx]
                coeffs = np.linalg.solve(lhs, rhs)

                vec = np.empty((m, n), dtype=float)
                for i in range(m):
                    for j in range(n):
                        vec[i, j] = self.Axtest[xidx[i], yidx[j]]
                out[xidx] = np.dot(vec, coeffs)
            return out
