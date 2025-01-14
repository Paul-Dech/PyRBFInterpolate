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
        kernel_class = getattr(RKernels, f"{_kernel}_kernel")
        self._rbfKernel = kernel_class()

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

        xdist, _yindices = self.tree.query(x, k=sz)
        if self.neighbors == 1:
            _yindices = _yindices[:, None]

        self.Ay = np.zeros((y.shape[0], y.shape[0]))
        self.Ax = np.zeros((x.shape[0], y.shape[0]))
        if self.neighbors is None:
            # Fill Ay
            for i, idx in enumerate(tmp):
                for j in range(len(idx)):
                    self.Ay[i, idx[j]] = self._rbfKernel.eval(ydist[i][j])

            # Fill Ax
            for i in range(xdist.shape[0]):
                for j in range(xdist.shape[1]):
                    self.Ax[i, _yindices[i][j]] = self._rbfKernel.eval(xdist[i][j])

        elif self.neighbors is not None:
            _yindices = np.sort(_yindices, axis=1)
            _yindices, inv = np.unique(_yindices, return_inverse=True, axis=0)
            _xindices = [[] for _ in range(len(_yindices))]
            for i, j in enumerate(inv):
                _xindices[j].append(i)
            self.yindices = _yindices
            self.xindices = _xindices

            for xidx, yidx in zip(self.xindices, self.yindices):
                xnbr = self.x[xidx]
                ynbr = self.y[yidx]
                for i in range(ynbr.shape[0]):
                    for j in range(i+1):
                        self.Ay[yidx[i], yidx[j]] = self._rbfKernel.eval(ynbr[i], ynbr[j])
                for i in range(xnbr.shape[0]):
                    for j in range(ynbr.shape[0]):
                        self.Ax[xidx[i], yidx[j]] = self._rbfKernel.eval(xnbr[i], ynbr[j])
            self.wk = []

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
            self.wk = np.linalg.solve(self.Ay, val)
            return self.Ax @ self.wk
        else:
            out = np.empty((self.x.shape[0], val.shape[1]), dtype=float)
            for xidx, yidx in zip(self.xindices, self.yindices):
                n = self.y[yidx].shape[0]
                m = self.x[xidx].shape[0]

                lhs = np.empty((n, n), dtype=float)
                for i in range(n):
                    for j in range(i+1):
                        lhs[i, j] = self.Ay[yidx[i], yidx[j]]
                        lhs[j, i] = lhs[i, j]
                lhs = lhs.T
                rhs = val[yidx]
                coeffs = np.linalg.solve(lhs, rhs)
                self.wk.append(coeffs)

                vec = np.empty((m, n), dtype=float)
                for i in range(m):
                    for j in range(n):
                        vec[i, j] = self.Ax[xidx[i], yidx[j]]
                out[xidx] = np.dot(vec, coeffs)
            return out
    
    def eval_grad_mesh(self):
        """Evaluate the gradient of the interpolation at the interpolation points
        """
        if self.neighbors is None:
            # Check if self.wk is already computed
            if not hasattr(self, 'wk'):
                raise ValueError('Interpolation coefficients not computed')
            
            grad = np.zeros((self.x.shape[0], self.y.shape[0]))
            for k in range(self.y.shape[0]):
                #The loop is the following but it is computationally inefficient
                # dAy_yk_3 = np.zeros((self.y.shape[0], self.y.shape[0]))
                # for i in range(self.y.shape[0]):
                #     for j in range(self.y.shape[0]):
                #         if self.y[i] - self.y[j] == 0:
                #             dAy_yk_3[i, j] = 0
                #         elif i == k:
                #             dAy_yk_3[i, j] = - 1/np.sqrt((self.y[i] - self.y[j])**2) * (self.y[i] - self.y[j])
                #         elif j == k:
                #             dAy_yk_3[i, j] = 1/np.sqrt((self.y[i] - self.y[j])**2) * (self.y[i] - self.y[j])
                dAy_dyk = np.zeros((self.y.shape[0], self.y.shape[0]))
                dAy_dyk[k, :] = np.squeeze(self._rbfKernel.eval_grad(self.y[k, None], self.y)[0])
                dAy_dyk[:, k] = dAy_dyk[k, :]
                dAy_dyk[np.linalg.norm(self.y[:, None] - self.y[None, :], axis=2) == 0] = 0

                dAx_dyk = np.zeros((self.x.shape[0], self.y.shape[0]))
                dAx_dyk[:, k] = np.squeeze(self._rbfKernel.eval_grad(self.x, self.y[k, None])[1])
                dAx_dyk[np.linalg.norm(self.x[:, None] - self.y[None, :], axis=2) == 0] = 0

                dwk_dyi = np.linalg.solve(self.Ay, - dAy_dyk @ self.wk)
                grad[:, k] = (dAx_dyk @ self.wk + self.Ax @ dwk_dyi)[:,0]
            return grad
        else:
            grad = np.zeros((self.x.shape[0], self.y.shape[0]))
            cpt = 0
            for xidx, yidx in zip(self.xindices, self.yindices):
                m = self.x[xidx].shape[0]
                n = self.y[yidx].shape[0]

                lhs = np.empty((n, n), dtype=float)
                for i in range(n):
                    for j in range(i+1):
                        lhs[i, j] = self.Ay[yidx[i], yidx[j]]
                        lhs[j, i] = lhs[i, j]
                lhs = lhs.T

                vec = np.empty((m, n), dtype=float)
                for i in range(m):
                    for j in range(n):
                        vec[i, j] = self.Ax[xidx[i], yidx[j]]
                
                for k in range(n):
                    # The loop is the following but it is computationally inefficient
                    # dAy_dyk_2 = np.zeros((n, n))
                    # for i in range(n):
                    #     for j in range(n):
                    #         if self.y[yidx[i]] - self.y[yidx[j]] == 0:
                    #             dAy_dyk_2[i, j] = 0
                    #         elif i == k:
                    #             dAy_dyk_2[i,j] = -1/np.sqrt((self.y[yidx[i]] - self.y[yidx[j]])**2) * (self.y[yidx[i]] - self.y[yidx[j]])
                    #         elif j == k:
                    #             dAy_dyk_2[i,j] = 1/np.sqrt((self.y[yidx[i]] - self.y[yidx[j]])**2) * (self.y[yidx[i]] - self.y[yidx[j]])
                    dAy_dyk = np.zeros((n, n))
                    dAy_dyk[k, :] = np.squeeze(self._rbfKernel.eval_grad(self.y[yidx[k], None], self.y[yidx])[0])
                    dAy_dyk[:, k] = dAy_dyk[k, :]
                    dAy_dyk[np.linalg.norm(self.y[yidx, None] - self.y[None, yidx], axis=2) == 0] = 0
                    
                    dAx_dyk = np.zeros((m, n))
                    dAx_dyk[:, k] = np.squeeze(self._rbfKernel.eval_grad(self.x[xidx], self.y[yidx[k], None])[1])
                    dAx_dyk[np.linalg.norm(self.x[xidx, None] - self.y[None, yidx], axis=2) == 0] = 0
                    
                    dwk_dyi = np.linalg.solve(lhs, -dAy_dyk @ self.wk[cpt])
                    grad[xidx, yidx[k]] = (dAx_dyk @ self.wk[cpt] + vec @ dwk_dyi)[:,0]
                cpt = cpt + 1
            return grad
