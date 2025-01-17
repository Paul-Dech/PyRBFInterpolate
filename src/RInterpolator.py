# RBFInterpolator class

# @authors: Paul Dechamps, Adrien Crovato
# @date: 2024

import numpy as np
from scipy.spatial import KDTree
import importlib

from itertools import combinations_with_replacement
from scipy.special import comb

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
    def __init__(self, y, x, _neighbors = None, _kernel='linear', smoothing=0., degree=None) -> None:

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
        self.smoothing = smoothing

        if degree < -1:
            raise ValueError(f"Degree must be greater than or equal to -1, got {degree}")
        ndim = y.shape[1]
        self.powers = self._monomial_powers(ndim, degree)
        if self.powers.shape[0] > sz:
            raise ValueError(f"Degree {degree} in {ndim} dimensions requires at least {self.powers.shape[0]} neighbors.")
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
            q = self.x.shape[0]
            p = self.y.shape[0]
            r = self.powers.shape[0]

            yhat = self.y
            self.lhs = np.empty((p + r, p + r), dtype=float).T
            # Kernel
            for i, idx in enumerate(tmp):
                for j in range(len(idx)):
                    self.lhs[i, idx[j]] = self._rbfKernel.eval(ydist[i][j])
            # Smoothing
            for i in range(p):
                self.lhs[i, i] = self.lhs[i, i] + self.smoothing
            # Polynomial
            for i in range(p):
                for j in range(r):
                    self.lhs[i, p + j] = np.prod(yhat[i]**self.powers[j])
            self.lhs[p:, :p] = self.lhs[:p, p:].T
            self.lhs[p:, p:] = 0.0

            # Evaluation coefficients
            #xhat = (self.x - self.shift) / self.scale
            xhat = self.x
            self.vec = np.empty((q, p + r), dtype=float)
            for i in range(xdist.shape[0]):
                for j in range(xdist.shape[1]):
                    self.vec[i, _yindices[i][j]] = self._rbfKernel.eval(xdist[i][j])
                for j in range(r):
                    self.vec[i, p+j] = np.prod(xhat[i]**self.powers[j])

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
            p = val.shape[0]
            s = val.shape[1]
            r = self.powers.shape[0]

            # RHS of the system
            rhs = np.empty((s, p+r), dtype=float).T
            rhs[:p] = val
            rhs[p:] = 0.0
            # Evaluate RBF weights
            self.wk = np.linalg.solve(self.lhs, rhs)

            out = np.dot(self.vec, self.wk)
        else:
            out = np.empty((self.x.shape[0], val.shape[1]), dtype=float)
            for xidx, yidx in zip(self.xindices, self.yindices):
                n = self.y[yidx].shape[0]
                m = self.x[xidx].shape[0]
                r = self.powers.shape[0]
                s = val.shape[1]

                yhat = self.y[yidx]
                lhs = np.empty((n+r, n+r), dtype=float).T
                # Kernel
                for i in range(n):
                    for j in range(i+1):
                        lhs[i, j] = self.Ay[yidx[i], yidx[j]]
                        lhs[j, i] = lhs[i, j]
                # Polynomial
                for i in range(n):
                    for j in range(r):
                        lhs[i, n + j] = np.prod(yhat[i]**self.powers[j])
                # Smoothing
                for i in range(n):
                    lhs[i, i] = lhs[i, i] + self.smoothing
                lhs[n:, :n] = lhs[:n, n:].T
                lhs[n:, n:] = 0.0

                rhs = np.empty((s, n+r), dtype=float).T
                rhs[:n] = val[yidx]
                rhs[n:] = 0.0
                coeffs = np.linalg.solve(lhs, rhs)
                self.wk.append(coeffs)

                # Evaluation coefficient
                #xhat = (self.x[xidx] - shift) / scale
                xhat = self.x[xidx]
                vec = np.empty((m, n+r), dtype=float)
                for i in range(m):
                    for j in range(n):
                        vec[i, j] = self.Ax[xidx[i], yidx[j]]
                    for j in range(r):
                        vec[i, n+j] = np.prod(xhat[i]**self.powers[j])
                out[xidx] = np.dot(vec, coeffs)
        return out
    
    def eval_grad_mesh(self):
        """Evaluate the gradient of the interpolation at the interpolation points
        """
        if self.neighbors is None:
            # Check if self.wk is already computed
            if not hasattr(self, 'wk'):
                raise ValueError('Interpolation coefficients not computed')
            
            p = self.y.shape[0]
            s = self.y.shape[1]
            q = self.x.shape[0]
            r = self.powers.shape[0]

            yhat = self.y
            xhat = self.x
            grad = np.zeros((q, p*s))
            for k in range(p):
                for idim in range(s):
                    dAy_dyk = np.zeros((p, p))
                    dAy_dyk[k, :] = self._rbfKernel.eval_grad(self.y[k, None], self.y)[0][:,idim]
                    dAy_dyk[:, k] = dAy_dyk[k, :]
                    dAy_dyk[np.linalg.norm(self.y[:, None] - self.y[None, :], axis=2) == 0] = 0

                    dAx_dyk = np.zeros((q, p))
                    dAx_dyk[:, k] = self._rbfKernel.eval_grad(self.x, self.y[k, None])[1][:,idim]
                    dAx_dyk[np.linalg.norm(self.x[:, None] - self.y[None, :], axis=2) == 0] = 0

                    # dPyhat_dyk
                    dPy_dyk = np.zeros((p, r), dtype=float)
                    for j in range(self.powers.shape[0]):
                        if self.powers[j, idim] - 1 == 0 and yhat[k, idim]**self.powers[j, idim] == 0:
                            dPy_dyk[k, j] = 1
                            continue
                        if yhat[k, idim] == 0 and self.powers[j, idim] == 0:
                            dPy_dyk[k, j] = 0
                            continue
                        dPy_dyk[k, j] = self.powers[j, idim] * yhat[k, idim]**(self.powers[j, idim] - 1) * self.lhs[k, p+j]
                        if (yhat[k, idim]**(self.powers[j, idim])) != 0:
                            dPy_dyk[k, j] /= (yhat[k, idim]**(self.powers[j, idim]))

                    # dPxhat_dyk
                    dPx_dyk = np.zeros((q, r), dtype=float)

                    rhs_grad = np.empty((s, p+r), dtype=float).T
                    rhs_grad[:p] = -np.dot(dAy_dyk, self.wk[:p]) - np.dot(dPy_dyk, self.wk[p:])
                    rhs_grad[p:] = - np.dot(dPy_dyk.T, self.wk[:p])
                    # Not that lhs_grad = lhs
                    dwk_dyi = np.linalg.solve(self.lhs, rhs_grad)
                    grad[:, k*s+idim] = (np.dot(dAx_dyk, self.wk[:p]) + np.dot(self.vec[:q, :p], dwk_dyi[:p]) + np.dot(dPx_dyk, self.wk[p:]) + np.dot(self.vec[:q, p:], dwk_dyi[p:]))[:,0]
            return grad
        else:
            grad = np.zeros((self.x.shape[0], self.y.shape[0]*self.y.shape[1]))
            s = self.y.shape[1]
            r = self.powers.shape[0]
            cpt = 0
            for xidx, yidx in zip(self.xindices, self.yindices):
                m = self.x[xidx].shape[0]
                n = self.y[yidx].shape[0]

                yhat = self.y[yidx]
                lhs = np.empty((n+r, n+r), dtype=float).T
                # Kernel
                for i in range(n):
                    for j in range(i+1):
                        lhs[i, j] = self.Ay[yidx[i], yidx[j]]
                        lhs[j, i] = lhs[i, j]
                # Polynomial
                for i in range(n):
                    for j in range(r):
                        lhs[i, n + j] = np.prod(yhat[i]**self.powers[j])
                # Smoothing
                for i in range(n):
                    lhs[i, i] = lhs[i, i] + self.smoothing
                lhs[n:, :n] = lhs[:n, n:].T
                lhs[n:, n:] = 0.0

                xhat = self.x[xidx]
                vec = np.empty((m, n+r), dtype=float)
                for i in range(m):
                    for j in range(n):
                        vec[i, j] = self.Ax[xidx[i], yidx[j]]
                    for j in range(r):
                        vec[i, n+j] = np.prod(xhat[i]**self.powers[j])
                
                for k in range(n):
                    for idim in range(self.y.shape[1]):
                        dAy_dyk = np.zeros((n, n))
                        dAy_dyk[k, :] = self._rbfKernel.eval_grad(self.y[yidx[k], None], self.y[yidx])[0][:,idim]
                        dAy_dyk[:, k] = dAy_dyk[k, :]
                        dAy_dyk[np.linalg.norm(self.y[yidx, None] - self.y[None, yidx], axis=2) == 0] = 0
                        
                        dAx_dyk = np.zeros((m, n))
                        dAx_dyk[:, k] = self._rbfKernel.eval_grad(self.x[xidx], self.y[yidx[k], None])[1][:,idim]
                        dAx_dyk[np.linalg.norm(self.x[xidx, None] - self.y[None, yidx], axis=2) == 0] = 0

                        # dPyhat_dyk
                        dPy_dyk = np.zeros((n, r), dtype=float)
                        for j in range(self.powers.shape[0]):
                            if self.powers[j, idim] - 1 == 0 and yhat[k, idim]**self.powers[j, idim] == 0:
                                dPy_dyk[k, j] = 1
                                continue
                            if yhat[k, idim] == 0 and self.powers[j, idim] == 0:
                                dPy_dyk[k, j] = 0
                                continue
                            dPy_dyk[k, j] = self.powers[j, idim] * yhat[k, idim]**(self.powers[j, idim] - 1) * lhs[k, n+j]
                            if (yhat[k, idim]**(self.powers[j, idim])) != 0:
                                dPy_dyk[k, j] /= (yhat[k, idim]**(self.powers[j, idim]))
                        
                        # dPy_dyk_fd = np.zeros((n, r), dtype=float)
                        # Py_plus = np.zeros((n, r), dtype=float)
                        # Py_minus = np.zeros((n, r), dtype=float)
                        # eps = 1e-8
                        # save = yhat[k, idim]
                        # yhat[k, idim] = yhat[k, idim] + eps
                        # for i in range(n):
                        #     for j in range(r):
                        #         Py_plus[i, j] = np.prod(yhat[i]**self.powers[j])
                        # yhat[k, idim] = save - eps
                        # for i in range(n):
                        #     for j in range(r):
                        #         Py_minus[i, j] = np.prod(yhat[i]**self.powers[j])
                        # yhat[k, idim] = save
                        # dPy_dyk_fd = (Py_plus - Py_minus) / (2*eps)
                        # print(dPy_dyk)
                        # print(dPy_dyk_fd)
                        # print(np.linalg.norm(dPy_dyk - dPy_dyk_fd))


                        # dPxhat_dyk
                        dPx_dyk = np.zeros((m, r), dtype=float)

                        rhs_grad = np.empty((s, n+r), dtype=float).T
                        rhs_grad[:n] = -np.dot(dAy_dyk, self.wk[cpt][:n]) - np.dot(dPy_dyk, self.wk[cpt][n:])
                        rhs_grad[n:] = - np.dot(dPy_dyk.T, self.wk[cpt][:n])
                        
                        dwk_dyi = np.linalg.solve(lhs, rhs_grad)
                        # print('np.dot(dAx_dyk, self.wk[cpt][:n])', np.dot(dAx_dyk, self.wk[cpt][:n]).shape)
                        # print('np.dot(vec[:m, :n], dwk_dyi[:n])', np.dot(vec[:m, :n], dwk_dyi[:n]).shape)
                        # print('np.dot(dPx_dyk, self.wk[cpt][n:])', np.dot(dPx_dyk, self.wk[cpt][n:]).shape)
                        # print('np.dot(vec[:m, n:], dwk_dyi[n:])', np.dot(vec[:m, n:], dwk_dyi[n:]).shape)
                        grad[xidx, yidx[k]*s+idim] = (np.dot(dAx_dyk, self.wk[cpt][:n]) + np.dot(vec[:m, :n], dwk_dyi[:n]) + np.dot(dPx_dyk, self.wk[cpt][n:]) + np.dot(vec[:m, n:], dwk_dyi[n:]))[:,0]
                cpt = cpt + 1
            return grad
        
    def _monomial_powers(self, ndim, degree):
        """Return the powers for each monomial in a polynomial.

        Parameters
        ----------
        ndim : int
            Number of variables in the polynomial.
        degree : int
            Degree of the polynomial.

        Returns
        -------
        (nmonos, ndim) int ndarray
            Array where each row contains the powers for each variable in a
            monomial.

        """
        nmonos = comb(degree + ndim, ndim, exact=True)
        out = np.zeros((nmonos, ndim), dtype=np.dtype("long"))
        count = 0
        for deg in range(degree + 1):
            for mono in combinations_with_replacement(range(ndim), deg):
                # `mono` is a tuple of variables in the current monomial with
                # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
                for var in mono:
                    out[count, var] += 1

                count += 1
        return out
