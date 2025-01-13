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
        
        # Check if points of y and x are in the same place
        # if np.any(np.isclose(y[:, np.newaxis], x)):
        #     raise ValueError("Data points and interpolation points must be different")
        
        RKernels = importlib.import_module("src.RKernels")
        self.__rbfKernel = getattr(RKernels, _kernel)
        self.__rbfKernel_grad = getattr(RKernels, f"{_kernel}_grad")

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

        """# Derivative of the distance with respect to the coordinates
        dryy_dy = np.zeros((y.shape[0], y.shape[0], y.shape[0]))
        for i, idx in enumerate(tmp):
            for j in range(len(idx)):
                if i == idx[j]:
                    continue
                dryy_dy[i, idx[j], i] = 1/(ydist[i][j]) * (y[i] - y[idx[j]])

        # Derivative of the RBF kernel with respect to the distance
        Ky_grad = np.zeros((y.shape[0], y.shape[0], y.shape[0]))
        for i, idx in enumerate(tmp):
            for j in range(len(idx)):
                Ky_grad[i, idx[j], i] = self.__rbfKernel_grad(ydist[i][j])
        
        # Chain rule
        self.Ay_grad = np.zeros((y.shape[0], y.shape[0], y.shape[0]))
        for i in range(y.shape[0]):
            self.Ay_grad[:,:,i] = Ky_grad[:,:,i] * dryy_dy[:,:,i]
        
        drxy_dy = np.zeros((x.shape[0], y.shape[0], y.shape[0]))
        for i in range(xdist.shape[0]):
            for j in range(xdist.shape[1]):
                if xdist[i][j] != 0:
                    drxy_dy[i, _yindices[i][j], i] = -(x[i] - y[_yindices[i][j]]) / xdist[i][j]
                else:
                    drxy_dy[i, _yindices[i][j], i] = 1

        Kx_grad = np.zeros((x.shape[0], y.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                Kx_grad[i, _yindices[i][j], i] = self.__rbfKernel_grad(xdist[i][j])

        # Chain rule
        self.Ax_grad = np.zeros((x.shape[0], y.shape[0], y.shape[0]))
        for i in range(y.shape[0]):
            self.Ax_grad[:,:,i] = Kx_grad[:,:,i] * drxy_dy[:,:,i]"""

        """# Check drdy with finite difference
        #print('base', ydist)
        drdy_fd = np.zeros((y.shape[0], y.shape[0]))
        eps = 1e-6
        for i in range(y.shape[0]):
            y_save = y[i].copy()
            y[i] = y_save + eps
            self.tree_iii = KDTree(y)
            ydist_iii, tmp_iii = self.tree_iii.query(y, k=sz)
            for j in range(len(tmp_iii[i])):
                if i == tmp_iii[i][j]:
                    continue
                drdy_fd[i, tmp_iii[i][j]] += (ydist_iii[i][j] - ydist[i][j]) / eps
            y[i] = y_save
        
        #print('dryy_dy an', dryy_dy)
        #print('dryy_dy fd', drdy_fd)
        print('norm dryy_dy', np.linalg.norm(dryy_dy - drdy_fd))
        
        # Check dK/dy with finite difference
        Ky_grad_fd = np.zeros((y.shape[0], y.shape[0]))
        eps = 1e-4
        for i, idx in enumerate(tmp):
            for j in range(len(idx)):
                dist_save = ydist[i][j]
                dist_plus = y[i] + eps
                dist_minus = y[i] - eps
                Ky_grad_fd[i, idx[j]] = (self.__rbfKernel(dist_plus) - self.__rbfKernel(dist_minus)) /( 2 * eps )
                ydist[i][j] = dist_save
        #print('Ky fd', Ky_grad_fd)
        #print('Ky an', Ky_grad)
        print('norm Ky', np.linalg.norm(Ky_grad - Ky_grad_fd))

        # Check dAy/dy with finite difference
        #print('ref Ay', self.Ay)
        dAy_fd = np.zeros((y.shape[0], y.shape[0]))
        eps = 1e-6
        for i in range(y.shape[0]):
            y_save = y[i].copy()
            y[i] = y_save + eps
            self.tree_iii = KDTree(y)
            ydist_iii, tmp_iii = self.tree_iii.query(y, k=sz)
            for j in range(len(tmp_iii[i])):
                if i == tmp_iii[i][j]:
                    continue
                dAy_fd[i, tmp_iii[i][j]] += (self.__rbfKernel(ydist_iii[i][j]) - self.__rbfKernel(ydist[i][j])) / eps
            y[i] = y_save

        #print('Ay an', self.Ay_grad)
        #print('Ay fd', dAy_fd)
        print('norm dAy/dy', np.linalg.norm(self.Ay_grad - dAy_fd))

        # Check drxy/dy with finite difference
        #print('base', self.Ax)
        drxy_dy_fd = np.zeros((x.shape[0], y.shape[0]))
        eps = 1e-6
        # Loop over each point in y
        for k in range(y.shape[0]):
            y_save = y[k].copy()
            y[k] = y_save + eps
            self.tree_iii = KDTree(y)
            xdist_iii, yindices_iii = self.tree_iii.query(x, k=sz)
            for i in range(x.shape[0]):
                for j in range(xdist.shape[1]):
                    # Need to find the new index because query sorts by distance. Since
                    # we do finite difference, the closest point might change (print yindices_iii and yindices
                    # to spot the difference)
                    # Find where yindices_iii[i] == yindices[i][j]
                    idx = np.where(yindices_iii[i] == _yindices[i][j])[0]
                    drxy_dy_fd[i, _yindices[i][j]] += (xdist_iii[i][idx] - xdist[i][j]) / eps
            y[k] = y_save
        #print('ref dryy_dy', dryy_dy)
        #print('dryx_dy an', drxy_dy)
        #print('dryx_dy fd', drxy_dy_fd)
        diff = drxy_dy - drxy_dy_fd
        #diff[abs(diff) < 1e-2] = 0
        print('diff', diff)
        print('norm dryx_dy', np.linalg.norm(drxy_dy - drxy_dy_fd))
        #quit()

        Kx_grad_fd = np.zeros((x.shape[0], y.shape[0]))
        eps = 1e-4
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                dist_save = xdist[i][j]
                dist_plus = xdist[i][j] + eps
                dist_minus = xdist[i][j] - eps
                Kx_grad_fd[i, _yindices[i][j]] = (self.__rbfKernel(dist_plus) - self.__rbfKernel(dist_minus)) /( 2 * eps )
                xdist[i][j] = dist_save
        
        #print('Kx_grad an', Kx_grad)
        #print('Kx_grad fd', Kx_grad_fd)
        print('norm Kx', np.linalg.norm(Kx_grad - Kx_grad_fd))

        # Check dAx/dy with finite difference
        #print('ref Ax', self.Ax)
        dAx_fd = np.zeros((x.shape[0], y.shape[0]))
        eps = 1e-6
        for k in range(y.shape[0]):
            y_save = y[k].copy()
            y[k] = y_save + eps
            self.tree_iii = KDTree(y)
            xdist_iii, yindices_iii = self.tree_iii.query(x, k=sz)
            for i in range(x.shape[0]):
                for j in range(xdist.shape[1]):
                    idx = np.where(yindices_iii[i] == _yindices[i][j])[0]
                    dAx_fd[i, _yindices[i][j]] += (self.__rbfKernel(xdist_iii[i][idx]) - self.__rbfKernel(xdist[i][j])) / eps
            y[k] = y_save

        #print('dAx_dy an', self.Ax_grad)
        #print('dAx_dy fd', dAx_fd)
        print('dAx/dy norm', np.linalg.norm(self.Ax_grad - dAx_fd))"""

        """if self.neighbors is not None:
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
                        self.Axtest[xidx[i], yidx[j]] = self.__rbfKernel(np.linalg.norm(xnbr[i] - ynbr[j]))"""

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
    
    def evalGrad(self, val):
        """Evaluate the gradient of the interpolation at the interpolation points
        """
        if val.ndim == 1:
            val = val[:, None]
        if self.neighbors is None:
            # Check if self.wk is already computed
            if not hasattr(self, 'wk'):
                raise ValueError('Interpolation coefficients not computed')
            
            grad = np.zeros((self.x.shape[0], self.y.shape[0]))
            for k in range(self.y.shape[0]):
                # The loop is the following but it is computationally inefficient
                """dAy_yk_2 = np.zeros((self.y.shape[0], self.y.shape[0]))
                for i in range(self.y.shape[0]):
                    for j in range(self.y.shape[0]):
                        if self.y[i] - self.y[j] == 0:
                            dAy_yk_2[i, j] = 0
                        elif i == k:
                            dAy_yk_2[i, j] = - 1/np.sqrt((self.y[i] - self.y[j])**2) * (self.y[i] - self.y[j])
                        elif j == k:
                            dAy_yk_2[i, j] = 1/np.sqrt((self.y[i] - self.y[j])**2) * (self.y[i] - self.y[j])"""

                diff_matrix = self.y.flatten()[:, None] - self.y.flatten()[None, :]
                denominator = np.sqrt(diff_matrix**2)
                dAy_yk = np.zeros_like(diff_matrix)
                dAy_yk[k, :] = -1 / denominator[k, :] * diff_matrix[k, :]
                dAy_yk[:, k] = -1 / denominator[k, :] * diff_matrix[k, :]
                dAy_yk[diff_matrix == 0] = 0

                # Ensure diagonal is zero
                #dAy_yk_2[np.diag_indices_from(dAy_yk_2)] = 0
                #print('dAy_yk', dAy_yk)
                #print('dAy_yk_2', dAy_yk_2)
                """print('dAy_dyk Time 1', end_time_y1 - start_time_y1)
                print('dAy_dyk Time 2', end_time_y2 - start_time_y2)
                print('norm dAy_yk - dAy_yk_2', np.linalg.norm(dAy_yk - dAy_yk_2))"""
                                
                
                """# Verify dAy_yk by finite difference
                eps = 1e-6
                dAy_yk_plus = np.zeros((self.y.shape[0], self.y.shape[0]))
                save = self.y[k].copy()
                self.y[k] = save + eps
                for i in range(self.y.shape[0]):
                    for j in range(self.y.shape[0]):
                        dAy_yk_plus[i, j] = self.__rbfKernel(np.linalg.norm(self.y[i] - self.y[j]))
                dAdy_fd = (dAy_yk_plus - self.Ay) / eps
                self.y[k] = save
                print('k = ', k)
                print('norm dAy_yk_an - dAy_yk_fd', np.linalg.norm(dAdy_fd - dAy_yk))"""
                
                # dAx_yk_2 = np.zeros((self.x.shape[0], self.y.shape[0]))
                # for i in range(self.x.shape[0]):
                #     for j in range(self.y.shape[0]):
                #         if self.x[i] - self.y[j] == 0:
                #             dAx_yk_2[i, j] = 0
                #         elif j == k:
                #             dAx_yk_2[i, j] = 1/np.sqrt((self.x[i] - self.y[j])**2) * (self.x[i] - self.y[j])


                # Compute the difference matrix
                diff_matrix = self.x.flatten()[:, None] - self.y.flatten()[None, :]
                denominator = np.sqrt(diff_matrix**2)
                denominator[denominator == 0] = np.inf
                dAx_yk = np.zeros_like(diff_matrix)
                dAx_yk[:, k] = 1 / denominator[:, k] * diff_matrix[:, k]
                dAx_yk[diff_matrix == 0] = 0


                # Ensure diagonal is zero
                #dAy_yk_2[np.diag_indices_from(dAy_yk_2)] = 0
                #print('dAx_yk', dAx_yk)
                #print('dAx_yk_2', dAx_yk_2)
                # print('dAx_dyk Time 1', end_time_x1 - start_time_x1)
                # print('dAx_dyk Time 2', end_time_x2 - start_time_x2)
                # print('norm dAx_yk - dAx_yk_2', np.linalg.norm(dAx_yk - dAx_yk_2))
                
                # Verify dAx_yk by finite difference
                """eps = 1e-6
                dAx_yk_plus = np.zeros((self.x.shape[0], self.y.shape[0]))
                save = self.y[k].copy()
                self.y[k] = save + eps
                for i in range(self.x.shape[0]):
                    for j in range(self.y.shape[0]):
                        dAx_yk_plus[i, j] = self.__rbfKernel(np.linalg.norm(self.x[i] - self.y[j]))

                dAx_yk_fd = (dAx_yk_plus - self.Ax) / eps
                self.y[k] = save
                #print('dAx_yk fd', dAx_yk_fd)
                #print('dAx_yk an', dAx_yk)
                print('norm diff Ax fd - Ax an', np.linalg.norm(dAx_yk_fd - dAx_yk))"""

                dwk_dyi = np.linalg.solve(self.Ay, - dAy_yk @ self.wk)
                grad[:, k] = (dAx_yk @ self.wk + self.Ax @ dwk_dyi)[:,0]
            return grad
