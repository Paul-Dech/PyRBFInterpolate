# RBFInterpolator class

# @authors: Paul Dechamps, Adrien Crovato
# @date: 2024

import numpy as np
from scipy.spatial import KDTree

class RBFInterpolator:
    def __init__(self, y, x, _neighbors = None) -> None:
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
        if self.neighbors is None:
            wk = np.linalg.solve(self.Ay, val)
            return self.Ax @ wk
        else:
            out = np.empty((self.x.shape[0], val.shape[1]), dtype=float)
            for xidx, yidx in zip(self.xindices, self.yindices):
                p = self.y[yidx].shape[0]
                q = self.x[xidx].shape[0]

                lhs = np.empty((p, p), dtype=float)
                for i in range(p):
                    for j in range(i+1):
                        lhs[i, j] = self.Aytest[yidx[i], yidx[j]]
                        lhs[j, i] = lhs[i, j]
                lhs = lhs.T
                rhs = val[yidx]
                coeffs = np.linalg.solve(lhs, rhs)

                vec = np.empty((q, p), dtype=float)
                for i in range(q):
                    for j in range(p):
                        vec[i, j] = self.Axtest[xidx[i], yidx[j]]
                out[xidx] = np.dot(vec, coeffs)
            return out

    def __rbfKernel(self, r):
        return -r
