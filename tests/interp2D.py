# Test the RBFInterpolator class against the SciPy implementation in 2D

# @authors: Paul Dechamps, Adrien Crovato
# @date: 2024

from matplotlib import pyplot as plt
import numpy as np
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RInterpolator import RBFInterpolator
from scipy.interpolate import RBFInterpolator as sprbf

##### Test parameters #####

n = 10           # Number of data points
p = 9            # Number of interpolation points
k = None         # Number of neighbors
kern = 'linear'  # RBF kernel
smooth = 0.      # Smoothing parameter
deg = 2

##### Test start #####
# Define the grid
x0 = np.linspace(-1, 1, n)
x1 = np.linspace(-1, 1, n)
x0, x1 = np.meshgrid(x0, x1)
x = np.vstack([x0.ravel(), x1.ravel()]).T
y = -(x[:, 0]**2 + (x[:, 1])**2)
#y = y.reshape(x0.shape)

xp = np.zeros((p, 2))
xp[:,0] = np.linspace(-0.5, 0.9, p)
xp[:,1] = np.linspace(-1, 0.5, p)

# Interpolation object
print('\033[94m' + 'pyInit' + '\033[0m')
start_pyRBF = time.time()
rbf = RBFInterpolator(x, xp, _neighbors = k, _kernel=kern, smoothing=smooth, degree=deg)
end_pyRBF = time.time()
print('\033[94m' + 'pyInterpolating' + '\033[0m')
start_pyRBFinterp = time.time()
yp = rbf.interpolate(y)[:,0]
end_pyRBFinterp = time.time()
start_pyRBFgrad = time.time()
print('\033[94m' + 'pyGradients' + '\033[0m')
yp_grad = rbf.eval_grad_mesh()
end_pyRBFgrad = time.time()

print('\033[94m' + 'pyFiniteDifference' + '\033[0m')
start_grad_fd = time.time()
dyp_dx_fd = np.zeros((xp.shape[0], x.shape[0]*x.shape[1]))
eps = 1e-5
for j in range(x.shape[0]):
    for idim in range(x.shape[1]):
        x_sav = x[j,idim]
        x[j,idim] = x[j,idim] + eps
        y_p_plus = RBFInterpolator(x, xp, _neighbors = k, _kernel=kern, smoothing=smooth, degree=deg).interpolate(y)
        x[j,idim] = x_sav - eps
        y_p_minus = RBFInterpolator(x, xp, _neighbors = k, _kernel=kern, smoothing=smooth, degree=deg).interpolate(y)
        dyp_dx_fd[:,j*x.shape[1] + idim] = ((y_p_plus - y_p_minus) / 2 / eps)[:,0]
        x[j,idim] = x_sav
end_grad_fd = time.time()

# SciPy
start_sp = time.time()
print('\033[94m' + 'pySciPyInterpolate' + '\033[0m')
ysp = sprbf(x, y, kernel=kern, neighbors=k, smoothing=smooth, degree=deg)(xp)
end_sp = time.time()

print('\033[94m' + 'pyTesting' + '\033[0m')
print('')
print(f'Maximum difference gradient (log): {np.log10(np.linalg.norm(abs((yp_grad - dyp_dx_fd)), np.inf)/np.linalg.norm(yp_grad)):.2f}')
print(f'Maximum difference SciPy (log): {np.log10(np.linalg.norm((yp - ysp), np.inf)):.2f}')
print('-------- Timers --------')
print('{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n'.format('Init:', end_pyRBF - start_pyRBF,\
                                                                     'Interpolate:', end_pyRBFinterp - start_pyRBFinterp,\
                                                                     'SciPy:', end_sp - start_sp,
                                                                     'Gradient:', end_pyRBFgrad - start_pyRBFgrad,
                                                                     'FD:', end_grad_fd - start_grad_fd))

# plot 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(xp[:,0], xp[:,1], yp, 'r', lw=5, markersize=10)
ax.plot_surface(x0, x1, y.reshape(x0.shape), edgecolor='black', alpha=0.5)
plt.title('Data points')
plt.show()
