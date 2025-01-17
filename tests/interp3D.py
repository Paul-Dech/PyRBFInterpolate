# Test the RBFInterpolator class against the SciPy implementation

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

n = 100         # Number of data points
p = 50          # Number of interpolation points
k = None          # Number of neighbors
kern = 'linear' # RBF kernel
deg = -1
smooth = 0

##### Test start #####

print('\033[94m' + 'pyStarting' + '\033[0m')

x = np.zeros((n, 2))
# Do a grid of points between 0 and 1 using repmat
x[:,0] = np.linspace(0, 1, n)
for i in range(n//10):
    x[i*n//10:(i+1)*n//10,1] = np.linspace(0, 1, n//10)
y = np.sin(2*np.pi*x[:,0]) * np.sin(2*np.pi*x[:,1])

xp = np.zeros((p, 2))
xp[:,0] = np.linspace(0.1, 0.9, p)
for i in range(p//10):
    xp[i*p//10:(i+1)*p//10,1] = np.linspace(0, 1, p//10)

# Interpolation object
start_pyRBF = time.time()
rbf = RBFInterpolator(x, xp, _neighbors = k, _kernel=kern, smoothing=smooth, degree=deg)
end_pyRBF = time.time()
print('\033[94m' + 'pyInterpolating' + '\033[0m')
start_pyRBFinterp = time.time()
yp = rbf.interpolate(y)
end_pyRBFinterp = time.time()

# SciPy
start_sp = time.time()
print('\033[94m' + 'pySciPyInterpolate' + '\033[0m')
ysp = sprbf(x, y, kernel=kern, neighbors=k, smoothing=smooth, degree=deg)(xp)
end_sp = time.time()

print('\033[94m' + 'pyTesting' + '\033[0m')
print('')
print(f'Maximum difference (log): {np.log10(np.linalg.norm((yp - ysp), np.inf)):.2f}')
print('-------- Timers --------')
print('{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n'.format('Init:', end_pyRBF - start_pyRBF,\
                                                                     'Interpolate:', end_pyRBFinterp - start_pyRBFinterp,\
                                                                     'SciPy:', end_sp - start_sp))

print(yp - ysp)
# Plot 3D with color
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x[:,0], x[:,1], y, color='blue', linewidth=0.1)
ax.plot_trisurf(xp[:,0], xp[:,1], yp[:,0], color='blue', linewidth=0.1)
ax.scatter(xp[:,0], xp[:,1], yp[:,0], label='Interpolated', color='red', s=100)
ax.scatter(xp[:,0], xp[:,1], ysp, label='Interpolated', color='black', s=100)
plt.show()
