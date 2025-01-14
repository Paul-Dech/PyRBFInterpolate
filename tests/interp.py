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

n = 100           # Number of data points
p = 30            # Number of interpolation points
k = 10            # Number of neighbors
kern = 'linear'   # RBF kernel

##### Test start #####

print('\033[94m' + 'pyStarting' + '\033[0m')

x = np.zeros((n, 1))
x[:,0] = np.linspace(0, 1, n)
y = np.sin(2*np.pi*x)

xp = np.zeros((p, 1))
xp[:,0] = np.linspace(0.1, 0.9, p)

# Interpolation object
start_pyRBF = time.time()
rbf = RBFInterpolator(x, xp, _neighbors = k, _kernel=kern)
end_pyRBF = time.time()
print('\033[94m' + 'pyInterpolating' + '\033[0m')
start_pyRBFinterp = time.time()
yp = rbf.interpolate(y)
end_pyRBFinterp = time.time()
start_pyRBFgrad = time.time()
print('\033[94m' + 'pyGradients' + '\033[0m')
yp_grad = rbf.eval_grad_mesh()
end_pyRBFgrad = time.time()


print('\033[94m' + 'pyFiniteDifference' + '\033[0m')
start_grad_fd = time.time()
dyp_dx_fd = np.zeros((p, n))
eps = 1e-8
for j in range(x.shape[0]):
    x_sav = x[j,0]
    x[j,0] = x[j,0] + eps
    y_p_plus = RBFInterpolator(x, xp, _neighbors = k, _kernel=kern).interpolate(y)
    x[j,0] = x_sav - eps
    y_p_minus = RBFInterpolator(x, xp, _neighbors = k, _kernel=kern).interpolate(y)
    dyp_dx_fd[:,j] = ((y_p_plus - y_p_minus) / 2/eps)[:,0]
    x[j,0] = x_sav
end_grad_fd = time.time()

# SciPy
start_sp = time.time()
print('\033[94m' + 'pySciPyInterpolate' + '\033[0m')
ysp = sprbf(x, y, kernel=kern, neighbors=k)(xp)
end_sp = time.time()

print('\033[94m' + 'pyTesting' + '\033[0m')
print('')
print(f'Maximum difference gradient (log): {np.log10(np.linalg.norm((yp_grad - dyp_dx_fd), np.inf)):.2f}')
print(f'Maximum difference SciPy (log): {np.log10(np.linalg.norm((yp - ysp), np.inf)):.2f}')
print('-------- Timers --------')
print('{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n'.format('Init:', end_pyRBF - start_pyRBF,\
                                                                     'Interpolate:', end_pyRBFinterp - start_pyRBFinterp,\
                                                                     'SciPy:', end_sp - start_sp,
                                                                     'Gradient:', end_pyRBFgrad - start_pyRBFgrad,
                                                                     'FD:', end_grad_fd - start_grad_fd))

plt.plot(x[:,0], y, '-', color='red', lw=2, label='Reference')
plt.plot(xp[:,0], yp, 's', fillstyle='none', markeredgewidth=1.5, color= 'blue', lw=1, label='Interpolated')
plt.plot(xp[:,0], ysp, 'x', color= 'black', markeredgewidth=2, lw=2, label='SciPy')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(frameon=False)
for side in ['top', 'right']:
    plt.gca().spines[side].set_visible(False)
plt.show()