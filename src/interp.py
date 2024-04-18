# Test the RBFInterpolator class against the SciPy implementation

# @authors: Paul Dechamps, Adrien Crovato
# @date: 2024

from matplotlib import pyplot as plt
import numpy as np
from RBFInterpolator import RBFInterpolator
from scipy.interpolate import RBFInterpolator as sprbf
import time

n = 100     # Number of data points
p = 50      # Number of interpolation points
k = 10      # Number of neighbors

print('\033[94m' + 'pyStarting' + '\033[0m')

x = np.zeros((n, 1))
x[:,0] = np.linspace(0, 1, n)
y = np.sin(2*np.pi*x)

xp = np.zeros((p, 1))
xp[:,0] = np.linspace(0, 1, p)

# Interpolation object
start_pyRBF = time.time()
rbf = RBFInterpolator(x, xp, _neighbors = k)
end_pyRBF = time.time()
print('\033[94m' + 'pyInterpolating' + '\033[0m')
start_pyRBFinterp = time.time()
yp = rbf.interpolate(y)
end_pyRBFinterp = time.time()

# SciPy
start_sp = time.time()
print('\033[94m' + 'pySciPyInterpolate' + '\033[0m')
ysp = sprbf(x, y[:,0], kernel='linear', neighbors=k)(xp)
end_sp = time.time()

print('\033[94m' + 'pyTesting' + '\033[0m')
print('Maximum difference', np.linalg.norm((yp.reshape(-1) - ysp), np.inf))
print('pyRBFInterpolator init took {:.6f} seconds'.format(end_pyRBF - start_pyRBF))
print('pyRBFInterpolator interpolate took {:.6f} seconds'.format(end_pyRBFinterp - start_pyRBFinterp))
print("SciPy interpolation took {:.6f} seconds".format(end_sp - start_sp))

plt.plot(x[:,0], y[:,0], color='red', lw=2, label='Reference')
plt.plot(xp[:,0], yp, 'x--', color= 'blue', lw=1, label='Interpolated')
plt.plot(xp[:,0], ysp, 'o--', color= 'green', lw=0.5, label='SciPy')
plt.legend()
plt.show()