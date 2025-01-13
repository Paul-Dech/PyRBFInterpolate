from matplotlib import pyplot as plt
import numpy as np
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RInterpolator import RBFInterpolator
from scipy.interpolate import RBFInterpolator as sprbf

# Interpolate from y to x
n = 100
p = 20
y = np.zeros((n, 1))
x = np.zeros((p, 1))
y[:,0] = np.linspace(0., 1., n)
x[:,0] = np.linspace(0., 1., p)

val = np.sin(2*np.pi*y)

rbf = RBFInterpolator(y, x, _neighbors = None, _kernel='linear')
fx = rbf.interpolate(val)

# Time this operation
start_an = time.time()

dfx_dy_an = rbf.evalGrad(val)

end_an = time.time()

# Finite difference
start_fd = time.time()
eps = 1e-8
dfx_dy = np.zeros((p, n))
for i in range(n):
    save = y[i,0]
    y[i,0] = save + eps
    fx1 = RBFInterpolator(y, x, _neighbors = None, _kernel='linear').interpolate(val)
    y[i,0] = save - eps
    fx2 = RBFInterpolator(y, x, _neighbors = None, _kernel='linear').interpolate(val)
    dfx_dy[:,i] = (fx1 - fx2)[:,0] / (2*eps)
    y[i,0] = save

end_fd = time.time()


#print('Analytical gradient \n', dfx_dy_an)
#print('Finite difference \n', dfx_dy)
#print('Difference \n', dfx_dy_an - dfx_dy)
print('norm difference \n', np.linalg.norm(dfx_dy_an - dfx_dy))
if np.linalg.norm(dfx_dy_an - dfx_dy) < 1e-4:
    print('\033[92m' + 'Test passed' + '\033[0m')
else:
    print('\033[91m' + 'Test failed' + '\033[0m')

print('Time analytical', end_an - start_an)
print('Time finite difference', end_fd - start_fd)
print('Ratio', abs((end_fd - start_fd)) / (end_an - start_an))

plt.figure()
plt.plot(y[:,0], np.zeros(len(y)), 'x')
plt.plot(x[:,0], np.zeros(len(x)),  'o')
plt.title('Points')
plt.draw()

plt.figure()
plt.plot(y[:,0], val[:,0], 'x-', label='baseline')
plt.plot(x[:,0], fx[:,0], 'o--', label='interpolated')
plt.title('Interpolation')
plt.legend(frameon=False)
plt.draw()

plt.show()


quit()