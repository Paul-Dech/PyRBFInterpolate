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
k = None        # Number of neighbors
kern = 'linear' # RBF kernel
smoothing = 1e-8

##### Test start #####

print('\033[94m' + 'pyStarting' + '\033[0m')

# Initial value
x0 = np.loadtxt(f'{os.path.dirname(__file__)}/data/xbase_3D_fine.dat')
M0 = np.loadtxt(f'{os.path.dirname(__file__)}/data/Mbase_3D_fine.dat')
rho0 = np.loadtxt(f'{os.path.dirname(__file__)}/data/rhobase_3D_fine.dat')
v0 = np.zeros((len(M0), 3))
for idim in range(3):
    v0[:,idim] = np.loadtxt(f'{os.path.dirname(__file__)}/data/vbase{idim}_3D_fine.dat')

# Interpolation reference
xinterp = np.loadtxt(f'{os.path.dirname(__file__)}/data/xinterp_3D_fine.dat')
Minterp_ref = np.loadtxt(f'{os.path.dirname(__file__)}/data/Minterp_3D_fine.dat')
rhointerp_ref = np.loadtxt(f'{os.path.dirname(__file__)}/data/rhointerp_3D_fine.dat')
vinterp_ref = np.zeros((len(Minterp_ref), 3))
for idim in range(3):
    vinterp_ref[:,idim] = np.loadtxt(f'{os.path.dirname(__file__)}/data/vinterp{idim}_3D_fine.dat')

# Interpolation object
start_pyRBF = time.time()
rbf = RBFInterpolator(x0, xinterp, _neighbors = k, _kernel=kern)
end_pyRBF = time.time()
print('\033[94m' + 'pyInterpolating' + '\033[0m')
start_pyRBFinterp = time.time()
rhointerp = rbf.interpolate(rho0).ravel()
Minterp = rbf.interpolate(M0).ravel()
vinterp = np.zeros((len(Minterp), 3))
for idim in range(3):
    vinterp[:,idim] = rbf.interpolate(v0[:,idim]).ravel()
end_pyRBFinterp = time.time()

# SciPy
start_sp = time.time()
print('\033[94m' + 'pySciPyInterpolate' + '\033[0m')
rhointerp_sp = sprbf(x0, rho0, kernel=kern, neighbors=k, smoothing=smoothing, degree=0)(xinterp).ravel()
Minterp_sp = sprbf(x0, M0, kernel=kern, neighbors=k, smoothing=smoothing, degree=0)(xinterp).ravel()
vinterp_sp = np.zeros((len(Minterp_sp), 3))
for idim in range(3):
    vinterp_sp[:,idim] = sprbf(x0, v0[:,idim], kernel=kern, neighbors=k, smoothing=smoothing, degree=0)(xinterp).ravel()
end_sp = time.time()

print('\033[94m' + 'pyTesting' + '\033[0m')
print('')
print(f'Maximum difference SciPy (log): {np.log10(np.linalg.norm((Minterp - Minterp_sp), np.inf)):.2f}')
print(f'Maximum difference Ref (log): {np.log10(np.linalg.norm((Minterp - Minterp_ref), np.inf)):.2f}')
print('-------- Timers --------')
print('{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n{:<15s}{:<15.6f}\n'.format('Init:', end_pyRBF - start_pyRBF,\
                                                                     'Interpolate:', end_pyRBFinterp - start_pyRBFinterp,\
                                                                     'SciPy:', end_sp - start_sp))

plt.figure()
#plt.plot(x0[:,0], M0, '-', color='black', lw=2, label='Initial')
plt.plot(xinterp[:,0], Minterp_ref, 'o--', color='red', lw=2, label='Reference interpolation')
plt.plot(xinterp[:,0], Minterp, 's', fillstyle='none', markeredgewidth=1.5, color= 'blue', lw=1, label='Interpolated')
plt.plot(xinterp[:,0], Minterp_sp, 'x', color= 'black', markeredgewidth=2, lw=2, label='SciPy')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(frameon=False)
for side in ['top', 'right']:
    plt.gca().spines[side].set_visible(False)
plt.title('Mach number')
plt.draw()

plt.figure()
#plt.plot(x0[:,0], rho0, '-', color='black', lw=2, label='Initial')
plt.plot(xinterp[:,0], rhointerp_ref, 'o--', color='red', lw=2, label='Reference interpolation')
plt.plot(xinterp[:,0], rhointerp, 's', fillstyle='none', markeredgewidth=1.5, color= 'blue', lw=1, label='Interpolated')
plt.plot(xinterp[:,0], rhointerp_sp, 'x', color= 'black', markeredgewidth=2, lw=2, label='SciPy')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(frameon=False)
for side in ['top', 'right']:
    plt.gca().spines[side].set_visible(False)
plt.title('Density')
plt.draw()

plt.figure()
#plt.plot(x0[:,0], v0[:,0], '-', color='black', lw=2, label='Initial')
plt.plot(xinterp[:,0], vinterp_ref[:,0], 'o--', color='red', lw=2, label='Reference interpolation')
plt.plot(xinterp[:,0], vinterp[:,0], 's', fillstyle='none', markeredgewidth=1.5, color= 'blue', lw=1, label='Interpolated')
plt.plot(xinterp[:,0], vinterp_sp[:,0], 'x', color= 'black', markeredgewidth=2, lw=2, label='SciPy')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(frameon=False)
for side in ['top', 'right']:
    plt.gca().spines[side].set_visible(False)
plt.title('Velocity x')
plt.draw()

plt.figure()
#plt.plot(x0[:,0], v0[:,1], '-', color='black', lw=2, label='Initial')
plt.plot(xinterp[:,0], vinterp_ref[:,1], 'o--', color='red', lw=2, label='Reference interpolation')
plt.plot(xinterp[:,0], vinterp[:,1], 's', fillstyle='none', markeredgewidth=1.5, color= 'blue', lw=1, label='Interpolated')
plt.plot(xinterp[:,0], vinterp_sp[:,1], 'x', color= 'black', markeredgewidth=2, lw=2, label='SciPy')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(frameon=False)
for side in ['top', 'right']:
    plt.gca().spines[side].set_visible(False)
plt.title('Velocity y')
plt.draw()

plt.figure()
#plt.plot(x0[:,0], v0[:,2], '-', color='black', lw=2, label='Initial')
plt.plot(xinterp[:,0], vinterp_ref[:,2], 'o--', color='red', lw=2, label='Reference interpolation')
plt.plot(xinterp[:,0], vinterp[:,2], 's', fillstyle='none', markeredgewidth=1.5, color= 'blue', lw=1, label='Interpolated')
plt.plot(xinterp[:,0], vinterp_sp[:,2], 'x', color= 'black', markeredgewidth=2, lw=2, label='SciPy')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(frameon=False)
for side in ['top', 'right']:
    plt.gca().spines[side].set_visible(False)
plt.title('Velocity z')
plt.draw()

plt.show()
