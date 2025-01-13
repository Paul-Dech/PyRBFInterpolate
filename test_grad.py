import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve

def rbf_kernel(r, epsilon=1.0):
    return -r

def rbf_kernel_gradient(x, y, epsilon=1.0):
    r = np.linalg.norm(x - y)
    return -1

def compute_interpolation_and_gradient(x, y, val, epsilon=1.0):
    # Compute distances
    r_yy = cdist(y[:, None], y[:, None], 'euclidean')
    r_xy = cdist(x[:, None], y[:, None], 'euclidean')
    
    # Compute kernel matrices
    K_yy = rbf_kernel(r_yy, epsilon)
    K_xy = rbf_kernel(r_xy, epsilon)
    
    # Solve for weights a
    a = np.linalg.solve(K_yy, val)

    # Compute interpolated values f(x)
    print('K_xy', K_xy)
    f_x = K_xy @ a
    
    # Compute gradient df/dy
    df_dy = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            df_dy[i, j] = rbf_kernel_gradient(x[i], y[j], epsilon) * a[j]
    
    return f_x, df_dy

# Example usage
n = 5  # Number of nodes in y
m = 3  # Number of elements in x

y = np.linspace(0, 1, n)
val = np.sin(2 * np.pi * y)
x = np.linspace(0, 1, m)

f_x, df_dy = compute_interpolation_and_gradient(x, y, val)

print("Interpolated values f(x):", f_x)
print("Gradient df/dy:", df_dy)

# Plotting
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y, val, 'o', label='Baseline (nodes)')
plt.plot(x, f_x, '-', label='Interpolated')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('RBF Interpolation')
plt.legend()
plt.grid(True)
plt.show()