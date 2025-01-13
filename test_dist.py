import numpy as np

y = np.linspace(0, 1, 6)
x = np.linspace(0.1, 0.9, 2)

ydist = np.zeros((y.shape[0], y.shape[0]))
for i in range(y.shape[0]):
    for j in range(y.shape[0]):
        ydist[i, j] = np.sqrt((y[i] - y[j])**2)

xdist = np.zeros((x.shape[0], y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        xdist[i, j] = np.linalg.norm(x[i] - y[j])

# Derivatives
ydist_grad = np.zeros((y.shape[0], y.shape[0]))
for i in range(y.shape[0]):
    for j in range(y.shape[0]):
        if y[i] == y[j]:
            ydist_grad[i, j] = 0
        else:
            ydist_grad[i, j] = (y[i] - y[j]) / np.sqrt((y[i] - y[j])**2)

xdist_grad = np.zeros((x.shape[0], y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        xdist_grad[i, j] = -(x[i] - y[j]) / np.sqrt((x[i] - y[j])**2)

# Finite difference
h = 1e-6
ydist_grad_fd = np.zeros((y.shape[0], y.shape[0]))
for i in range(y.shape[0]):
    for j in range(y.shape[0]):
        ydist_grad_fd[i, j] = (np.sqrt(((y[i]+h) - (y[j]))**2) - np.sqrt(((y[i]-h) - (y[j]))**2)) / (2 * h)


xdist_grad_fd = np.zeros((x.shape[0], y.shape[0]))
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        xdist_grad_fd[i, j] = ((np.linalg.norm(x[i] - (y[j] + h)) - np.linalg.norm(x[i] - (y[j] - h))) / (2 * h))
print(y)
print('ygrad', ydist_grad)
print(x)
print('xgrad', xdist_grad)
quit()
print(ydist_grad_fd)
print(xdist_grad)
print(xdist_grad_fd)
print('diff y', ydist_grad - ydist_grad_fd)
print('diff x', xdist_grad - xdist_grad_fd)
print(np.allclose(ydist_grad, ydist_grad_fd))
print(np.allclose(xdist_grad, xdist_grad_fd))