import os 
import sys

import numpy as np
import matplotlib.pyplot as plt
import common as com

from scipy.optimize import least_squares
from numpy import ndarray

K     = np.loadtxt(os.path.join(sys.path[0], '../data/data/K.txt'))
uv    = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_corners_image.txt'))
XY01T = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_corners_metric.txt'))
I     = plt.imread(os.path.join(sys.path[0], '../data/quanser/video0000.jpg')) # Only used for plotting

# Switch between task a) and b) for 2.1
is_task_a = False

# Extracting values. Copy to prevent references destroying everything
XY01 = XY01T.T
XY = XY01[:,:2].copy()
XY1 = np.hstack([XY.copy(), np.ones((XY.shape[0], 1))])

# Calculating the homogenized pixel-coordinates
u_tilde = np.vstack((uv, np.ones(uv.shape[1])))

# Calculating the calibrated camera-coordinates
xy = com.project(np.linalg.inv(K), u_tilde)

# Estimate H
H = com.estimate_H(xy, XY.T)

# Decompusing H into T = [R|t]
T_0, T_1 = com.decompose_H(H)

# Extracting the one with positive z-value
if (T_0 @ XY01)[2, 3] > 0:
  T_hat = T_0 
else:
  T_hat = T_1

# Calculate the predicted image locations
u_hat_a = com.project(K, H @ XY1.T)
u_hat_b = com.project(K, T_hat @ XY01.T)

# Switch between a) and b) for task 2.1
if is_task_a:
  u_hat = u_hat_a
else:
  u_hat = u_hat_b

# Calculating errors for naive implementation 
errors = uv - u_hat
print(errors)

# Function for iteratively optimizing the errors between the current values and the optimal values
def u_hat_residual(
      x     : ndarray, 
      R0    : ndarray, 
      uv    : ndarray,
      XY01T : ndarray,
      K     : ndarray
    )->ndarray:
  # Extracting values
  phi, theta, psi = x[0], x[1], x[2]
  tx, ty, tz = x[3], x[4], x[5]
  
  # Creating the transformation-matrix
  R = com.rotate_x(phi) @ com.rotate_y(theta) @ com.rotate_z(psi) @ R0
  T = com.translate(tx, ty, tz)

  T[:3,:3] = R[:3,:3]

  # Calculating the estimated coordinates
  u_hat = com.project(K, T_hat @ XY01T) 

  # Calculating the residual function
  r_0 = u_hat[0] - uv[0]
  r_1 = u_hat[1] - uv[1]
  r = np.hstack([r_0, r_1])
  return r

# Initial conditions for task 2.2
x = np.array([0,0,0,0,0,0])
R0 = np.eye(4)

# Creating the residual function
resfun = lambda x : u_hat_residual(x, R0, uv, XY01T, K)

# Minimizing the residual function
x = least_squares(resfun, x0=x, method='lm').x

# Calculating the errors from LM-optimization
errors = u_hat_residual(x, R0, uv, XY01T, K).reshape((-1,4))
print(errors)

normed_errors = np.linalg.norm(errors, axis=0)

# Print the reprojection errors requested in Task 2.1 and 2.2.
print('Reprojection errors: ')
print('all:', ' '.join(['%.03f' % e for e in normed_errors]))
print('mean: %.03f px' % np.mean(normed_errors))
print('median: %.03f px' % np.median(normed_errors))

plt.imshow(I)
plt.scatter(uv[0,:], uv[1,:], marker='o', facecolors='white', edgecolors='black', label='Detected')
plt.scatter(u_hat[0,:], u_hat[1,:], marker='.', color='red', label='Predicted')
plt.legend()

# Tip: Draw lines connecting the points for easier understanding
plt.plot(u_hat[0,:], u_hat[1,:], linestyle='--', color='white')

# Tip: To draw a transformation's axes (only requested in Task 2.3)
com.draw_frame(K, T_hat, scale=0.05, labels=True)

# Tip: To zoom in on the platform:
plt.xlim([200, 500])
plt.ylim([600, 350])

# Tip: To see the entire image:
# plt.xlim([0, I.shape[1]])
# plt.ylim([I.shape[0], 0])

# Tip: To save the figure:
# plt.savefig('out_part2.png')

plt.show()
