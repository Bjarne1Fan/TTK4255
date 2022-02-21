import os 
import sys

import numpy as np
import matplotlib.pyplot as plt
import common as com

# Tip: The solution from HW4 is inside common.py

K = np.loadtxt(os.path.join(sys.path[0], '../data/data/K.txt'))
u = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_corners_image.txt'))
X = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_corners_metric.txt'))
I = plt.imread(os.path.join(sys.path[0], '../data/quanser/video0000.jpg')) # Only used for plotting

# Example: Compute predicted image locations and reprojection errors
T_hat = com.translate(-0.3, 0.1, 1.0) @ com.rotate_x(1.8)
u_hat = com.project(K, T_hat @ X)
errors = np.linalg.norm(u - u_hat, axis=0)

# Print the reprojection errors requested in Task 2.1 and 2.2.
print('Reprojection error: ')
print('all:', ' '.join(['%.03f' % e for e in errors]))
print('mean: %.03f px' % np.mean(errors))
print('median: %.03f px' % np.median(errors))

plt.imshow(I)
plt.scatter(u[0,:], u[1,:], marker='o', facecolors='white', edgecolors='black', label='Detected')
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
