import os 
import sys

import numpy as np
import matplotlib.pyplot as plt
import common as com

K     = np.loadtxt(os.path.join(sys.path[0], '../data/data/K.txt'))
uv    = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_corners_image.txt'))
XY01T = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_corners_metric.txt'))
I     = plt.imread(os.path.join(sys.path[0], '../data/quanser/video0000.jpg')) # Only used for plotting

is_task_a = False

# Extracting XY and XY1 (Thanks for the stupid format on XY)
XY = (XY01T.T[:,:2]).T
XY1 = np.vstack([XY, np.ones((1, XY.shape[1]))])

# Calculating H and u_hat for 2.1
uv1 = np.vstack((uv, np.ones(uv.shape[1])))
xy = com.project(np.linalg.inv(K), uv1)

H = com.estimate_H(xy, XY)
u_hat_a = com.project(K, H @ XY1)

T_0, T_1 = com.decompose_H(H)

if (T_0 @ XY01T.T)[2, 3] > 0:
  T_hat = T_0 
else:
  T_hat = T_1

print(T_hat @ XY01T.T)
u_hat_b = com.project(K, (T_hat @ XY01T.T).T)

if is_task_a:
  u_hat = u_hat_a
else:
  u_hat = u_hat_b

# u_hat = com.project(K, T_hat @ XY01)
errors = np.linalg.norm(uv - u_hat, axis=0)

# Print the reprojection errors requested in Task 2.1 and 2.2.
print('Reprojection error: ')
print('all:', ' '.join(['%.03f' % e for e in errors]))
print('mean: %.03f px' % np.mean(errors))
print('median: %.03f px' % np.median(errors))

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
