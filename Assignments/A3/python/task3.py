import os
import sys 

import numpy as np
import matplotlib.pyplot as plt
import common as com

# Tip: Use np.loadtxt to load data into an array
K = np.loadtxt(os.path.join(sys.path[0], '../data/task2K.txt'))
X = np.loadtxt(os.path.join(sys.path[0], '../data/task3points.txt'))

# Task 3.2
# Apply transformation before projection
# Based on the figure, it looks like it is rotated around y with 45 degrees and
# around x with 15
deg_x = 15
deg_y = 45
deg_z = 0

# Rx = com.rotate_x(np.deg2rad(deg_x))
# Ry = com.rotate_y(np.deg2rad(deg_y))
# Rz = com.rotate_z(np.deg2rad(deg_z))

# R = Rz @ Ry @ Rx
# t = np.array([[0], [0], [6]])
# K = np.array(
#   [
#     [R[0,0], R[0,1], R[0,2], t[0,0]],
#     [R[1,0], R[1,1], R[1,2], t[1,0]],
#     [R[2,0], R[2,1], R[2,2], t[2,0]],
#     [0, 0, 0, 1]
#   ]
# )

T = com.translate(0, 0, 6) @ com.rotate_x(deg_x) @ com.rotate_y(deg_y)
X_c = T @ X

u, v = com.project(K, X_c)
width, height = 600, 400

#
# Figure for Task 3.2: Show pinhole projection of 3D points
#
plt.figure(figsize=(4,3))
plt.scatter(u, v, c='black', marker='.', s=20)

# The following commands are useful when the figure is meant to simulate
# a camera image. Note: these must be called after all draw commands!!!!

plt.axis('image')     # This option ensures that pixels are square in the figure (preserves aspect ratio)
                      # This must be called BEFORE setting xlim and ylim!
plt.xlim([0, width])
plt.ylim([height, 0]) # The reversed order flips the figure such that the y-axis points down
plt.show()
