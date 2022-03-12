import os 
import sys

import matplotlib.pyplot as plt
import numpy as np
import figures as fig
import estimate_E as est_E
import decompose_E as dec_E
import triangulate_many as triangulate
import epipolar_distance as epi_dist
import F_from_E 

K = np.loadtxt(os.path.join(sys.path[0], '../data/K.txt'))
I1 = plt.imread(os.path.join(sys.path[0], '../data/image1.jpg')) / 255.0
I2 = plt.imread(os.path.join(sys.path[0], '../data/image2.jpg')) / 255.0
matches = np.loadtxt(os.path.join(sys.path[0], '../data/matches.txt'))
# matches = np.loadtxt(os.path.join(sys.path[0], '../data/task4matches.txt')) # Part 4

uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])

# Task 2: Estimate E
# E = ...

# Task 3: Triangulate 3D points
# X = ...

#
# Uncomment in Task 2
#
# np.random.seed(123) # Leave as commented out to get a random selection each time
# draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K), sample_size=8)

#
# Uncomment in Task 3
#
# draw_point_cloud(X, I1, uv1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])

plt.show()
