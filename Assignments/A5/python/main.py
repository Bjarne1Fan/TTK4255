import os 
import sys

import matplotlib.pyplot as plt
import numpy as np
import estimate_E
import decompose_E
import triangulate_many
import epipolar_distance
import F_from_E 
import plotting
import project

K = np.loadtxt(os.path.join(sys.path[0], '../data/K.txt'))
I1 = plt.imread(os.path.join(sys.path[0], '../data/image1.jpg')) / 255.0
I2 = plt.imread(os.path.join(sys.path[0], '../data/image2.jpg')) / 255.0
# matches = np.loadtxt(os.path.join(sys.path[0], '../data/matches.txt'))
matches = np.loadtxt(os.path.join(sys.path[0], '../data/task4matches.txt')) # Part 4

uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])

# Task 2: Estimate E
K_inv = np.linalg.inv(K)

xy1 = project.project(arr=uv1, K_inv=K_inv)
xy2 = project.project(arr=uv2, K_inv=K_inv)

E = estimate_E.eight_point_algorithm(xy1=xy1, xy2=xy2)
F = F_from_E.F_from_E(E=E, K=K)

# Task 3: Triangulate 3D points

# Determine which is in front of both cameras
# Creating projection matrices
P_matrices = decompose_E.decompose_E(E)

# Assuming that the first camera is in world frame
P1 = np.hstack(
  [
    np.eye(3), np.zeros((3, 1))
  ]
)

def determine_P2(
      P_matrices  : list,
      P_world     : np.ndarray,
      xy1         : np.ndarray,
      xy2         : np.ndarray
    ) -> np.ndarray:

  assert isinstance(P_matrices, list)
  
  pos_z = 0
  best_idx = 0

  # Iterate over all of the possible matrices and find the matrix with most
  # measurements in front of the camera
  for (idx, P) in enumerate(P_matrices):
    X = triangulate_many.triangulate_many(
      xy1=xy1, 
      xy2=xy2, 
      P1=P_world, 
      P2=P
    )
    
    # Find maximum points with a positive z-value (in front of the camera)
    num_pos_z = np.sum(((P @ X)[2] >= 0))
    if num_pos_z >= pos_z:
      pos_z = num_pos_z
      best_idx = idx

  return P_matrices[best_idx]

# Find optimal P2-matrix
# P2 = determine_P2(
#   P_matrices=P_matrices, 
#   P_world=P1,
#   xy1=xy1, 
#   xy2=xy2
# )

# Triangulate 
# X = triangulate_many.triangulate_many(xy1=xy1, xy2=xy2, P1=P1, P2=P2)

# Task 4.1
# residuals = epipolar_distance.epipolar_distance(F=F, uv1=uv1, uv2=uv2)
# plotting.draw_residual_histograms(
#   residuals=residuals, 
#   matrix_name="initial", 
#   num_bins=2500,
#   ranges=(-1500, 1500)
# )

# Task 4.2
E = np.zeros(3)
inlier_set = 0

num_inliers = 0
while num_inliers < 2000:
  E, inlier_set = estimate_E.ransac(
    xy1=xy1, 
    xy2=xy2,
    uv1=uv1,
    uv2=uv2, 
    K=K, 
    distance_threshold=4, 
    num_trials=50
  )
  num_inliers = len(inlier_set)

print(inlier_set.shape)

# Extractring the measurements
xy1 = xy1[:,inlier_set]
xy2 = xy2[:,inlier_set]
uv1 = uv1[:,inlier_set]
uv2 = uv2[:,inlier_set]

P_matrices = decompose_E.decompose_E(E=E)
P2 = determine_P2(
  P_matrices=P_matrices, 
  P_world=P1,
  xy1=xy1, 
  xy2=xy2
)


X = triangulate_many.triangulate_many(xy1=xy1, xy2=xy2, P1=P1, P2=P2)



#
# Uncomment in Task 2
#
# np.random.seed(123) # Leave as commented out to get a random selection each time
# plotting.draw_correspondences(
#   I1=I1, 
#   I2=I2, 
#   uv1=uv1, 
#   uv2=uv2, 
#   F=F, 
#   sample_size=8
# )

#
# Uncomment in Task 3 and 4
#
plotting.draw_point_cloud(X, I1, uv1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])

plt.show()
