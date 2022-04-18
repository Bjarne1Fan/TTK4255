import common
import math

import numpy as np

def eight_point_algorithm(
      xy1 : np.ndarray, 
      xy2 : np.ndarray
    ) -> np.ndarray:
  """
  Using parts of the solution for assignment 5, namely the
  calculation of A-matrix
  """
  
  n = xy1.shape[1]
  assert n >= 8, "Not enough correspondances to estimate E. \
    Use 5-point or another algorithm instead!"

  # Calculating A
  A = np.empty((n, 9))
  for i in range(n):
    x1,y1 = xy1[:2,i]
    x2,y2 = xy2[:2,i]
    A[i,:] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

  # Using SVD to decompose into E
  _, _, V_T = np.linalg.svd(A) 
  return V_T[-1].reshape((3, 3))


def calculate_num_ransac_trials(
      sample_size     : int, 
      confidence      : float, 
      inlier_fraction : float
    ) -> int:
  """
  Using the solution for assignment 5
  """
  return int(np.log(1 - confidence)/np.log(1 - inlier_fraction**sample_size))


def ransac(
      xy1                 : np.ndarray, 
      xy2                 : np.ndarray, 
      uv1                 : np.ndarray,
      uv2                 : np.ndarray,
      K                   : np.ndarray, 
      distance_threshold  : float, 
      num_trials          : int
    ) -> np.ndarray:
  """
  Using the solution for assignment 5
  """
  print(
    "Running RANSAC with inlier threshold {} pixels and {} trials...".format(
      distance_threshold, 
      num_trials
    )
  )
  best_num_inliers = -1
  for i in range(num_trials):
    sample = np.random.choice(xy1.shape[1], size=8, replace=False)
    E_i = eight_point_algorithm(xy1[:,sample], xy2[:,sample])
    d_i = common.epipolar_distance(common.F_from_E(E_i, K), uv1, uv2)
    inliers_i = np.absolute(d_i) < distance_threshold
    num_inliers_i = np.sum(inliers_i)
    if num_inliers_i > best_num_inliers:
      best_num_inliers = num_inliers_i
      E = E_i
      inliers = inliers_i
  print('Done!')
  print('Found solution with %d/%d inliers' % (np.sum(inliers), xy1.shape[1]))
  return E, inliers

