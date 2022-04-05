import common
import math

import numpy as np

def eight_point_algorithm(
      xy1 : np.ndarray, 
      xy2 : np.ndarray
    ) -> np.ndarray:
  
  n = xy1.shape[1]
  assert n >= 8, "Not enough correspondances to estimate E. \
    Use 5-point or another algorithm instead!"

  # Calculating A
  A = np.zeros((n, 9))
  for row in range(n):
    # Assuming xy1 to come from the left image, and xy2 to be the right
    A[row] = np.array(
      [
        xy2[0, row] * xy1[0, row], xy2[0, row] * xy1[1, row], xy2[0, row], \
        xy2[1, row] * xy1[0, row], xy2[1, row] * xy1[1, row], xy2[1, row], \
        xy1[0, row], xy1[1, row], 1
      ]
    )
    
  # Using SVD to decompose into E
  _, _, V_T = np.linalg.svd(A) 
  return V_T[-1].reshape((3, 3))


def ransac(
      xy1                 : np.ndarray, 
      xy2                 : np.ndarray, 
      uv1                 : np.ndarray,
      uv2                 : np.ndarray,
      K                   : np.ndarray, 
      distance_threshold  : float, 
      num_trials          : int
    ) -> np.ndarray:

  K_inv = np.linalg.inv(K)

  # Defining memory
  E = np.eye(3)
  inlier_set = set()

  max_num_inliers = 0

  # Trying possible values
  for i in range(num_trials):
    # Extract m correspondences randomly and estimate using eight-point-algorithm
    sample = np.random.choice(xy1.shape[1], size=8, replace=False)
    estimated_E = eight_point_algorithm(xy1[:,sample], xy2[:,sample])

    # Calculate the fundamental matrix and the residuals
    F = K_inv.T @ E @ K_inv #F_from_E.F_from_E(E=E, K=K)
    residuals = common.epipolar_distance(F=F, uv1=uv1, uv2=uv2)
    inliers = np.abs(residuals) < distance_threshold
    num_inliers = np.sum(inliers)
    #avg_residuals = 1/2.0 * (residuals[0,:] + residuals[1,:])
    #abs_avg_residuals = np.abs(avg_residuals)

    # Check number of points being inliers
    # num_inliers = np.sum(abs_avg_residuals <= distance_threshold)

    if num_inliers >= max_num_inliers:
      # print("Better sample found at iteration number: {}".format(i))
      max_num_inliers = num_inliers
      inlier_set = inliers#np.where(abs_avg_residuals <= distance_threshold)[0]
      E = estimated_E

    if i % 100 == 0 and i > 0:
      print(i)
  inlier_set = np.nonzero(inlier_set)[0]

  return E, inlier_set 


def prosac(
      xy1                 : np.ndarray, 
      xy2                 : np.ndarray, 
      uv1                 : np.ndarray,
      uv2                 : np.ndarray,
      K                   : np.ndarray, 
      distance_threshold  : float, 
      inlier_probability  : float,
      success_probability : float
    ) -> np.ndarray:
  m = 8

  S = math.log(1 - success_probability) / math.log(1 - inlier_probability**m)
  S = math.ceil(S)

  # Defining memory
  E = np.eye(3)
  inlier_set = set()

  max_num_inliers = 0

  # Trying possible values
  for i in range(S):
    # Extract m correspondences randomly and estimate using eight-point-algorithm
    sample = np.random.choice(xy1.shape[1], size=m, replace=False)
    estimated_E = eight_point_algorithm(xy1[:,sample], xy2[:,sample])

    # Calculate the fundamental matrix and the residuals
    F = common.F_from_E(E=E, K=K)
    residuals = common.epipolar_distance(F=F, uv1=uv1, uv2=uv2)
    avg_residuals = 1/2.0 * (residuals[0,:] + residuals[1,:])
    abs_avg_residuals = np.abs(avg_residuals)

    # Check number of points being inliers
    num_inliers = np.sum(abs_avg_residuals <= distance_threshold)

    if num_inliers >= max_num_inliers:
      print("Better sample found at iteration number: {}".format(i))
      max_num_inliers = num_inliers
      inlier_set = np.where(abs_avg_residuals <= distance_threshold)[0]
      E = estimated_E

    if i % 100 == 0 and i > 0:
      print(i)

  return E, inlier_set 

