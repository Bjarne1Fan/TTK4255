import os
from secrets import randbelow 
import sys

import matplotlib.pyplot as plt
import numpy as np

import warnings
import cv2
import calibrate_camera
# from assignment_5 import decompose_E, epipolar_distance, estimate_E, F_from_E, plotting, project, triangulate_many
import decompose_E, epipolar_distance, estimate_E, F_from_E, plotting, project, triangulate_many
from matlab_inspired_interface import match_features, show_matched_features

def match_raw(I1, I2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  # Documentation: https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
  sift = cv2.SIFT_create(
    nfeatures=5000, 
    contrastThreshold=0.005,
    edgeThreshold=50
  )
  keypoints_1, descriptors_1 = sift.detectAndCompute(I1, None)
  keypoints_2, descriptors_2 = sift.detectAndCompute(I2, None)
  keypoints_1 = np.array([kp.pt for kp in keypoints_1])
  keypoints_2 = np.array([kp.pt for kp in keypoints_2])

  index_pairs, match_metric = match_features(
    descriptors_1, 
    descriptors_2, 
    max_ratio=1.0, 
    unique=True
  )
  # print(index_pairs[:50])
  print('Found %d matches' % index_pairs.shape[0])

  # Plot the 50 best matches in two ways
  best_matches = np.argsort(match_metric)[:50]
  best_index_pairs = index_pairs[best_matches]
  best_keypoints_1 = keypoints_1[best_index_pairs[:,0]]
  best_keypoints_2 = keypoints_2[best_index_pairs[:,1]]
  plt.figure()
  show_matched_features(I1, I2, best_keypoints_1, best_keypoints_2, method='falsecolor')
  plt.figure()
  show_matched_features(I1, I2, best_keypoints_1, best_keypoints_2, method='montage')

  return best_index_pairs, best_matches, keypoints_1, best_keypoints_2
  # return index_pairs, match_metric, keypoints_1, keypoints_2


def ransac(
      uv1     : np.ndarray,
      uv2     : np.ndarray,
      K       : np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
  """
  Assuming that the camera is calibrated in the previous task.

  Using the methods from assignment 5 to calculate the things 
  needed for ransac. 

  Returns the essential matrix and the full inlier set
  """
  K_inv = np.linalg.inv(K)

  xy1 = project.project(arr=uv1, K_inv=K_inv)
  xy2 = project.project(arr=uv2, K_inv=K_inv)

  E, inlier_set = estimate_E.ransac(
    xy1=xy1, 
    xy2=xy2,
    uv1=uv1,
    uv2=uv2, 
    K=K, 
    distance_threshold=4, 
    num_trials=4000
  )

  return E, inlier_set

def determine_P2(
      P_matrices  : list,
      P_world     : np.ndarray,
      xy1         : np.ndarray,
      xy2         : np.ndarray
    ) -> np.ndarray:

  assert isinstance(P_matrices, list), "Input must be a list"
  assert isinstance(P_matrices[0], np.ndarray), "Input must be a list of ndarrays"
  
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

def closest_rotation_matrix(Q: np.ndarray) -> np.ndarray:
  U, _, VT = np.linalg.svd(Q)
  R = U @ VT
  return R

if __name__ == '__main__':
  # Choose these later - currently just using the default images
  I1 = cv2.imread(os.path.join(sys.path[0], '../data/hw5_ext/IMG_8210.jpg'), cv2.IMREAD_GRAYSCALE)
  I2 = cv2.imread(os.path.join(sys.path[0], '../data/hw5_ext/IMG_8211.jpg'), cv2.IMREAD_GRAYSCALE)

  # Assuming that the same camera-parameters from task 1 are still valid
  distortion_coefficients = np.loadtxt(os.path.join(sys.path[0], '../data/hw5_ext/calibration/dc.txt'))
  K = np.loadtxt(os.path.join(sys.path[0], '../data/hw5_ext/calibration/K.txt'))
  K_inv = np.linalg.inv(K)

  # Undistorting image
  I1 = calibrate_camera.undistort_image(distorted_image=I1, K=K, distortion_coefficients=distortion_coefficients)
  I2 = calibrate_camera.undistort_image(distorted_image=I2, K=K, distortion_coefficients=distortion_coefficients)

  # Matching features
  best_index_pairs, best_matches, best_keypoints_1, best_keypoints_2 = match_raw(I1, I2)

  # Running RANSAC
  uv1 = np.vstack([best_keypoints_1.T, np.ones(best_keypoints_1.shape[0])])
  uv2 = np.vstack([best_keypoints_2.T, np.ones(best_keypoints_2.shape[0])])

  # Obs! Must use the best inliers to estimate the position and orientation
  # of the second camera - using these will cause poor estimates to have far 
  # too large impact on the performance

  xy1 = project.project(arr=uv1, K_inv=K_inv)
  xy2 = project.project(arr=uv2, K_inv=K_inv)

  E, inlier_set = ransac(uv1=uv1, uv2=uv2, K=K)
  F = F_from_E.F_from_E(E, K)

  print(np.sum(inlier_set == 1))

  # Extracting the inliers and caluclating the pose
  xy1 = xy1[inlier_set]
  xy2 = xy2[inlier_set]

  P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
  P_matrices = decompose_E.decompose_E(E)
  P2 = determine_P2(P_matrices=P_matrices, P_world=P1, xy1=xy1, xy2=xy2)

  R2 = closest_rotation_matrix(P2[:3,:3])
  t2 = P2[:3,-1]

  print(R2)
  print(np.linalg.det(R2))
  print(t2)

  # Get the features in world frame (camera 1)
  X1 = triangulate_many.triangulate_many(xy1=xy1, xy2=xy2, P1=P1, P2=P2)

  # Save the detected features in world frame

  plotting.draw_correspondences(I1=I1, I2=I2, uv1=uv1, uv2=uv2, F=F, sample_size=8)
  plt.show()


