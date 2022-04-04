import os
from secrets import randbelow 
import sys

import matplotlib.pyplot as plt
import numpy as np

import cv2 as cv
# from assignment_5 import decompose_E, epipolar_distance, estimate_E, F_from_E, plotting, project, triangulate_many
import decompose_E, epipolar_distance, estimate_E, F_from_E, plotting, project, triangulate_many
from matlab_inspired_interface import match_features, show_matched_features

def match_raw(I0, I1) -> tuple[np.ndarray, np.ndarray]:
  # NB! This script uses a very small number of features so that it runs quickly.
  # You will want to pass other options to SIFT_create. See the documentation:
  # https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
  sift = cv.SIFT_create(
    nfeatures=0, 
    contrastThreshold=0.1,
    edgeThreshold=10
  )
  keypoints_0, descriptors_0 = sift.detectAndCompute(I0, None)
  keypoints_1, descriptors_1 = sift.detectAndCompute(I1, None)
  keypoints_0 = np.array([kp.pt for kp in keypoints_0])
  keypoints_1 = np.array([kp.pt for kp in keypoints_1])

  # NB! You will want to experiment with different options for the ratio test and
  # "unique" (cross-check).
  index_pairs, match_metric = match_features(
    descriptors_0, 
    descriptors_1, 
    max_ratio=0.5, 
    unique=True
  )
  print(index_pairs[:50])
  print('Found %d matches' % index_pairs.shape[0])

  # Plot the 50 best matches in two ways
  best_matches = np.argsort(match_metric)[:50]
  best_index_pairs = index_pairs[best_matches]
  best_keypoints_0 = keypoints_0[best_index_pairs[:,0]]
  best_keypoints_1 = keypoints_1[best_index_pairs[:,1]]
  # plt.figure()
  # show_matched_features(I1, I2, best_keypoints_0, best_keypoints_1, method='falsecolor')
  # plt.figure()
  # show_matched_features(I1, I2, best_keypoints_0, best_keypoints_1, method='montage')

  return best_index_pairs, best_matches, np.hstack([best_keypoints_0, best_keypoints_1])


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
    distance_threshold=10, 
    num_trials=500
  )

  return E, inlier_set


if __name__ == '__main__':
  # Choose these later - currently just using the default images
  I1 = cv.imread(os.path.join(sys.path[0], '../data/hw5_ext/IMG_8210.jpg'), cv.IMREAD_GRAYSCALE)
  I2 = cv.imread(os.path.join(sys.path[0], '../data/hw5_ext/IMG_8211.jpg'), cv.IMREAD_GRAYSCALE)

  _, _, best_keypoints = match_raw(I1, I2)
  
  uv1 = np.vstack([best_keypoints[:,:2].T, np.ones(best_keypoints.shape[0])])
  uv2 = np.vstack([best_keypoints[:,2:4].T, np.ones(best_keypoints.shape[0])])

  # Assuming that the same camera-parameters from task 1 are still valid
  K = np.loadtxt(os.path.join(sys.path[0], '../data/hw5_ext/calibration/K.txt'))

  E, inlier_set = ransac(uv1=uv1, uv2=uv2, K=K)
  F = F_from_E.F_from_E(E, K)

  # Extracting the 8 best inliers and calculating the pose
  sample_size = 8
  if len(inlier_set) < sample_size:
    sample_size = len(inlier_set) - 1
    best_inliers = inlier_set[:len(inlier_set) - 1]
  else:
    best_inliers = inlier_set[:sample_size]

  plotting.draw_correspondences(I1=I1, I2=I2, uv1=uv1, uv2=uv2, F=F, sample_size=sample_size)

  plt.show()




# The goal is to create a 3D-reconstruction using two images
# Things that need to be developed:
# -feature correspondence
# -RANSAC
# -epipolar lines
# -inlier correspondance (through RANSAC?)

# Load these values correctly

