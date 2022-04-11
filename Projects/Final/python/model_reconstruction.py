from msilib.schema import Error
import os
from random import sample
import sys
from typing import Type

import matplotlib.pyplot as plt
import numpy as np

import warnings
import cv2
import calibrate_camera
import common, estimate_E, plotting
from matlab_inspired_interface import match_features, show_matched_features

class ExtractFeaturesSIFT:
  def __init__(
        self,
        n_features          : int,
        contrast_threshold  : float,
        edge_threshold      : float 
      ) -> None:
    # Documentation: https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
    self.sift = cv2.SIFT_create(
      nfeatures=0, 
      contrastThreshold=0.05,
      edgeThreshold=25
    )

  def extract_features(self, image : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the keypoints and the descriptors from an image
    """
    keypoints, descriptors = self.sift.detectAndCompute(image, None)
    keypoints = np.array([kp.pt for kp in keypoints])
    return keypoints, descriptors


def __match_raw(
      I1 : np.ndarray, 
      I2 : np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  sift = ExtractFeaturesSIFT(n_features=0, contrast_threshold=0.05, edge_threshold=25)
  keypoints_1, descriptors_1 = sift.extract_features(image=I1)
  keypoints_2, descriptors_2 = sift.extract_features(image=I2)

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

  return  keypoints_1[index_pairs[:,0]], keypoints_2[index_pairs[:,1]], \
          descriptors_1[index_pairs[:,0]], descriptors_2[index_pairs[:,1]]

def __ransac(
      xy1 : np.ndarray,
      xy2 : np.ndarray,
      uv1 : np.ndarray,
      uv2 : np.ndarray,
      K   : np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
  """
  Assuming that the camera is calibrated in the previous task.

  Using the methods from assignment 5 to calculate the things 
  needed for ransac. 

  Returns the essential matrix and the full inlier set
  """

  num_trials = estimate_E.calculate_num_ransac_trials(
    sample_size=xy1.shape[0],
    confidence=0.99,
    inlier_fraction=0.5 # Assuming that the same inlier fraction as assignment 5 can be used
  )
  # Hardcoded, as the function gave really low values 
  num_trials = 2000

  E, inlier_set = estimate_E.ransac(
    xy1=xy1, 
    xy2=xy2,
    uv1=uv1,
    uv2=uv2, 
    K=K, 
    distance_threshold=4, 
    num_trials=num_trials 
  )

  return E, inlier_set


def two_view_reconstruction(
      I1_path_str                 : str   = '../data/hw5_ext/IMG_8210.jpg',
      I2_path_str                 : str   = '../data/hw5_ext/IMG_8211.jpg',
      calibration_folder_path_str : str   = '../data/hw5_ext/calibration/'
    ) -> None:
  # Currently just using the initial images
  I1 = cv2.imread(os.path.join(sys.path[0], I1_path_str), cv2.IMREAD_GRAYSCALE)
  I2 = cv2.imread(os.path.join(sys.path[0], I2_path_str), cv2.IMREAD_GRAYSCALE)

  K = np.loadtxt(os.path.join(sys.path[0], "".join([calibration_folder_path_str, 'K.txt'])))
  K_inv = np.linalg.inv(K) 

  # Matching features
  # best_index_pairs, best_matches, best_keypoints_1, best_keypoints_2 = match_raw(I1, I2)
  keypoints_1, keypoints_2, descriptors1, descriptors2 = __match_raw(I1, I2)

  # Running RANSAC to optimize the features extracted
  # Is it the best keypoints that are going to be sent to ransac?
  # That does not make sense! It should instead be the other points that are 
  # not already selected to be the best. Otherwise, ransac may have too few
  # points to determine the inliers. The problem with having too many points 
  # to evaluate, is that the problem becomes far more ocmputational complex 
  # uv1 = np.vstack([best_keypoints_1.T, np.ones(best_keypoints_1.shape[0])])
  # uv2 = np.vstack([best_keypoints_2.T, np.ones(best_keypoints_2.shape[0])])

  uv1 = np.vstack([keypoints_1.T, np.ones(keypoints_1.shape[0])])
  uv2 = np.vstack([keypoints_2.T, np.ones(keypoints_2.shape[0])])

  xy1 = common.project(arr=uv1, K_inv=K_inv)
  xy2 = common.project(arr=uv2, K_inv=K_inv)

  E, inlier_set = __ransac(xy1=xy1, xy2=xy2, uv1=uv1, uv2=uv2, K=K)
  F = common.F_from_E(E, K)

  # This appears to be too small...
  # The inlier set will naturally be limited to the dataset being sent into the ransac-module.
  # However, this is perhaps far too small for our use case? By just sending in the best values 
  # it may be difficult to get a proper estimate on the actual inliers? However, by sending in
  # more values, it becomes too heavy for ransac to calculate all... 
  # print("Found {} inliers".format(np.sum(inlier_set == 1))) 

  # Extracting the inliers and caluclating the pose
  # Why are these so strange? The indexing caused by the inlier set causes this!
  xy1 = xy1[:,inlier_set]
  xy2 = xy2[:,inlier_set]

  P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
  pose_matrices = common.decompose_E(E)
  P2 = common.find_optimal_pose(
    pose_matrices=pose_matrices, 
    pose_world=P1, 
    xy1=xy1, 
    xy2=xy2
  )

  R2 = common.closest_rotation_matrix(P2[:3,:3])
  t2 = P2[:3,-1]

  # Get the features in world frame (camera 1)
  X = common.triangulate_many(xy1=xy1, xy2=xy2, P1=P1, P2=P2)

  # "You will need to save the 3D point coordinates and a feature descriptor associated
  # with each 3D point."
  model_keypoints = keypoints_1[inlier_set,:]
  X3D = X[:3,:]
  descriptors = descriptors1[inlier_set,:]

  # Assume that camera view 1 is the world frame which other queries 
  # will be matched against later
  matched_features = np.block([model_keypoints, X3D.T, descriptors])

  # Save the detected features in world frame
  np.savetxt(os.path.join(sys.path[0], '../data/results/task_2_1/X.txt'), X)
  np.savetxt(os.path.join(sys.path[0], '../data/results/task_2_1/R2.txt'), R2)
  np.savetxt(os.path.join(sys.path[0], '../data/results/task_2_1/t2.txt'), t2)
  np.savetxt(os.path.join(sys.path[0], '../data/results/task_2_1/K.txt'), K)
  np.savetxt(os.path.join(sys.path[0], '../data/results/task_2_1/matched_features.txt'), matched_features)

  plotting.draw_correspondences(I1=I1, I2=I2, uv1=uv1, uv2=uv2, F=F, sample_size=8)
  plt.show()


if __name__ == '__main__':
  I1_path_str = '../data/hw5_ext/undistorted/IMG_8210.jpg'
  I2_path_str = '../data/hw5_ext/undistorted/IMG_8211.jpg'
  calibration_folder_path_str = '../data/hw5_ext/undistorted/'
  two_view_reconstruction(
    I1_path_str=I1_path_str, 
    I2_path_str=I2_path_str, 
    calibration_folder_path_str=calibration_folder_path_str
  )


