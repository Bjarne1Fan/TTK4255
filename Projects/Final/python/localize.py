import os 
import sys
import cv2

import matplotlib.pyplot as plt
import numpy as np

import plotting

from model_reconstruction import ExtractFeaturesSIFT
from matlab_inspired_interface import match_features, show_matched_features

def __nonlinear_least_squares(
      R0 : np.ndarray, 
      T0 : np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
  """
  Nonlinear least squares that must be solved to get R and T better 
  refined.
  """
  R = np.eye(3)
  T = np.zeros((3, 1))
  reprojection_error = np.zeros((3, 1))
  return R, T, reprojection_error


def localize(
      model_path    : str   = '../example_localization',
      query_path    : str   = '../example_localization/query/',
      image_str  : str   = 'IMG_8210.jpg', 
      default       : bool  = True
    ) -> None:
  """
  From the discussion on overleaf:

  What is meant by localization of a random image? Haven't we done this in task 2.1?
  
  What is assumed to be meant by localization:
  Have an image with known features. Receive a new image with unknown features. 
  Search through the new image and detect if there are any similar features. 
  If there are enough of them, try to calculate the pose difference between the first and 
  the second image based on this. 
  
  Psudocode:
   Load random image in dataset
   Extract features on the random image, and store the features to a file
   Use solvePnPRansac to get initial guess on R and T
   Refine R and T with non linear least square
   Set R and T together to form m2q
   Plot point cloud with m2q

  Based on the pseudocode, it would be fine if the code received a random image 
  as input. This means that other functions may be responsible for choosing the 
  optimal data input. 

  It is assumed that the image/query is undistorted before use
  """

  assert isinstance(image_str, str), "Image id must be a string"

  model = os.path.join(*[sys.path[0], model_path])
  query = os.path.join(*[sys.path[0], query_path])

  # Load features from the world frame
  # 3D points [4 x num_points].
  X = np.loadtxt(f'{model}/X.txt')

  if not default:
    K = np.loadtxt(f'{model}/K.txt')
    
    # matched_features = [features | X3D]
    matched_features = np.loadtxt(f'{model}/matched_features.txt')

    model_keypoints = matched_features[:, :2]
    X3D = matched_features[:, 2:]
    # X3D1 = np.block([X3D, np.ones((X3D.shape[0], 1))])

    query_image = cv2.imread((query + image_str), cv2.IMREAD_GRAYSCALE)
    sift = ExtractFeaturesSIFT(n_features=0, contrast_threshold=0.05, edge_threshold=25)

    # Extract the same features in the image plane with the same method 
    # as previously 
    query_keypoints, query_descriptors = sift.extract_features(image=query_image)

    index_pairs, match_metric = match_features(
      features1=model_keypoints, 
      features2=query_keypoints, 
      max_ratio=1.0, 
      unique=True
    )
    model_keypoints = model_keypoints[index_pairs[:,0]]
    query_keypoints = query_keypoints[index_pairs[:,1]]

    model_uv1 = np.vstack([model_keypoints.T, np.ones(model_keypoints.shape[0])])
    query_uv1 = np.vstack([query_keypoints.T, np.ones(query_keypoints.shape[0])])

    # Use solvePnPRansac to get initial guess on R and T
    # It is assumed that the image is undistorted before use
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
      objectPoints=X3D,
      imagePoints=query_uv1,
      cameraMatrix=K
    )
    R = rvecs.reshape((3, 3))
    T = tvecs.reshape((3, 1))

    np.savetxt(f'{query}/sfm/inliers.txt', inliers)

    # Use a nonlinear least squares to refine R and T
    R, T, reprojection_error = __nonlinear_least_squares(R0=R, T0=T)
    np.savetxt(f'{query}/sfm/reprojection_error.txt', reprojection_error)

    # Develop model-to-query transformation by [[R, T], [0, 0, 0, 1]]
    T_m2q = np.block([[R, T], [np.zeros((1, 3)), 1]]) # TODO: Check if this must be inverted

  else:
    # Model-to-query transformation.
    # If you estimated the query-to-model transformation,
    # then you need to take the inverse.
    T_m2q = np.loadtxt(f'{query}_T_m2q.txt')

    # If you have colors for your point cloud model...
    colors = np.loadtxt(f'{model}/c.txt') # RGB colors [num_points x 3].
    # ...otherwise...
    # colors = np.zeros((X.shape[1], 3))

  # Plot point-cloud using the model-to-query

  # These control the visible volume in the 3D point cloud plot.
  # You may need to adjust these if your model does not show up.
  xlim = [-10,+10]
  ylim = [-10,+10]
  zlim = [0,+20]

  frame_size = 1
  marker_size = 5

  plt.figure('3D point cloud', figsize=(6,6))
  plotting.draw_point_cloud(
    X, T_m2q, xlim, ylim, zlim, 
    colors=colors, marker_size=marker_size, frame_size=frame_size
  )
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  model_path = os.path.join(sys.path[0], "../data/results/task_2_1")
  query_path = os.path.join(sys.path[0], "../data/hw5_ext/undistorted/")
  image_str = "IMG_8220.jpg"
  localize(
    model_path=model_path,
    query_path=query_path,
    image_str=image_str,
    default=False
  )
