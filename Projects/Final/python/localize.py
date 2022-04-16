from msilib.schema import Error
import os 
import sys
import warnings
import cv2
import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import least_squares

import plotting
import common
from model_reconstruction import ExtractFeaturesSIFT
from matlab_inspired_interface import match_features

class OptimizeQueryPose:
  def __init__(
        self,
        K           : np.ndarray,
        query_uv    : np.ndarray,
        X3D         : np.ndarray,
        use_weights : bool        = False,
        sigma_u_std : float       = 50.0,
        sigma_v_std : float       = 0.1    
      ) -> None:
    """
    Class for optimizing the estimate of query pose using nonlinear
    optimization. 

    Input:
      K           : Camera coefficient matrix 
      query_uv    : Detected features in the query camera points
      X3D         : Detected features in the world frame 
      use_weights : Bool determining whether weighted residuals to be used
      sigma_u_std : Standard deviation used for horizontal pixles
      sigma_v_std : Standard deviation used for vertical pixles

    """
    self.__K_inv = np.linalg.inv(K)
    self.__query_uv = query_uv
    self.__X3D = X3D 
    
    self.__N = np.max([X3D.shape[0], X3D.shape[1]])
    
    self.__use_weights = use_weights
    if use_weights:
      __us = sigma_u_std * np.ones((1, self.__N), dtype=float)
      __vs = sigma_v_std * np.ones((1, self.__N), dtype=float)

      __sigma_r = np.diag(np.hstack([__us, __vs])[0])
      __L = np.linalg.cholesky(__sigma_r)
      self.__L_inv = np.linalg.inv(__L)


  def __residual_function(
        self,
        x : np.ndarray
      ) -> np.ndarray:
    """
    Residual function for minimzing the reprojection errors. To be used for 
    optimizing the estimated pose.

    Input:
      x : 1x6 Current state estimate on the form [rvecs | tvecs]
    
    Output:
      residuals : 1x(2n) 
    """
    rvecs = x[:3].reshape((3, 1))
    R, _ = cv2.Rodrigues(rvecs)
    t = x[3:].reshape((3, 1))

    X = R @ self.__X3D.T + t
    uv_hat = common.project(arr=X, K_inv=self.__K_inv).T

    residuals = self.__query_uv - uv_hat
    residuals = np.hstack([residuals[:,0].T, residuals[:,1].T]) # Horizontal and then the vertical errors

    if self.__use_weights:
      residuals = self.__L_inv @ residuals

    return residuals

  def __jacobian(self) -> np.ndarray:
    """
    Returns the jacobian corresponding to the optimization problem. 

    Has no impact if the LM-method is used. Currently returns None
    """
    return None # Until further is known about the sparsity of the model

  def nonlinear_least_squares(
        self,
        rvecs : np.ndarray, 
        tvecs : np.ndarray
      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Nonlinear least squares using the LM-method used to refine the estimates for 
    the rotation matrix and the translation matrix.

    Input:
      rvecs : Compact form of rotation vector, estimated from solvePnPRansac
      tvecs : Translation vector estimated from solvePnPRansac

    Output:
      x                   : 6x1   Optimized state
      R                   : 3x3   pose rotation matrix
      t                   : 3x1   pose translation vector
      reprojection_error  : float 
      cov_x               : 6x6   covariance matrix
      std_x               : 6x1   standard deviation vector
    """
    R, _ = cv2.Rodrigues(rvecs) 
    t = tvecs.reshape((3, 1))  

    x0 = np.hstack([rvecs.reshape((1, -1)), t.reshape((1, -1))]).flatten()

    optimization_results = least_squares(
      fun=self.__residual_function,
      x0=x0,
      jac_sparsity=self.__jacobian(),
      method='lm'
    )
    success = optimization_results.success

    if success:
      # Optimization converged
      print("Optimization converged after {} invocations".format(optimization_results.nfev))
      x = (optimization_results.x).reshape((6,1))

      rvecs = x[:3]
      R, _ = cv2.Rodrigues(rvecs)
      t = x[3:].reshape((3, 1))
      reprojection_error = optimization_results.cost
      jacobian = optimization_results.jac
      cov_x, std_x = self.__uncertainty(jacobian=jacobian)

    else:
      warnings.warn("Optimization did not converge! Reason: {}. Returning initial values!".format(optimization_results.message))
      x = np.infty * np.ones((6, 1))
      reprojection_error = np.infty
      cov_x, std_x = (np.infty * np.ones((6, 6)), np.infty * np.ones((6, 1)))

    return x, R, t, reprojection_error, cov_x, std_x

  def __uncertainty(
        self,
        jacobian : np.ndarray
      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses the estimated jacobian from the optimization to calculate the 
    uncertainty regarding the estimates. 

    Input:
      jacobian : (2n)x6

    Output:
      cov_p : 6x6 Covariance matrix
      std_p : 1x6 Standard deviation vector

    Returns the full covariance-matrix, as well as the standard deviations 
    """
    assert isinstance(jacobian, np.ndarray), "Jacobian must be given as an ndarray"
    assert len(jacobian.shape) == 2, "Jacobian must have two dimensions"
    assert jacobian.shape[0] == 2 * self.__N, "Jacobian must match the data"
    
    cov_r_inv = np.eye(2 * self.__N) # inv(eye) = eye 
    
    try:
      cov_p = np.linalg.inv(jacobian.T @ cov_r_inv @ jacobian)  
      std_p = np.sqrt(np.diag(cov_p)).reshape((6, 1))
    except np.linalg.LinAlgError as e:
      warnings.warn("Linalg-error occured with message: {}".format(e))
      cov_p = np.nan * np.ones((6,6))
      std_p = np.nan * np.ones((6,1))

    return cov_p, std_p


def localize(
      model_path  : str   = '../example_localization',
      query_path  : str   = '../example_localization/query/',
      image_str   : str   = 'IMG_8210.jpg'
    ) -> None:
  """
  Tries to localize a query image compared to the world frame.

  Input:
    model_path : str Path to folder containing the model (world)
    query_path : str Path to folder containing the query (image to localize)
    image_str  : str Which image should be used for localization
  """
  assert isinstance(model_path, str), "model_path must be a string"
  assert isinstance(query_path, str), "query_path must be a string"
  assert isinstance(image_str, str), "image_str must be a string"

  # For task 3.5
  use_weights = False
  sigma_u_std = 50.0
  sigma_v_std = 0.1

  # For task 3.6
  use_monte_carlo = False
  monte_carlo_iterations = 5

  sigma_f_std = 50
  sigma_cx_std = 0.1
  sigma_cy_std = 0.1

  default = (
    model_path == '../example_localization' and \
    query_path == '../example_localization/query/' and \
    image_str == 'IMG_8210.jpg'
  )

  model = os.path.join(*[sys.path[0], model_path])
  query = os.path.join(*[sys.path[0], query_path])

  if not default:
    K = np.loadtxt(f'{model}/K.txt')
    
    # matched_features = [keypoints | X3D | descriptors]
    matched_features = np.loadtxt(f'{model}/matched_features.txt')

    model_keypoints = matched_features[:, :2]
    X3D = matched_features[:, 2:5]
    model_descriptors = matched_features[:, 5:] 

    query_image = cv2.imread((query + image_str), cv2.IMREAD_GRAYSCALE)
    sift = ExtractFeaturesSIFT()#n_features=30000, contrast_threshold=0.01, edge_threshold=30)

    # Extract the same features in the image plane with the same method as previously 
    query_keypoints, query_descriptors = sift.extract_features(image=query_image)

    model_descriptors = model_descriptors.astype(np.float32, casting='same_kind')
    query_descriptors = query_descriptors.astype(np.float32, casting='same_kind')

    index_pairs, match_metric = match_features(
      features1=model_descriptors, 
      features2=query_descriptors, 
      max_ratio=0.85, 
      unique=True
    )
    model_keypoints = model_keypoints[index_pairs[:,0]]
    query_keypoints = query_keypoints[index_pairs[:,1]]

    model_descriptors = model_descriptors[index_pairs[:,0]]
    query_descriptors = query_descriptors[index_pairs[:,1]]

    X3D = X3D[index_pairs[:,0]] 

    # Filter out all values that have a negative Z-value
    # positive_indeces = X3D[:, 2] >= 0
    # X3D = X3D[positive_indeces]
    # query_keypoints = query_keypoints[positive_indeces]
    # query_descriptors = query_descriptors[positive_indeces]
    # model_descriptors = model_descriptors[positive_indeces]

    # Use solvePnPRansac to get initial guess on R and T
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
      objectPoints=X3D,
      imagePoints=query_keypoints,
      cameraMatrix=K,
      distCoeffs=np.zeros((1,4), dtype=np.float32), # Assuming that the images are undistorted before use
      reprojectionError=3.0
    )

    X3D = X3D[inliers[:, 0]]
    query_keypoints = query_keypoints[inliers[:, 0]]

    if use_monte_carlo and monte_carlo_iterations > 0:
      print("Running in total {} monte-carlo simulations".format(monte_carlo_iterations))
      cov = np.diag([sigma_f_std**2, sigma_cx_std**2, sigma_cy_std**2])
    else:
      monte_carlo_iterations = 1
      cov = np.zeros((3, 3))

    state_estimates = np.zeros((monte_carlo_iterations, 6))
    standard_deviations = np.zeros((monte_carlo_iterations, 6))
    reprojection_errors = np.zeros((monte_carlo_iterations, 1))

    for idx in range(monte_carlo_iterations):
      if use_monte_carlo:
        print("Simulation number {}".format(idx))
      
      # Create multivariate distribution of the intrinsic parameter
      eta = np.random.multivariate_normal(mean=np.zeros((3, 1)).flatten(), cov=cov)
      eta = np.array(
        [
          [eta[0], 0,      eta[1]], 
          [0,      eta[0], eta[2]], 
          [0,      0,      0]
        ]
      )
      K = K + eta

      # Optimize the pose using nonlinear least squares
      optimize_query_pose = OptimizeQueryPose(
        K=K, 
        query_uv=query_keypoints, 
        X3D=X3D,
        use_weights=use_weights,
        sigma_u_std=sigma_u_std,
        sigma_v_std=sigma_v_std
      )
      x, R, t, reprojection_error, cov_x, std_x = optimize_query_pose.nonlinear_least_squares(rvecs=rvecs, tvecs=tvecs)

      standard_deviations[idx, :] = std_x.T
      reprojection_errors[idx, :] = reprojection_error
      state_estimates[idx, :] = x.T
    
    if use_monte_carlo:
      print(state_estimates)
      cov_p = np.cov(state_estimates.T) 
      std_p = np.sqrt(np.diag(cov_p))
      np.savetxt(f'{query}/sfm/{image_str}_cov.txt', cov_p)
      np.savetxt(f'{query}/sfm/{image_str}_std.txt', std_p)
      
      print("Covariance matrix over {} iterations".format(monte_carlo_iterations))
      print(cov_p.reshape((1,-1)))

      print("Standard deviations over {} iterations".format(monte_carlo_iterations))
      print(std_p.reshape((1,-1)))

    else:
      print("Covariance matrix")
      print(cov_x)

      rod_std = std_x[:3]
      R_std, _ = cv2.Rodrigues(rod_std)
      
      # This might suffer Gimbal lock
      try:
        # https://stackoverflow.com/questions/11514063/extract-yaw-pitch-and-roll-from-a-rotationmatrix
        yaw = math.atan2(R_std[1,0], R_std[0,0])
        pitch = math.atan2(-R_std[2,0], math.sqrt(R_std[2,1]**2 + R_std[2,2]**2))
        roll = math.atan2(R_std[2,1], R_std[2,2])
        rpy = np.array([roll, pitch, yaw])
        std_x[:3] = rpy.reshape((3,1))
      except Error as e:
        print("Error occured when calculating roll, pitch, yaw with message: {}".format(e))
      print("Standard deviations [rad | m]") # Uncertain on how to calculate the std into m
      print(std_x)

      print("Rotation matrix")
      print(R)

      print("Translation")
      print(t)
    
    np.savetxt(f'{query}/sfm/{image_str}_standard_deviations.txt', standard_deviations)
    np.savetxt(f'{query}/sfm/{image_str}_reprojection_errors.txt', reprojection_errors)

    print("Average reprojection error over {} iterations".format(monte_carlo_iterations))
    print(reprojection_errors.mean(axis=0))

    # Develop model-to-query transformation by [[R, t], [0, 0, 0, 1]]
    # NOTE: Will only use the last rotation matrix and the last translation vector if one uses the 
    # monte-carlo simulations
    R, _ = cv2.Rodrigues(rvecs)
    t = tvecs.reshape((3, 1))
    T_m2q = np.block(
      [
        [R,                t], 
        [np.zeros((1, 3)), 1]
      ]
    )
    # T_m2q = np.linalg.inv(T_m2q)
    print("T_m2q")
    print(T_m2q)

    # Prepare for plotting
    X = X3D.T
    colors = np.zeros((X3D.shape[0], 3))

    # Store the data
    np.savetxt(f'{query}/sfm/{image_str}_T_m2q.txt', T_m2q)
    # np.savetxt(f'{query}/sfm/reprojection_error.txt', reprojection_error)
    np.savetxt(f'{query}/sfm/{image_str}_inliers.txt', inliers)

  else:
    # Load features from the world frame
    # 3D points [4 x num_points].
    X = np.loadtxt(f'{model}/X.txt')

    # Model-to-query transformation.
    # If you estimated the query-to-model transformation,
    # then you need to take the inverse.
    T_m2q = np.loadtxt(f'{query}/IMG_8210_T_m2q.txt')

    # If you have colors for your point cloud model...
    colors = np.loadtxt(f'{model}/c.txt') 

  # These control the visible volume in the 3D point cloud plot.
  # You may need to adjust these if your model does not show up.
  xlim = [-10,+10]
  ylim = [-10,+10]
  zlim = [0,+20]

  frame_size = 1
  marker_size = 5

  plt.figure('3D point cloud with image {}'.format(image_str), figsize=(6,6))
  plotting.draw_point_cloud(
    X=X, 
    T_m2q=T_m2q, 
    xlim=xlim, 
    ylim=ylim, 
    zlim=zlim, 
    colors=colors, 
    marker_size=marker_size, 
    frame_size=frame_size
  )
  plt.tight_layout()
  

if __name__ == '__main__':
  model_path = os.path.join(sys.path[0], "../data/results/task_2_1")
  query_path = os.path.join(sys.path[0], "../data/hw5_ext/undistorted/")
  image_str = "IMG_8220.jpg"
  localize(
    model_path=model_path,
    query_path=query_path,
    image_str=image_str
  )
  plt.show()
  # localize()
  # plt.show()
