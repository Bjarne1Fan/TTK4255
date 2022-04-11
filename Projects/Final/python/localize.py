import os 
import sys
import warnings
import cv2

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import least_squares

import plotting
import common
from model_reconstruction import ExtractFeaturesSIFT
from matlab_inspired_interface import match_features, show_matched_features

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
        x     : np.ndarray
      ) -> np.ndarray:
    """
    Residual function for minimzing the reprojection errors, by
    optimizing the estimated pose.

    You can then estimate the camera pose via minimizing 
    reprojection errors. 

    Projecting the 3D-coordinates down into the camera plane, and calculating the 
    distance to the actual points. Returning this error
    
    The array that must be projected down into the plane however, must first be 
    rotated such that the values are correct

    OBS! Must make sure that the estimated values also fits as a rotation matrix.
    Must therefore run the closest rotation matrix to guarantee that the results
    become proper

    It is assumed that the input will be in the order
    x = [r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3] # NOTE This is outdated
    x = [roll, theta, yaw, t1, t2, t3] # NOTE Is better
    x = [rvecs, tvecs] NOTE: Is optimal, such that one does not require to interpret what is
    roll, pitch and yaw, and the algorithm may do everything for ourselves

    Or will it really? Couldn't it just be angles returned in yaw, theta, roll as
    in the midterm project? 
    
    Also there is a bug where the initial values are not used properly

    Makes more sence that the optimization algorithm will return the three different
    angles instead of the full rotation matrix. It will be far less work to optimize
    this as well

    One sees that the rotation matrix does not fit. Can this be caused by the offset 
    between the axes used by cv2 and us?????? TODO: Check out this theory!
    """
    roll, pitch, yaw = x[:3]
    # # R = common.closest_rotation_matrix(x[:9].reshape((3, 3)))
    R_4x4 = common.rotate_x(roll) @ common.rotate_y(pitch) @ common.rotate_z(yaw) @ self.__R0_4x4
    R = common.closest_rotation_matrix(R_4x4[:3,:3])

    # rvecs = x[:3].reshape((3, 1))
    # R, _ = cv2.Rodrigues(rvecs)
    
    t = x[3:].reshape((3, 1))
    # T = common.translate(t[0,0], t[1,0], t[2,0]) @ R_4x4
    # print(x)

    # NOTE: I am certain that this contains at least one error
    # I think that the inverse must be used in this case, as we would like the
    # points given in the plane belonging to the camera in question 
    # X = common.closest_rotation_matrix(R_4x4[:3,:3]) @ self.__X3D.T + t # Will the rotation and the translation be the inverse of what is used here?
    # X = T @ self.__X3D1.T
    # X = X[:3,:] / X[3,:]
    X = R @ self.__X3D.T + t # I think that it is correct according to L3 S64
    uv_hat = common.project(arr=X, K_inv=self.__K_inv).T

    assert uv_hat.shape[0] == self.__query_uv.shape[0], "Rows must be identical"
    assert uv_hat.shape[1] == self.__query_uv.shape[1], "Cols must be identical"

    residuals = self.__query_uv - uv_hat
    residuals = np.hstack([residuals[:,0].T, residuals[:,1].T]) # Horizontal and then the vertical errors

    if self.__use_weights:
      residuals = self.__L_inv @ residuals

    return residuals

  def __jacobian(self) -> np.ndarray:
    """
    Returns the jacobian corresponding to the optimization
    problem. 

    Has no impact if the LM-method is used
    """
    return None # Until further is known about the sparsity of the model

  def nonlinear_least_squares(
        self,
        rvecs : np.ndarray, 
        tvecs : np.ndarray
      ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Nonlinear least squares that must be solved to get R and T better 
    refined. Using LM-optimization, as used in the midterm project
    """
    R, _ = cv2.Rodrigues(rvecs) 
    t = -tvecs.reshape((3, 1))  # Output from decomposeProjectionMatrix imply that the t-vec should be negated. Why?
    R0 = np.eye(4)
    R0[:3,:3] = R
    self.__R0_4x4 = R0
    # self.__R0_4x4 = np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]])
    # self.__t0 = -t # Output from decomposeProjectionMatrix imply that this should be negated. Why?

    # P = np.block([R, t])
    # # print(R)
    # # https://github.com/mpatacchiola/deepgaze/issues/3
    # __K, __R, __trans_vec, __Rx, __Ry, __Rz, euler_angles = cv2.decomposeProjectionMatrix(
    #   projMatrix=P, 
    #   cameraMatrix=self.__K, 
    #   rotMatrix=R, 
    #   transVect=np.block([[t],[1]])
    # )
    # # print(euler_angles)

    # # NOTE: The euler angles are returned as pitch - yaw - roll
    # # It is desired to have it in the format roll - pitch - yaw
    # # https://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp/?answer=52913#post-id-52913
    # euler_angles = np.array([euler_angles[2], euler_angles[0], euler_angles[1]])

    # x0 = np.block([R0.flatten(), t0.flatten()])
    # x0 = np.hstack([euler_angles.T, t.T]).flatten() 
    # Output from decomposeProjectionMatrix imply that the t-vec should be negated
    zeros = np.zeros((1, 3))
    x0 = np.hstack([zeros.reshape((1, -1)), -tvecs.reshape((1, -1))]).flatten()
    # x0 = np.hstack([rvecs.reshape((1, -1)), -tvecs.reshape((1, -1))]).flatten()
    # print(R)

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
      x = optimization_results.x

      # roll, pitch, yaw = x[:3]

      # R_4x4 = common.rotate_x(roll) @ common.rotate_y(pitch) @ common.rotate_z(yaw) @ self.__R0_4x4
      # R = R_4x4[:3, :3]
      rvecs = x[:3]
      R, _ = cv2.Rodrigues(rvecs)
      t = x[3:].reshape((3, 1))
      reprojection_error = optimization_results.cost
      jacobian = optimization_results.jac
      uncertainty = self.__uncertainty(jacobian=jacobian)
    else:
      warnings.warn("Optimization did not converge! Reason: {}. Returning initial values!".format(optimization_results.message))
      reprojection_error = np.infty
      uncertainty = (np.infty, np.infty)
    # R = common.closest_rotation_matrix(R)
    return R, t, reprojection_error, uncertainty[0], uncertainty[1]

  def __uncertainty(
        self,
        jacobian : np.ndarray
      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the uncertainty regarding the estimates. By using that 
    the jacobian is calculated during the optimization.

    Returns the full covariance-matrix, as well as the standard deviations 
    """
    assert isinstance(jacobian, np.ndarray), "Jacobian must be given as an ndarray"
    assert len(jacobian.shape) == 2, "Jacobian must have two dimensions"
    assert jacobian.shape[0] == 2 * self.__N, "Jacobian must match the data"
    
    cov_r_inv = np.eye(2 * self.__N) # inv(eye) = eye 
    # JTJ = jacobian.T @ jacobian
    
    try:
      cov_p = np.linalg.inv(jacobian.T @ cov_r_inv @ jacobian)  
      std_p = np.sqrt(np.diag(cov_p))
    except np.linalg.LinAlgError as e:
      warnings.warn("Linalg-error occured with message: {}".format(e))
      cov_p = np.nan((6,6))
      std_p = np.nan((1,6))

    return cov_p, std_p


def localize(
      model_path      : str   = '../example_localization',
      query_path      : str   = '../example_localization/query/',
      image_str       : str   = 'IMG_8210.jpg'
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

  # For task 3.5
  use_weights = False
  sigma_u_std = 50.0
  sigma_v_std = 0.1

  # For task 3.6
  use_monte_carlo = False
  monte_carlo_iterations = 500

  sigma_f_std = 50
  sigma_cx_std = 0.1
  sigma_cy_std = 0.1

  assert isinstance(image_str, str), "Image id must be a string"

  default = (
    model_path == '../example_localization' and \
    query_path == '../example_localization/query/' and \
    image_str == 'IMG_8210.jpg'
  )

  model = os.path.join(*[sys.path[0], model_path])
  query = os.path.join(*[sys.path[0], query_path])

  if not default:
    K = np.loadtxt(f'{model}/K.txt')
    
    # matched_features = [features | X3D | descriptors]
    matched_features = np.loadtxt(f'{model}/matched_features.txt')

    model_keypoints = matched_features[:, :2]
    X3D = matched_features[:, 2:5]
    model_descriptors = matched_features[:, 5:] 

    query_image = cv2.imread((query + image_str), cv2.IMREAD_GRAYSCALE)
    sift = ExtractFeaturesSIFT()#n_features=30000, contrast_threshold=0.01, edge_threshold=30)

    # Extract the same features in the image plane with the same method as previously 
    query_keypoints, query_descriptors = sift.extract_features(image=query_image)

    # model_keypoints = model_keypoints.astype(np.float32, casting='same_kind')
    # query_keypoints = query_keypoints.astype(np.float32, casting='same_kind')

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

    standard_deviations = np.zeros((monte_carlo_iterations, 6))
    reprojection_errors = np.zeros((1, monte_carlo_iterations))

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
      R, t, reprojection_error, cov_p, std_p = optimize_query_pose.nonlinear_least_squares(rvecs=rvecs, tvecs=tvecs)

      standard_deviations[idx, :] = std_p
      reprojection_errors[0, idx] = reprojection_error
    
    np.savetxt(f'{query}/sfm/standard_deviations.txt', standard_deviations)
    np.savetxt(f'{query}/sfm/reprojection_errors.txt', reprojection_errors)

    print("Average standard deviations over {} iterations".format(monte_carlo_iterations))
    print(standard_deviations.mean(axis=0))
    print("Average reprojection error over {} iterations".format(monte_carlo_iterations))
    print(reprojection_errors.mean(axis=1))

    # Develop model-to-query transformation by [[R, t], [0, 0, 0, 1]]
    # NOTE: Will only use the last rotation matrix and the last translation vector if one uses the 
    # monte-carlo simulations
    T_m2q = np.block(
      [
        [R,                t], 
        [np.zeros((1, 3)), 1]
      ]
    ) # TODO: Check if this must be inverted
    print(T_m2q)
    # T_m2q = np.linalg.inv(T_m2q)

    # Prepare for plotting
    X = X3D.T
    colors = np.zeros((X3D.shape[0], 3))

    # Store the data
    np.savetxt(f'{query}/sfm/T_m2q.txt', T_m2q)
    # np.savetxt(f'{query}/sfm/reprojection_error.txt', reprojection_error)
    np.savetxt(f'{query}/sfm/inliers.txt', inliers)

  else:
    # Load features from the world frame
    # 3D points [4 x num_points].
    X = np.loadtxt(f'{model}/X.txt')
    print(X.shape)

    # Model-to-query transformation.
    # If you estimated the query-to-model transformation,
    # then you need to take the inverse.
    T_m2q = np.loadtxt(f'{query}/IMG_8210_T_m2q.txt')

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

  # if include_default:
  #   # Include the default pose
  #   model_path = '../example_localization'
  #   query_path = '../example_localization/query/'
  #   image_str = 'IMG_8210'

  #   model = os.path.join(*[sys.path[0], model_path])
  #   query = os.path.join(*[sys.path[0], query_path + image_str])

  #   # Load features from the world frame
  #   # 3D points [4 x num_points].
  #   X = np.loadtxt(f'{model}/X.txt')

  #   # Model-to-query transformation.
  #   # If you estimated the query-to-model transformation,
  #   # then you need to take the inverse.
  #   T_m2q = np.loadtxt(f'{query}_T_m2q.txt')

  #   # If you have colors for your point cloud model...
  #   colors = np.loadtxt(f'{model}/c.txt') # RGB colors [num_points x 3].

  #   plotting.draw_point_cloud(
  #     X=X, 
  #     T_m2q=T_m2q, 
  #     xlim=xlim, 
  #     ylim=ylim, 
  #     zlim=zlim, 
  #     colors=colors, 
  #     marker_size=marker_size, 
  #     frame_size=frame_size
  #   )
  #   plt.tight_layout()
  

if __name__ == '__main__':
  model_path = os.path.join(sys.path[0], "../data/results/task_2_1")
  query_path = os.path.join(sys.path[0], "../data/hw5_ext/undistorted/")
  image_str = "IMG_8218.jpg"
  localize(
    model_path=model_path,
    query_path=query_path,
    image_str=image_str
  )
  # localize()
  plt.show()
