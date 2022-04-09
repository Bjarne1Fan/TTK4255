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
        K         : np.ndarray,
        query_uv  : np.ndarray,
        X3D       : np.ndarray
      ) -> None:
    self.__K = K
    self.__K_inv = np.linalg.inv(K)
    self.__query_uv = query_uv
    self.__X3D = X3D 

  def __residual_function(
        self,
        x         : np.ndarray
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

    Or will it really? Couldn't it just be angles returned in yaw, theta, roll as
    in the midterm project? 
    
    Also there is a bug where the initial values are not used properly

    Makes more sence that the optimization algorithm will return the three different
    angles instead of the full rotation matrix. It will be far less work to optimize
    this as well

    NOTE: SHould this include R0?
    """
    roll, pitch, yaw = x[:3]
    # R = common.closest_rotation_matrix(x[:9].reshape((3, 3)))
    R = common.rotate_x(roll) @ common.rotate_y(pitch) @ common.rotate_z(yaw) @ self.__R0    
    t = x[3:].reshape((3, 1))

    # NOTE: I am certain that this contains at least one error
    # I think that the inverse must be used in this case, as we would like the
    # points given in the plane belonging to the camera in question 
    X = R[:3,:3] @ self.__X3D.T + t # Will the rotation and the translation be the inverse of what is used here?
    uv_hat = common.project(arr=X, K_inv=self.__K_inv).T

    assert uv_hat.shape[0] == self.__query_uv.shape[0], "Rows must be identical"
    assert uv_hat.shape[1] == self.__query_uv.shape[1], "Cols must be identical"

    N = X.shape[1]
    residuals = np.zeros((1, N))
    
    for i in range(N):
      temp0 = i*2
      temp1 = (i+1)*2 - 1
      residuals[0 : 1] = (self.__query_uv[i,:] - uv_hat[i,:]).reshape((1, 2)) # Fucked here
    
    return residuals

    # residuals = uv_hat - self.__query_uv
    # return residuals.flatten() # Does it work to flatten this shit?

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
      ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Nonlinear least squares that must be solved to get R and T better 
    refined. Using LM-optimization, as used in the midterm project
    """
    R, _ = cv2.Rodrigues(rvecs) 
    t = tvecs.reshape((3, 1))  # Should this be positive or negative?
    self.__R0 = np.block([[R, np.zeros((3, 1))], [np.zeros((1, 4))]])
    self.__t0 = t

    P = np.block([R, t])
    # https://github.com/mpatacchiola/deepgaze/issues/3
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
      projMatrix=P, 
      cameraMatrix=self.__K, 
      rotMatrix=R, 
      transVect=np.block([[t],[1]])
    )

    # NOTE: The euler angles are returned as pitch - yaw - roll
    # It is desired to have it in the format roll - pitch - yaw
    # https://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp/?answer=52913#post-id-52913
    euler_angles = np.array([euler_angles[2], euler_angles[0], euler_angles[1]])

    # x0 = np.block([R0.flatten(), t0.flatten()])
    x0 = np.hstack([euler_angles.T, t.T]).flatten()

    optimization_results = least_squares(
      fun=self.__residual_function,
      x0=x0,
      jac_sparsity=self.__jacobian()
    )
    success = optimization_results.success

    if success:
      # Optimization converged
      x = optimization_results.x

      roll, pitch, yaw = x[:3]

      t = x[3:].reshape((3, 1))
      reprojection_error = optimization_results.cost
    else:
      warnings.warn("Optimization did not converge! Reason: {}. Returning initial values!".format(optimization_results.message))
      reprojection_error = np.infty
    R = common.closest_rotation_matrix(R)
    return R, t, reprojection_error


def localize(
      model_path  : str   = '../example_localization',
      query_path  : str   = '../example_localization/query/',
      image_str   : str   = 'IMG_8210.jpg'
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

  default = (
    model_path == '../example_localization' and \
    query_path == '../example_localization/query/' and \
    image_str == 'IMG_8210.jpg'
  )

  model = os.path.join(*[sys.path[0], model_path])
  query = os.path.join(*[sys.path[0], query_path])

  if not default:
    K = np.loadtxt(f'{model}/K.txt')
    
    # matched_features = [features | X3D]
    matched_features = np.loadtxt(f'{model}/matched_features.txt')

    model_keypoints = matched_features[:, :2]
    X3D = matched_features[:, 2:]
    X3D1 = np.block([X3D, np.ones((X3D.shape[0], 1))])

    query_image = cv2.imread((query + image_str), cv2.IMREAD_GRAYSCALE)
    sift = ExtractFeaturesSIFT(n_features=0, contrast_threshold=0.05, edge_threshold=25)

    # Extract the same features in the image plane with the same method 
    # as previously 
    query_keypoints, query_descriptors = sift.extract_features(image=query_image)

    model_keypoints = model_keypoints.astype(np.float32, casting='same_kind')
    query_keypoints = query_keypoints.astype(np.float32, casting='same_kind')
    index_pairs, match_metric = match_features(
      features1=model_keypoints, 
      features2=query_keypoints, 
      max_ratio=1.0, 
      unique=True
    )
    model_keypoints = model_keypoints[index_pairs[:,0]]
    query_keypoints = query_keypoints[index_pairs[:,1]]

    X3D = X3D[index_pairs[:,0]] # 0 or 1? NOTE. Or is it that we would like to find 2D-3D point correspondances?

    # model_uv1 = np.vstack([model_keypoints.T, np.ones(model_keypoints.shape[0])])
    # query_uv1 = np.vstack([query_keypoints.T, np.ones(query_keypoints.shape[0])])

    # Use solvePnPRansac to get initial guess on R and T
    # It is assumed that the image is undistorted before use
    # NOTE: Important that the type is float or uint8_t
    # X3D1 = X3D1.astype(np.float64, casting='same_kind')
    # query_uv1 = query_uv1.astype(np.float32, casting='same_kind')
    # K = K.astype(np.float32, casting='same_kind')
    # model_uv1 = model_uv1.astype(np.float64)
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
      objectPoints=X3D,#model_uv1.T, # Thinks that it will be wrong to send in these values, as it does 
                                # not contain any information regarding the z-vector. Only the uv-plane
      imagePoints=query_keypoints,#query_uv1[:2].T, # Slicing can be dangerous!
      cameraMatrix=K,
      distCoeffs=np.zeros((1,4), dtype=np.float32) # Assuming that the images are undistorted before use
    )
    # print(rvecs)  # Why does this only returning three different values in the vector? This makes no fucking
                  # sense at all! One could skew it, but that would definetly not be correct
                  # Is it really euler angles????? WHy the fuck would one represent it using euler angles ¨
                  # and not quaternions or the rotation matrix??
    # I assume that the rvecs contain the rotation in euler angles
    # After reading a bit through some forums, a major pain in the but might be the use of !! LEFT !! hand-
    # convection used by opencv. 

    # Or could it be another convention, like rodriques?
    # https://answers.opencv.org/question/134017/need-explaination-about-rvecs-returned-from-solvepnp/
    # https://www.reddit.com/r/opencv/comments/kczhoc/question_solvepnp_rvecs_what_do_they_mean/
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#rodrigues
    # It is returned as a rotation vector. See documentation for the rodrigues in cv2

    # NOTE from documentation of rodrigues: 
    # A rotation vector is a convenient and most compact representation of a rotation matrix (since any 
    # rotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry 
    # optimization procedures like @ref calibrateCamera, @ref stereoCalibrate, or @ref solvePnP .
    # Use a nonlinear least squares to refine R and T
    optimize_query_pose = OptimizeQueryPose(
      K=K, 
      query_uv=query_keypoints, 
      X3D=X3D
    )
    R, t, reprojection_error = optimize_query_pose.nonlinear_least_squares(rvecs=rvecs, tvecs=tvecs) # Can't one just input the 3x1 rodrigues vector?
    print(reprojection_error)

    # Develop model-to-query transformation by [[R, t], [0, 0, 0, 1]]
    T_m2q = np.block(
      [
        [R,                t], 
        [np.zeros((1, 3)), 1]
      ]
    ) # TODO: Check if this must be inverted

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

  plt.figure('3D point cloud with image {}'.format(image_str), figsize=(6,6))
  plotting.draw_point_cloud(
    X, T_m2q, xlim, ylim, zlim, 
    colors=colors, marker_size=marker_size, frame_size=frame_size
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
