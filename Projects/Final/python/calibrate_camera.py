import os 
import sys

import warnings
import numpy as np
import cv2 as cv
import glob

from os.path import join, basename, realpath, dirname, exists


def calibrate() -> tuple:
  # This string should point to the folder containing the images
  # used for calibration. The same folder will hold the output.
  # "*.jpg" means that any file with a .jpg extension is used
  image_path_pattern = os.path.join(sys.path[0], '../data/hw5_ext/calibration/*.jpg')
  output_folder = dirname(image_path_pattern)

  board_size = np.array([4, 7]) # Number of internal corners of the checkerboard (see tutorial)
  square_size = 50 # Real world length of the sides of the squares (see HW6 Task 1.5)

  #
  # Tip:
  # You can pass flags to calibrateCameraExtended to specify what distortion
  # coefficients to estimate. See https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
  # Below are two examples of flags that may be interesting to try.
  #
  calibrate_flags = 0 # Use default settings (three radial and two tangential)
  # calibrate_flags = cv.CALIB_ZERO_TANGENT_DIST|cv.CALIB_FIX_K3 # Disable tangential distortion and third radial distortion coefficient

  # Flags to findChessboardCorners that improve performance
  detect_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

  # Termination criteria for cornerSubPix routine
  subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  #
  # Detect checkerboard points
  #
  # Note: This first tries to use existing checkerboard detections,
  # if present in the output folder. You can delete the 'u_all.npy'
  # file to force the script to re-detect all the corners.
  #
  if exists(join(output_folder, 'u_all.npy')):
    u_all = np.load(join(output_folder, 'u_all.npy'))
    X_all = np.load(join(output_folder, 'X_all.npy'))
    image_size = np.loadtxt(join(output_folder, 'image_size.txt')).astype(np.int32)
    print('Using previous checkerboard detection results.')
  else:
    X_board = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    X_board[:,:2] = square_size * np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    X_all = []
    u_all = []
    image_names = []
    image_size = None
    for image_path in glob.glob(image_path_pattern):
      print('%s...' % basename(image_path), end='')

      I = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
      if not image_size:
        image_size = I.shape
      elif I.shape != image_size:
        warnings.warn_explicit('Image size is not identical for all images.')
        warnings.warn_explicit('Check image "%s" against the other images.' % basename(image_path))
        quit()

      ok, u = cv.findChessboardCorners(I, (board_size[0],board_size[1]), detect_flags)
      if ok:
        print('detected all %d checkerboard corners.' % len(u))
        X_all.append(X_board)
        u = cv.cornerSubPix(I, u, (11,11), (-1,-1), subpix_criteria)
        u_all.append(u)
        image_names.append(basename(image_path))
      else:
        print('failed to detect checkerboard corners, skipping.')

    with open(join(output_folder, 'image_names.txt'), 'w+') as f:
      for i,image_name in enumerate(image_names):
        f.write('%d: %s\n' % (i, image_name))

    np.savetxt(join(output_folder, 'image_size.txt'), image_size)
    np.save(join(output_folder, 'u_all.npy'), u_all) # Detected checkerboard corner locations
    np.save(join(output_folder, 'X_all.npy'), X_all) # Corresponding 3D pattern coordinates

  print('Calibrating. This may take a minute or two...', end='')
  results = cv.calibrateCameraExtended(X_all, u_all, image_size, None, None, flags=calibrate_flags)
  print('Done!')

  ok, K, dc, rvecs, tvecs, std_int, std_ext, per_view_errors = results

  mean_errors = []
  for i in range(len(X_all)):
    u_hat, _ = cv.projectPoints(X_all[i], rvecs[i], tvecs[i], K, dc)
    vector_errors = (u_hat - u_all[i])[:,0,:] # The indexing here is because OpenCV likes to add extra dimensions.
    scalar_errors = np.linalg.norm(vector_errors, axis=1)
    mean_errors.append(np.mean(scalar_errors))

  np.savetxt(join(output_folder, 'K.txt'), K) # Intrinsic matrix (3x3)
  np.savetxt(join(output_folder, 'dc.txt'), dc) # Distortion coefficients
  np.savetxt(join(output_folder, 'mean_errors.txt'), mean_errors)
  np.savetxt(join(output_folder, 'std_int.txt'), std_int) # Standard deviations of intrinsics (entries in K and distortion coefficients)
  print('Calibration data is saved in the folder "%s"' % realpath(output_folder))



def __resize_with_aspect_ratio(
      image   : np.ndarray, 
      width   : int         = None, 
      height  : int         = None, 
      inter   : int         = cv.INTER_AREA
    ) -> np.ndarray:
  # From SO:
  # https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display?fbclid=IwAR1WQrO2nbWIHFwkHNrYpxZf2hbv1Xzq7AmF420q22vAquxTlwkVlsVR3K8
  
  dim = None
  (h, w) = image.shape[:2]

  if width is None and height is None:
    return image
  if width is None:
    r = height / float(h)
    dim = (int(w * r), height)
  else:
    r = width / float(w)
    dim = (width, int(h * r))

  return cv.resize(image, dim, interpolation=inter)


def undistort_image(
      distorted_image         : np.ndarray, 
      K                       : np.ndarray,
      distortion_coefficients : np.ndarray
    ) -> np.ndarray:
  k1, k2, p1, p2, k3 = distortion_coefficients[:]

  # Undistorting with the original 
  undistorted_image = cv.undistort(
    src=distorted_image,
    cameraMatrix=K, 
    distCoeffs=np.array([k1, k2, p1, p2, k3])
  )

  return undistorted_image


def test_camera_distortion_n_sigma(n_sigma : float = 3.0):
  """
  This method tries to undistort an image, multiplying the 
  obtained standard deviations with n_sigma. Only the distortion
  parameters are affected.

  By multiplying with n_sigma, it is theorized that the undistortion
  will include more dramatic effects compared to just using the 
  normal distributions to sample the distortion coefficients
  """

  folder = os.path.join(sys.path[0], '../data/hw5_ext/calibration')

  K                       = np.loadtxt(join(folder, 'K.txt'))
  distortion_coefficients = np.loadtxt(join(folder, 'dc.txt'))
  std_int                 = np.loadtxt(join(folder, 'std_int.txt'))
  u_all                   = np.load(join(folder, 'u_all.npy'))
  image_size              = np.loadtxt(join(folder, 'image_size.txt')).astype(np.int32) # height,width
  mean_errors             = np.loadtxt(join(folder, 'mean_errors.txt'))

  # Extract components of intrinsics standard deviation vector. See:
  # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
  k1, k2, p1, p2, k3 = distortion_coefficients[:]
  _, _, _, _, k1_std, k2_std, p1_std, p2_std, k3_std, _, _, _, _, _, _, _, _, _ = std_int

  image_path_pattern = os.path.join(sys.path[0], '../data/hw5_ext/calibration/*.jpg')
  distorted_image = cv.imread(glob.glob(image_path_pattern)[53])

  # Undistorting with the original 
  undistorted_image = cv.undistort(
    src=distorted_image,
    cameraMatrix=K, 
    distCoeffs=np.array([k1, k2, p1, p2, k3]) + np.array([k1_std, k2_std, p1_std, p2_std, k3_std]) * n_sigma
  )

  resized_image = __resize_with_aspect_ratio(undistorted_image, height=800)
  cv.imshow('Undistorted image', resized_image)
  cv.waitKey(0)

  return undistorted_image

if __name__ == '__main__':
  calibrate()
  test_camera_distortion_n_sigma(n_sigma=3.0)
