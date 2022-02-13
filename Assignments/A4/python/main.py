import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import common as com

K           = np.loadtxt(os.path.join(sys.path[0], '../data/K.txt'))
detections  = np.loadtxt(os.path.join(sys.path[0], '../data/detections.txt'))
XY          = np.loadtxt(os.path.join(sys.path[0],'../data/XY.txt')).T
n_total     = XY.shape[1] # Total number of markers (= 24)

fig = plt.figure(figsize=plt.figaspect(0.35))
K_inv = np.linalg.inv(K)

# for image_number in range(23): # Use this to run on all images
for image_number in [4]: # Use this to run on a single image

  # Load data
  # valid : Boolean mask where valid[i] is True if marker i was detected
  #     n : Number of successfully detected markers (<= n_total)
  #    uv : Pixel coordinates of successfully detected markers
  valid = detections[image_number, 0::3] == True
  uv = np.vstack((detections[image_number, 1::3], detections[image_number, 2::3]))
  uv = uv[:, valid]
  n = uv.shape[1]

  # Tip: The 'valid' array can be used to perform Boolean array indexing,
  # e.g. to extract the XY values of only those markers that were detected.
  # Use this when calling estimate_H and when computing reprojection error.

  # Tip: Helper arrays with 0 and/or 1 appended can be useful if
  # you want to replace for-loops with array/matrix operations.

  # XY01 = np.vstack((XY, np.zeros(n_total), np.ones(n_total)))

  uv1 = np.vstack((uv, np.ones(n)))
  xyz = K_inv @ uv1
  xy = xyz[:2]
  H = com.estimate_H(xy, XY[:, valid])  

  XY1 = np.vstack((XY, np.ones(n_total)))
  uv_from_H = com.project(K, H @ XY1) 

  # Calculating the reprojection errors
  reprojection_errors = np.linalg.norm(uv - uv_from_H, axis=0)
  min_reprojection_error = np.min(reprojection_errors, axis=0)
  max_reprojection_error = np.max(reprojection_errors, axis=0)
  avg_reprojection_error = np.mean(reprojection_errors, axis=0)

  print(min_reprojection_error)
  print(avg_reprojection_error)
  print(max_reprojection_error)
  # print(reprojection_errors)

  T1, T2 = com.decompose_H(H)           

  T = T1                                

  # The figure should be saved in the data directory as out0000.png, etc.
  # NB! generate_figure expects the predicted pixel coordinates as 'uv_from_H'.
  plt.clf()
  com.generate_figure(fig, image_number, K, T, uv, uv_from_H, XY)
  plt.savefig(os.path.join(sys.path[0],'../data/out/out%04d.png' % image_number))
