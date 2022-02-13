import os
import sys

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from numpy import ndarray

def estimate_H(xy: ndarray, XY: ndarray)->ndarray:
  # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
  # The column of V corresponding to the smallest singular value
  # is the last column, as the singular values are automatically
  # ordered by decreasing magnitude. However, note that it returns
  # V transposed.

  n = XY.shape[1]
  A = np.zeros((2*n, 9))

  # Find a better method instead of just a for-loop
  for r, c in zip(range(0, A.shape[0], 2), range(n)):
    A[r] = np.array([XY[0, c], XY[1, c], 1, 0, 0, 0, -XY[0, c] * xy[0, c], -XY[1, c] * xy[0, c], -xy[0, c]])
    A[r + 1] = np.array([0, 0, 0, XY[0,c], XY[1, c], 1, -XY[0, c]*xy[1, c], -XY[1, c] * xy[1, c], -xy[1, c]])

  _, _, VT = np.linalg.svd(A)
  # h is the last column in V, thus being the last row in VT
  h = VT[-1]

  H = h.reshape((3, 3))
  return H


def decompose_H(H: ndarray)->Tuple:
  # Tip: Use np.linalg.norm to compute the Euclidean length
  # The function is supposed to return the two transformation-matrices
  # corresponding from transforming from camera to the world 
  abs_k = np.linalg.norm(H[:,0])
  r0 = H[:,0] / abs_k
  r1 = H[:,1] / abs_k
  r2 = np.cross(r0, r1)
  
  t = H[:,2] / abs_k

  t1 = t
  t2 = -t1

  R1 = np.column_stack((r0, r1, r2))
  R2 = np.column_stack((-r0, -r1, r2)) # Note sign of r2

  print(np.linalg.det(R1))
  print(np.linalg.det(R2))

  R1 = closest_rotation_matrix(R1)
  R2 = closest_rotation_matrix(R2)

  print(np.linalg.det(R1))
  print(np.linalg.det(R2))

  T1 = np.eye(4)
  T2 = np.eye(4)
  T1[:3,:4] = np.column_stack((R1, t1))
  T2[:3,:4] = np.column_stack((R2, t2))

  return T1, T2

def closest_rotation_matrix(Q: ndarray)->ndarray:
  U, _, VT = np.linalg.svd(Q)
  R = U @ VT
  return R

def project(K: ndarray, X: ndarray)->ndarray:
  """
  Computes the pinhole projection of an (3 or 4)xN array X using
  the camera intrinsic matrix K. Returns the dehomogenized pixel
  coordinates as an array of size 2xN.
  """
  uvw = K @ X[:3,:]
  uvw /= uvw[2,:]
  return uvw[:2,:]

def draw_frame(K: ndarray, T: ndarray, scale=1)->None:
  """
  Visualize the coordinate frame axes of the 4x4 object-to-camera
  matrix T using the 3x3 intrinsic matrix K.

  Control the length of the axes by specifying the scale argument.
  """
  X = T @ np.array(
    [
      [0,scale,0,0],
      [0,0,scale,0],
      [0,0,0,scale],
      [1,1,1,1]
    ]
  )
  u,v = project(K, X)
  plt.plot([u[0], u[1]], [v[0], v[1]], color='red')   # X-axis
  plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
  plt.plot([u[0], u[3]], [v[0], v[3]], color='blue')  # Z-axis

def generate_figure(fig, image_number, K, T, uv, uv_predicted, XY):
  fig.suptitle('Image number %d' % image_number)

  #
  # Visualize reprojected markers and estimated object coordinate frame
  #
  I = plt.imread(os.path.join(sys.path[0],'../data/image%04d.jpg' % image_number))
  plt.subplot(121)
  plt.imshow(I)
  draw_frame(K, T, scale=4.5)
  plt.scatter(uv[0,:], uv[1,:], color='red', label='Detected')
  plt.scatter(uv_predicted[0,:], uv_predicted[1,:], marker='+', color='yellow', label='Predicted')
  plt.legend()
  plt.xlim([0, I.shape[1]])
  plt.ylim([I.shape[0], 0])

  #
  # Visualize scene in 3D
  #
  ax = fig.add_subplot(1, 2, 2, projection='3d')
  ax.plot(XY[0,:], XY[1,:], np.zeros(XY.shape[1]), '.') # Draw markers in 3D
  pO = np.linalg.inv(T) @ np.array([0,0,0,1]) # Compute camera origin
  pX = np.linalg.inv(T) @ np.array([6,0,0,1]) # Compute camera X-axis
  pY = np.linalg.inv(T) @ np.array([0,6,0,1]) # Compute camera Y-axis
  pZ = np.linalg.inv(T) @ np.array([0,0,6,1]) # Compute camera Z-axis
  plt.plot([pO[0], pZ[0]], [pO[1], pZ[1]], [pO[2], pZ[2]], color='blue')  # Draw camera Z-axis
  plt.plot([pO[0], pY[0]], [pO[1], pY[1]], [pO[2], pY[2]], color='green') # Draw camera Y-axis
  plt.plot([pO[0], pX[0]], [pO[1], pX[1]], [pO[2], pX[2]], color='red')   # Draw camera X-axis
  ax.set_xlim([-40, 40])
  ax.set_ylim([-40, 40])
  ax.set_zlim([-25, 25])
  ax.set_xlabel('X')
  ax.set_zlabel('Y')
  ax.set_ylabel('Z')

  plt.tight_layout()
