import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray

# def rotate_x(radians): Rotation about X-axis
# def rotate_y(radians): Rotation about Y-axis
# def rotate_z(radians): Rotation about Z-axis

def translate(x: float, y: float, z: float)->ndarray:
  return np.array(
    [
      [1, 0, 0, x],
      [0, 1, 0, y],
      [0, 0, 1, z],
      [0, 0, 0, 1]
    ]
  )

# def rotate_x(theta: float)->ndarray:
#   return np.array(
#     [
#       [1, 0, 0],
#       [0, np.cos(theta), -np.sin(theta)],
#       [0, np.sin(theta), np.cos(theta)]
#     ]
#   )

# def rotate_y(theta: float)->ndarray:
#   return np.array(
#     [
#       [np.cos(theta), 0, -np.sin(theta)],
#       [0, 1, 0],
#       [np.sin(theta), 0, np.cos(theta)]
#     ]
#   )

# def rotate_z(theta: float)->ndarray:
#   return np.array(
#     [
#       [np.cos(theta), -np.sin(theta), 0],
#       [np.sin(theta), np.cos(theta), 0],
#       [0, 0, 1]
#     ]
#   )

def rotate_x(degrees):
  phi = np.deg2rad(degrees)
  c = np.cos(phi)
  s = np.sin(phi)
  return np.array(
    [
      [1, 0, 0, 0],
      [0, c, -s, 0],
      [0, s, c, 0],
      [0, 0, 0, 1]
    ]
  )

# Rotation about Y-axis
def rotate_y(degrees): 
  theta = np.deg2rad(degrees)
  c = np.cos(theta)
  s = np.sin(theta)
  return np.array(
    [
      [c, 0, s, 0],
      [0, 1, 0, 0],
      [-s, 0, c, 0],
      [0, 0, 0, 1]
    ]
  )

# Rotation about Z-axis
def rotate_z(degrees): 
  psi = np.deg2rad(degrees)
  c = np.cos(psi)
  s = np.sin(psi)
  return np.array(
    [
      [c, -s, 0, 0],
      [s, c, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ]
  )


def project(K: ndarray, X: ndarray)->ndarray:
  """
  Computes the pinhole projection of a 3xN array of 3D points X
  using the camera intrinsic matrix K. Returns the dehomogenized
  pixel coordinates as an array of size 2xN.
  """
  # u_tilde = K @ X
  # if is_dim_geq4:
  #   # Task 3
  #   # Extract the first three rows
  #   # K = K[:3, :3]
  #   # X = X[:3, :3]
  #   u_tilde = u_tilde[:3, :]
  if X.shape[0] == 3:
    u_tilde = K @ X
  elif X.shape[0] == 4:
    u_tilde = K @ X[:3,:]

  u_hom = u_tilde / u_tilde[-1]
  uv = np.array(
    [
      [u_hom[0]], 
      [u_hom[1]]
    ]
  )

  return uv

def draw_frame(K, T, scale=1):
  """
  Visualize the coordinate frame axes of the 4x4 object-to-camera
  matrix T using the 3x3 intrinsic matrix K.

  This uses your project function, so implement it first.

  Control the length of the axes using 'scale'.
  """
  X = T @ np.array(
    [
      [0,scale,0,0],
      [0,0,scale,0],
      [0,0,0,scale],
      [1,1,1,1]
    ])
  u,v = project(K, X) # If you get an error message here, you should modify your project function to accept 4xN arrays of homogeneous vectors, instead of 3xN.
  plt.plot([u[0], u[1]], [v[0], v[1]], color='red') # X-axis
  plt.plot([u[0], u[2]], [v[0], v[2]], color='green') # Y-axis
  plt.plot([u[0], u[3]], [v[0], v[3]], color='blue') # Z-axis
