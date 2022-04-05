import numpy as np

def project(
      arr   : np.ndarray, 
      K_inv : np.ndarray
    ) -> np.ndarray:

  arr_tilde = K_inv @ arr
  return arr_tilde[:2,:] / arr_tilde[2,:]


def closest_rotation_matrix(Q: np.ndarray) -> np.ndarray:
  U, _, VT = np.linalg.svd(Q)
  R = U @ VT
  return R


def epipolar_distance(
      F   : np.ndarray, 
      uv1 : np.ndarray, 
      uv2 : np.ndarray
    ) -> np.ndarray:
  n = uv1.shape[1]
  e = np.zeros((2, n))

  for pt in range(n):
    u1 = uv1[:, pt]
    u2 = uv2[:, pt]

    e1 = (u1.T @ F.T @ u2) / np.linalg.norm(F.T @ u2)
    e2 = (u2.T @ F @ u1) / np.linalg.norm(F @ u1)

    e[:, pt] = np.hstack([e1, e2])

  return e


def __SE3(
      R : np.ndarray,
      t : np.ndarray
    ) -> np.ndarray:
  T = np.eye(4)
  T[:3,:3] = R
  T[:3,3] = t
  return T


def decompose_E(E : np.ndarray) -> np.ndarray:
    """
    Computes the four possible decompositions of E into a relative
    pose, as described in Szeliski 7.2.

    Returns a list of 4x4 transformation matrices.
    """
    U, _, VT = np.linalg.svd(E)
    R90 = np.array(
      [
        [0, -1, 0], 
        [+1, 0, 0], 
        [0, 0, +1]
      ]
    )
    R1 = U @ R90 @ VT
    R2 = U @ R90.T @ VT
    if np.linalg.det(R1) < 0: R1 = -R1
    if np.linalg.det(R2) < 0: R2 = -R2
    t1, t2 = U[:,2], -U[:,2]
    return [__SE3(R1,t1), __SE3(R1,t2), __SE3(R2, t1), __SE3(R2, t2)]


def F_from_E( 
      E : np.ndarray, 
      K : np.ndarray
    ) -> np.ndarray:
  K_inv = np.linalg.inv(K)
  F = K_inv.T @ E @ K_inv
  return F


def triangulate_many(
      xy1 : np.ndarray, 
      xy2 : np.ndarray, 
      P1  : np.ndarray, 
      P2  : np.ndarray
    ) -> np.ndarray:
  """
  Arguments
      xy: Calibrated image coordinates in image 1 and 2
          [shape 3 x n]
      P:  Projection matrix for image 1 and 2
          [shape 3 x 4]
  Returns
      X:  Dehomogenized 3D points in world frame
          [shape 4 x n]
  """

  # Memory allocation
  n = xy1.shape[1]
  X = np.zeros((4, n))

  # Iterating over all points
  for ptn in range(n):
    # Creating A-matrix and solving it using SVD
    A = np.array(
      [
        xy1[0, ptn] * P1[2,:] - P1[0,:], 
        xy1[1, ptn] * P1[2,:] - P1[1,:], 
        xy2[0, ptn] * P2[2,:] - P2[0,:], 
        xy2[1, ptn] * P2[2,:] - P2[1,:]
      ]
    )
    _, _, V_T = np.linalg.svd(A)
    X[:, ptn] = V_T[-1]

  # Normalize the matrix
  return X / X[-1]


def find_optimal_pose(
      pose_matrices : list,
      pose_world    : np.ndarray,
      xy1           : np.ndarray,
      xy2           : np.ndarray
    ) -> np.ndarray:

  assert isinstance(pose_matrices, list), "Input must be a list"
  assert isinstance(pose_matrices[0], np.ndarray), "Input must be a list of ndarrays"
  
  pos_z = 0
  best_pose_idx = 0

  # Iterate over all of the possible matrices and find the matrix with most
  # measurements in front of the camera
  for (idx, P) in enumerate(pose_matrices):
    X = triangulate_many(
      xy1=xy1, 
      xy2=xy2, 
      P1=pose_world, 
      P2=P
    )
    
    # Find maximum points with a positive z-value (in front of the camera)
    num_pos_z = np.sum(((P @ X)[2] >= 0))
    if num_pos_z >= pos_z:
      pos_z = num_pos_z
      best_pose_idx = idx

  return pose_matrices[best_pose_idx]


