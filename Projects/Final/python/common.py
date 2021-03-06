import numpy as np

def project(
      arr   : np.ndarray, 
      K     : np.ndarray
    ) -> np.ndarray:
  arr_tilde = K @ arr
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
  """
  Using the solution from assignment 5
  """
  l2 = F @ uv1
  l1 = F.T @ uv2
  
  e = np.sum(uv2*l2, axis=0)
  
  norm1 = np.linalg.norm(l1[:2,:], axis=0)
  norm2 = np.linalg.norm(l2[:2,:], axis=0)
  
  return 0.5 * e * (1 / norm1 + 1 / norm2)


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
  for i in range(n):
    # Creating A-matrix and solving it using SVD
    A = np.array(
      [
        xy1[0, i] * P1[2,:] - P1[0,:], 
        xy1[1, i] * P1[2,:] - P1[1,:], 
        xy2[0, i] * P2[2,:] - P2[0,:], 
        xy2[1, i] * P2[2,:] - P2[1,:]
      ]
    )
    _, _, V_T = np.linalg.svd(A)
    X[:, i] = V_T[-1]

  # Normalize the matrix
  return X / X[-1]


def find_optimal_pose(
      pose_matrices : list[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
      pose_world    : np.ndarray,
      xy1           : np.ndarray,
      xy2           : np.ndarray
    ) -> np.ndarray:
  """
  Calculates the pose with the most camera coordinates in 
  front of the camera (Z > 0)

  Input: 
    pose_matrices : List of 4 pose matrices
    pose_world    : Pose of the world frame, camera 1
    xy1           : Calibrated camera coordinates for camera 1
    xy2           : Calibrated camera coordinates for camera 2

  Output:
    optimal_pose_matrix : Pose with most coordinates in front of the camera 
  """

  assert isinstance(pose_matrices, list), "Input must be a list"
  assert isinstance(pose_matrices[0], np.ndarray), "Input must be a list of ndarrays"
  
  pos_z = 0
  best_pose_idx = 0

  # Iterate over all of the possible matrices and find the matrix with most
  # measurements in front of the camera
  for (idx, pose) in enumerate(pose_matrices):
    X = triangulate_many(
      xy1=xy1, 
      xy2=xy2, 
      P1=pose_world, 
      P2=pose
    )
    
    # Find maximum points with a positive z-value (in front of the camera)
    num_pos_z = np.sum(((pose @ X)[2] >= 0))
    if num_pos_z >= pos_z:
      pos_z = num_pos_z
      best_pose_idx = idx

  return pose_matrices[best_pose_idx]


