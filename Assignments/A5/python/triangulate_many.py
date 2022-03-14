import numpy as np

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
