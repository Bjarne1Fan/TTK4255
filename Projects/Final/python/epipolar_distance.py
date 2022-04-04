import numpy as np

def epipolar_distance(
      F   : np.ndarray, 
      uv1 : np.ndarray, 
      uv2 : np.ndarray
    ) -> np.ndarray:
  """
  F should be the fundamental matrix (use F_from_E)
  uv1, uv2 should be 3 x n homogeneous pixel coordinates
  """
  n = uv1.shape[1]
  e = np.zeros((2, n))

  for pt in range(n):
    u1 = uv1[:, pt]
    u2 = uv2[:, pt]

    e1 = (u1.T @ F.T @ u2) / np.linalg.norm(F.T @ u2)
    e2 = (u2.T @ F @ u1) / np.linalg.norm(F @ u1)

    e[:, pt] = np.hstack([e1, e2])

  return e