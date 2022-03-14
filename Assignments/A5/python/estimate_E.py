import numpy as np

def estimate_E(
      xy1 : np.ndarray, 
      xy2 : np.ndarray
    ) -> np.ndarray:
  
  n = xy1.shape[1]
  assert n > 8, "Not enough correspondances to estimate E. \
    Use 5-point or another algorithm instead!"

  # Calculating A
  A = np.zeros((n, 9))
  for row in range(n):
    # Assuming xy1 to come from the left image, and xy2 to be the right
    A[row] = np.array(
      [
        xy2[0, row] * xy1[0, row], xy2[0, row] * xy1[1, row], xy2[0, row], \
        xy2[1, row] * xy1[0, row], xy2[1, row] * xy1[1, row], xy2[1, row], \
        xy1[0, row], xy1[1, row], 1
      ]
    )
    
  # Using SVD to decompose into E
  _, _, V_T = np.linalg.svd(A) 
  return V_T[-1].reshape((3, 3))

