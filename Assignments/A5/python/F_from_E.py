import numpy as np

def F_from_E( 
      E : np.ndarray, 
      K : np.ndarray
    ) -> np.ndarray:
  K_inv = np.linalg.inv(K)
  F = K_inv.T @ E @ K_inv
  return F
