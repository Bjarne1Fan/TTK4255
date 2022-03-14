import numpy as np

def project(
      arr   : np.ndarray, 
      K_inv : np.ndarray
    ) -> np.ndarray:

  arr_tilde = K_inv @ arr
  return arr_tilde[:2,:] / arr_tilde[2,:]
