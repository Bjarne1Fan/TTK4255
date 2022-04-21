import numpy as np

def project(
      arr : np.ndarray, 
      K   : np.ndarray
    ) -> np.ndarray:

  arr_tilde = K @ arr
  return arr_tilde[:2,:] / arr_tilde[2,:]
