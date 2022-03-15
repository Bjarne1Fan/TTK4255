import numpy as np
import estimate_E as est_E
import epipolar_distance as epi_dist
import F_from_E 

def estimate_E_ransac(
      xy1                 : np.ndarray, 
      xy2                 : np.ndarray, 
      K                   : np.ndarray, 
      distance_threshold  : float, 
      num_trials          : int
    ) -> np.ndarray:

  # Tip: The following snippet extracts a random subset of 8
  # correspondences (w/o replacement) and estimates E using them.
  #   sample = np.random.choice(xy1.shape[1], size=8, replace=False)
  #   E = estimate_E(xy1[:,sample], xy2[:,sample])

  # Defining memory
  E = np.eye(3)
  inlier_set = {}

  for i in range(num_trials):
    return

  pass 
