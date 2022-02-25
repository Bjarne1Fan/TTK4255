import os 
import sys

import matplotlib.pyplot as plt
import numpy as np
import plot_all
from scipy.optimize import least_squares
from quanser import Quanser

from numpy import ndarray

all_detections = np.loadtxt(os.path.join(sys.path[0], '../data/data/detections.txt'))
quanser = Quanser()

# Batch residual function
def residual_function(
      x       : ndarray,
      uv      : ndarray,
      weights : ndarray,
      method  : int = 0
    )->np.ndarray:
  assert method >= 0 and method <= 2, "Desired method out of range"
  assert isinstance(method, int), "Method could only be passed as an integer"

  def model_A():

    return np.zeros((1, 26))

  def model_B():
    return np.zeros((1, 33))

  def model_C():
    return np.zeros((1, 39))

  if method == 0:
    model = model_A() 
  elif method == 1:
    model = model_B()
  else:
    model = model_C
  
  return model

p = np.array([0.0, 0.0, 0.0])
all_r = []
all_p = []
for i in range(len(all_detections)):
  weights = all_detections[i, ::3]
  uv = np.vstack((all_detections[i, 1::3], all_detections[i, 2::3]))

  resfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])

  p = least_squares(resfun, x0=p, method='lm').x

  all_r.append(resfun(p))
  all_p.append(p)

all_p = np.array(all_p)
all_r = np.array(all_r)
# Tip: See comment in plot_all.py regarding the last argument.
plot_all.plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
# plt.savefig('out_part1b.png')
plt.show()
