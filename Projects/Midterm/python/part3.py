import os 
import sys

import matplotlib.pyplot as plt
import numpy as np
import plot_all
import common as com

from scipy.optimize import least_squares, minimize
from quanser import Quanser

from numpy import character, ndarray

all_detections = np.loadtxt(os.path.join(sys.path[0], '../data/data/detections.txt'))
quanser = Quanser()

# Batch residual function
def residual_function(
      x           : ndarray,
      uv          : ndarray,
      weights     : ndarray,
      K           : ndarray,
      heli_points : ndarray,
      T_p_to_c    : ndarray,
      method      : int = 0
    )->np.ndarray:
  assert method >= 'A' and method <= 'C', "Desired method out of range"
  assert isinstance(method, str), "Method could only be passed as a string"

  def residual_A(      
      x           : ndarray,
      uv          : ndarray,
      weights     : ndarray,
      K           : ndarray,
      T_p_to_c    : ndarray
    )->ndarray:
    # Extract the values from the optimization algorithm
    marks = x[:21].reshape(3, -1)

    l1 = x[21]
    l2 = x[22]
    l3 = x[23]
    l4 = x[24]
    l5 = x[25]

    # Only using the latest value for optimization
    roll = x[-3]
    pitch = x[-2]
    yaw = x[-1]

    # Compute the helicopter coordinate frames given the estimated states
    base_to_platform = com.translate(l1 / 2, l1 / 2, 0.00) @ com.rotate_z(yaw)
    hinge_to_base    = com.translate(0.00, 0.00,  l2) @ com.rotate_y(pitch)
    arm_to_hinge     = com.translate(0.00, 0.00, l3)
    rotors_to_arm    = com.translate(l4, 0.00, l5) @ com.rotate_x(roll)
    base_to_camera   = T_p_to_c @ base_to_platform
    hinge_to_camera  = base_to_camera @ hinge_to_base
    arm_to_camera    = hinge_to_camera @ arm_to_hinge
    rotors_to_camera = arm_to_camera @ rotors_to_arm

    # Compute the predicted image location of the marks
    # The first three marks corresponds to the arm, while the last four corresponds to the rotor
    # Must also have it such that the marks have dimension of 4x21
    arm_marks = np.vstack((marks[:,:3], np.ones(marks[:,:3].shape[1])))
    rotor_marks = np.vstack((marks[:,3:], np.ones(marks[:,3:].shape[1])))

    p1 = arm_to_camera @ arm_marks
    p2 = rotors_to_camera @ rotor_marks
    uv_hat = com.project(K, np.hstack([p1, p2]))

    # The residual-function r could only maintain the values for the optimized coordinates
    # This means that it could only

    r = (uv_hat - uv) * weights
    r = np.hstack([r[0], r[1]])
    return r
    # return np.zeros((1, 26))

  def residual_B(      
      x           : ndarray,
      uv          : ndarray,
      weights     : ndarray,
      detections  : ndarray,
      iteration   : int
    )->ndarray:
    return np.zeros((1, 33))

  def residual_C(      
      x           : ndarray,
      uv          : ndarray,
      weights     : ndarray,
      detections  : ndarray,
      iteration   : int
    )->ndarray:
    return np.zeros((1, 39))

  if method == 'A':
    return residual_A(x, uv, weights, K, T_p_to_c) 
  # elif method == 'B':
  #   return residual_B(x, uv, weights, detections, iteration)
  # else:
  #   return residual_C(x, uv, weights, detections, iteration)
  

K = np.loadtxt(os.path.join(sys.path[0], '../data/data/K.txt'))
heli_points = np.loadtxt(os.path.join(sys.path[0], '../data/data/heli_points.txt')).T
T_p_to_c = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_to_camera.txt'))

method = 'A'
if method == 'A':
  x = np.zeros(26)
elif method == 'B':
  x = np.zeros(33)
elif method == 'C':
  x = np.zeros(39)
else:
  raise "Invalid option" 

all_r = []
all_x = []
for i in range(len(all_detections)):
  weights = all_detections[i, ::3]
  uv = np.vstack((all_detections[i, 1::3], all_detections[i, 2::3]))

  # resfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])
  resfun = lambda x : residual_function(x, uv, weights, K, heli_points, T_p_to_c, method)

  # But I cannot see how one should make this into a batch-optimization problem...
  x = least_squares(resfun, x0=x).x
  # x = minimize(resfun, x0=x).x

  all_r.append(resfun(x))
  all_x.append(x)

all_p = np.array(all_x)
all_r = np.array(all_r)
# Tip: See comment in plot_all.py regarding the last argument.
plot_all.plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
# plt.savefig('out_part1b.png')
plt.show()
