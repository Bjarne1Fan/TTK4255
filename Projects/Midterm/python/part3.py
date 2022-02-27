import os
import sys
from turtle import shape

import matplotlib.pyplot as plt
import numpy as np
from prometheus_client import Counter
import plot_all
import common as com

from scipy.optimize import least_squares, minimize
from quanser import Quanser

from numpy import character, ndarray

counter = 0

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

    r = np.zeros((2,7)) # residual function stats at 0

    marks = x[:21].reshape(3, -1)

    l1 = x[0]
    l2 = x[1]
    l3 = x[2]
    l4 = x[3]
    l5 = x[4]
   
    # x[5:25] = ?

    for i in range(351):

      roll = x[26 + 3*i]
      pitch = x[27 + 3*i]
      yaw = x[28 +3*i]

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
      w = weights[i, :]
      u_diff = uv_hat - uv[i:i+1,:]

      update_step = u_diff * w

      r = r + update_step
    r = np.hstack([r[0], r[1]])

    global counter 
    counter += 1
    if (counter % 100) == 0:
      print(counter)

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



N = 351
Kin = 26
M = 7

method = 'A'
if method == 'A':
  x = np.zeros((Kin+3*N))
  x[:5] = np.array([0.1145, 0.325, -0.05, 0.65, -0.03]) 

elif method == 'B':
  x = np.zeros(33)
elif method == 'C':
  x = np.zeros(39)
else:
  raise "Invalid option" 


all_r = []
all_x = []
#for i in range(len(all_detections)):
weights = all_detections[:, ::3]
uv = np.vstack((all_detections[:, 1::3], all_detections[:, 2::3]))

sparsity_block = np.ones(( 2*M,3 ))
state_sparsity =  np.kron(np.eye(N), sparsity_block)

jac_sparsity = np.hstack( [ np.ones((2*M*N, Kin)), state_sparsity ] )

resfun = lambda x : residual_function(x, uv, weights, K, heli_points, T_p_to_c, method)

  # But I cannot see how one should make this into a batch-optimization problem...
x = least_squares(resfun, x0=x, jac_sparsity=jac_sparsity).x
  # x = minimize(resfun, x0=x).x

  #print(resfun(x))

all_r.append(resfun(x))
all_x.append(x)

all_p = np.array(all_x)
all_r = np.array(all_r)
# Tip: See comment in plot_all.py regarding the last argument.
plot_all.plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
# plt.savefig('out_part1b.png')
plt.show()

"""
K??????????
How can you calculated resedual function for K?

What should be optimized? roll, pitch, yaw or marker positions?

Task 2.3: How are we supposed to show difference, not clear what the task asks
Task 2.3: How big difference is expected? 

"""