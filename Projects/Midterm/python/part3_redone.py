# This is a total rework of the original work in part3.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plot_all
import common as com

from scipy.optimize import least_squares
from scipy.linalg import block_diag
from quanser import Quanser

from typing import Callable

class ModelA:
  def __init__(
      self,
      K_camera  : np.ndarray,
      T_p_to_c  : np.ndarray,
      uv        : np.ndarray, 
      weights   : np.ndarray,
      K         : int,
      M         : int,
      N         : int
    ) -> None:
    self.K_camera = K_camera
    self.T_p_to_c = T_p_to_c
    
    self.uv = uv 
    self.weights = weights

    self.K = K
    self.M = M
    self.N = N

    self.detections = np.loadtxt(os.path.join(sys.path[0], "../data/data/detections.txt"))


  def residual_function(
      self, 
      x : np.ndarray
    ) -> np.ndarray:
    # The residuals must have dim 2 * M * N
    r = np.zeros((2 * self.M * self.N), dtype=float)
    # return r

    # print(self.weights[0, :])
    # print(self.weights[0, ::3])
    # quit()

    # Extracting the static parameters
    l1 = x[0]
    l2 = x[1]
    l3 = x[2]
    l4 = x[3]
    l5 = x[4]
   
    marks = x[5:self.K].reshape(3, -1)
    marks_1 = np.vstack((marks, np.ones((1, self.M))))

    # Iterating over the 'dynamic' parameters
    for i in range(self.N):
      roll = x[self.K + 3 * i]
      pitch = x[self.K + 1 + 3 * i]
      yaw = x[self.K + 2 + 3 * i]

      # Compute the helicopter coordinate frames given the estimated states
      base_to_platform = com.translate(l1 / 2, l1 / 2, 0.00) @ com.rotate_z(roll) # yaw
      hinge_to_base    = com.translate(0.00, 0.00,  l2) @ com.rotate_y(pitch)
      arm_to_hinge     = com.translate(0.00, 0.00, l3)
      rotors_to_arm    = com.translate(l4, 0.00, l5) @ com.rotate_x(yaw) # roll
      base_to_camera   = self.T_p_to_c @ base_to_platform
      hinge_to_camera  = base_to_camera @ hinge_to_base
      arm_to_camera    = hinge_to_camera @ arm_to_hinge
      rotors_to_camera = arm_to_camera @ rotors_to_arm

      # Compute the predicted image location of the marks
      arm_marks = marks_1[:,:3]#np.vstack((marks[:,:3], np.ones(marks[:,:3].shape[1])))
      rotor_marks = marks_1[:,3:]#np.vstack((marks[:,3:], np.ones(marks[:,3:].shape[1])))

      p1 = arm_to_camera @ arm_marks
      p2 = rotors_to_camera @ rotor_marks
      uv_hat = com.project(self.K_camera, np.hstack([p1, p2]))

      ui = self.detections[i, 1::3]
      vi = self.detections[i, 2::3]
      uv_star = np.vstack((ui, vi))

      w = self.detections[i, ::3]
      # w = self.weights[i, :]  # Are the weights correct? 
      # u_diff = uv_hat - self.uv[i:i+1,:] # This is incorrect! u_diff should be of size(2,7) and not size(1,7) 

      # print(self.uv[i:i+1,:])
      # print(u_diff)
      # quit()
      u_diff = (w * (uv_hat - uv_star)).flatten()

      # update_step = w * u_diff

      r[14 * i : 14 * (i + 1)] = u_diff

    return r

  def jacobian(
      self
    ) -> np.ndarray:
    sparsity_block = np.ones((2 * self.M, 3), dtype=float)
    state_sparsity = np.kron(np.eye(self.N), sparsity_block)
    return np.hstack([np.ones((2 * self.M * self.N, self.K), dtype=float), state_sparsity])

def ls_optimize(
      residual  : Callable,
      x0        : np.ndarray,
      jacobian  : np.ndarray,
      x_tol     : float       = 1e-8
  ) -> tuple:
  return least_squares(
    residual,
    x0=x0,
    jac_sparsity=jacobian,
    xtol=x_tol
  )


# Loading data from the files
all_detections = np.loadtxt(os.path.join(sys.path[0], '../data/data/detections.txt'))
T_p_to_c = np.loadtxt(os.path.join(sys.path[0], "../data/data/platform_to_camera.txt"))
K_camera = np.loadtxt(os.path.join(sys.path[0], "../data/data/K.txt"))
uv = np.vstack((all_detections[:, 1::3], all_detections[:, 2::3]))
weights = all_detections[:, ::3]

if __name__ == "__main__":
  # Choosing the model
  chosen_model = 'A'

  if chosen_model == 'A':
    # For model A

    M = 7
    K = 26
    N = 351
    x_tol = 1e-2

    # Setting the initial parameters, under the assumption of the lengths being 
    # approzimately known
    x0 = np.zeros((K + 3 * N), dtype=float)
    x0[:5] = np.array([0.1145, 0.325, -0.05, 0.65, -0.03]).reshape((1,-1))
    x0[5:K] = np.ones(K - 5, dtype=float)
    x0[K:] = 0.25 * np.ones(3 * N, dtype=float)

    model = ModelA(
      K_camera=K_camera,
      T_p_to_c=T_p_to_c,
      uv=uv,
      weights=weights,
      K=K,
      M=M,
      N=N
    )

  jacobian = model.jacobian()
  residual = model.residual_function
  optimization_results = ls_optimize(
    residual=residual,
    x0=x0, 
    jacobian=jacobian,
    x_tol=x_tol
  )
  x = optimization_results.x

  all_r = []
  all_x = []
  # for image_number in range(N):
  all_r = []
  all_p = []
  for image_number in range(N):
    rpy = x[
      K + 3 * image_number : K + 3 * (image_number + 1)
    ]
    all_p.append(rpy)
    all_r.append(residual(x))
  all_p = np.array(all_p)
  all_r = np.zeros((N, 2 * M))

  # all_r.append((residual(x)))
  # all_x.append(x)

  # all_p = np.array(all_x)
  # all_r = np.array(all_r)

  #   # rpy = optimized_parameters[
  #   #   KP + 3 * image_number : KP + 3 * (image_number + 1)
  #   # ]
  #   # all_p.append(rpy)
  #   # all_r.append(optimizer.residuals(optimized_parameters))
  # all_p = np.array(all_p)
  # all_r = np.zeros((N, 2 * M))

  plot_all.plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
  # plt.savefig('out_part3.png')
  plt.show()
