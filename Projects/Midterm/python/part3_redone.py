# This is a total rework of the original work in part3.py

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plot_all
import common as com

from scipy.optimize import least_squares

from typing import Callable

class ModelA:
  def __init__(
      self,
      K_camera    : np.ndarray,
      T_p_to_c    : np.ndarray,
      # uv          : np.ndarray, 
      detections  : np.ndarray,
      K           : int,
      M           : int,
      N           : int
    ) -> None:
    self.K_camera = K_camera
    self.T_p_to_c = T_p_to_c
    
    # self.uv = uv 
    self.detections = detections

    self.K = K
    self.M = M
    self.N = N

  def residual_function(
      self, 
      x : np.ndarray
    ) -> np.ndarray:
    # The residuals must have dim 2 * M * N
    r = np.zeros((2 * self.M * self.N), dtype=float)

    # Extracting the kinematic parameters
    l1 = x[0]
    l2 = x[1]
    l3 = x[2]
    l4 = x[3]
    l5 = x[4]
   
    marks = x[5:self.K].reshape((3, -1))

    # Iterating over the 'dynamic' parameters
    for i in range(self.N):
      # Why tf would anyone use yaw-pitch-roll instead of roll-pitch-yaw????
      yaw   = x[self.K     + 3 * i]
      pitch = x[self.K + 1 + 3 * i]
      roll  = x[self.K + 2 + 3 * i]

      # Compute the helicopter coordinate frames given the estimated states
      base_to_platform = com.translate(l1 / 2, l1 / 2, 0.00) @ com.rotate_z(yaw) # yaw
      hinge_to_base    = com.translate(0.00, 0.00,  l2) @ com.rotate_y(pitch)
      arm_to_hinge     = com.translate(0.00, 0.00, l3)
      rotors_to_arm    = com.translate(l4, 0.00, l5) @ com.rotate_x(roll) # roll
      base_to_camera   = self.T_p_to_c @ base_to_platform
      hinge_to_camera  = base_to_camera @ hinge_to_base
      arm_to_camera    = hinge_to_camera @ arm_to_hinge
      rotors_to_camera = arm_to_camera @ rotors_to_arm

      # Compute the predicted image location of the marks
      arm_marks = np.vstack((marks[:,:3], np.ones(marks[:,:3].shape[1])))
      rotor_marks = np.vstack((marks[:,3:], np.ones(marks[:,3:].shape[1])))

      p1 = arm_to_camera @ arm_marks
      p2 = rotors_to_camera @ rotor_marks
      uv_hat = com.project(self.K_camera, np.hstack([p1, p2]))

      u = self.detections[i, 1::3]
      v = self.detections[i, 2::3]
      uv_star = np.vstack((u, v))

      w = self.detections[i, ::3]
      u_diff = (w * (uv_hat - uv_star))

      r[2*self.M*i:2*self.M*i + 2*self.M] = np.hstack([u_diff[0], u_diff[1]])

    return r

  def jacobian(
      self
    ) -> np.ndarray:
    # Why would it not allow us to have it outside of the function????????
    # Fuck python!!!!!!
    sparsity_block = np.ones((2 * self.M, 3), dtype=float)
    state_sparsity = np.kron(np.eye(self.N), sparsity_block)
    return np.hstack([np.ones((2 * self.M * self.N, self.K), dtype=float), state_sparsity])

class ModelB:
  def __init__(
      self,
      K_camera    : np.ndarray,
      T_p_to_c    : np.ndarray,
      detections  : np.ndarray,
      K           : int,
      M           : int,
      N           : int
    ) -> None:
    self.K_camera = K_camera
    self.T_p_to_c = T_p_to_c
    
    self.detections = detections

    self.K = K
    self.M = M
    self.N = N

  def residual_function(
      self, 
      x : np.ndarray
    ) -> np.ndarray:
    # The residuals must have dim 2 * M * N
    r = np.zeros((2 * self.M * self.N), dtype=float)

    # Assuming that the parameters are given as
    # x = [ax1, ay1, lx1, ly1, ax2, az2, lx2, lz2, ay3, az3, ly3, lz3, marks, states]
    ax1 = x[0]
    ay1 = x[1]

    lx1 = x[2]
    ly1 = x[3]
    
    ax2 = x[4]
    az2 = x[5]
    
    lx2 = x[6]
    lz2 = x[7]
    
    ay3 = x[8]
    az3 = x[9]
    
    ly3 = x[10]
    lz3 = x[11]
   
    marks = x[12:self.K].reshape((3, -1))

    # Iterating over the 'dynamic' parameters
    for i in range(self.N):
      # Why tf would anyone use yaw-pitch-roll instead of roll-pitch-yaw????
      yaw   = x[self.K     + 3 * i]
      pitch = x[self.K + 1 + 3 * i]
      roll  = x[self.K + 2 + 3 * i]

      T_1_to_platform = com.rotate_x(ax1) @ com.rotate_y(ay1) @ com.translate(lx1, ly1, 0.0) @ com.rotate_z(yaw)
      T_2_to_1        = com.rotate_x(ax2) @ com.rotate_z(az2) @ com.translate(lx2, 0.0, lz2) @ com.rotate_y(pitch)
      T_3_to_2        = com.rotate_y(ay3) @ com.rotate_z(az3) @ com.translate(0.0, ly3, lz3) @ com.rotate_x(roll) 

      T_1_to_camera   = self.T_p_to_c @ T_1_to_platform
      T_2_to_camera   = T_1_to_camera @ T_2_to_1
      T_3_to_camera   = T_2_to_camera @ T_3_to_2

      # Compute the predicted image location of the marks
      arm_marks   = np.vstack((marks[:,:3], np.ones(marks[:,:3].shape[1])))
      rotor_marks = np.vstack((marks[:,3:], np.ones(marks[:,3:].shape[1])))

      p1 = T_2_to_camera @ arm_marks
      p2 = T_3_to_camera @ rotor_marks
      uv_hat = com.project(self.K_camera, np.hstack([p1, p2]))

      u = self.detections[i, 1::3]
      v = self.detections[i, 2::3]
      uv_star = np.vstack((u, v))

      w = self.detections[i, ::3]
      u_diff = (w * (uv_hat - uv_star))

      r[2*self.M*i:2*self.M*i + 2*self.M] = np.hstack([u_diff[0], u_diff[1]])

    return r

  def jacobian(
      self
    ) -> np.ndarray:
    sparsity_block = np.ones((2 * self.M, 3), dtype=float)
    state_sparsity = np.kron(np.eye(self.N), sparsity_block)
    return np.hstack([np.ones((2 * self.M * self.N, self.K), dtype=float), state_sparsity])

class ModelC:
  def __init__(
      self,
      K_camera    : np.ndarray,
      T_p_to_c    : np.ndarray,
      detections  : np.ndarray,
      K           : int,
      M           : int,
      N           : int
    ) -> None:
    self.K_camera = K_camera
    self.T_p_to_c = T_p_to_c
    
    self.detections = detections

    self.K = K
    self.M = M
    self.N = N

  def residual_function(
      self, 
      x : np.ndarray
    ) -> np.ndarray:
    # The residuals must have dim 2 * M * N
    r = np.zeros((2 * self.M * self.N), dtype=float)

    # Assuming that the parameters are given as
    # x = [
    # ax1, ay1, az1, 
    # lx1, ly1, lz1, 
    # ax2, ay2, az2, 
    # lx2, ly2, lz2, 
    # ax3, ay3, az3, 
    # lx3, ly3, lz3, 
    # marks, states
    # ]
    ax1 = x[0]
    ay1 = x[1]
    az1 = x[2]

    lx1 = x[3]
    ly1 = x[4]
    lz1 = x[5]

    ax2 = x[6]
    ay2 = x[7]
    az2 = x[8]
    
    lx2 = x[9]
    ly2 = x[10]
    lz2 = x[11]

    ax3 = x[12]
    ay3 = x[13]
    az3 = x[14]
    
    lx3 = x[15]
    ly3 = x[16]
    lz3 = x[17]    
   
    marks = x[18:self.K].reshape(3, -1)

    # Iterating over the 'dynamic' parameters
    for i in range(self.N):
      # Why tf would anyone use yaw-pitch-roll instead of roll-pitch-yaw????
      yaw   = x[self.K     + 3 * i]
      pitch = x[self.K + 1 + 3 * i]
      roll  = x[self.K + 2 + 3 * i]

      T_1_to_platform = com.rotate_x(ax1) @ com.rotate_y(ay1) @ com.rotate_z(az1) @ com.translate(lx1, ly1, lz1) @ com.rotate_z(yaw)
      T_2_to_1        = com.rotate_x(ax2) @ com.rotate_y(ay2) @ com.rotate_z(az2) @ com.translate(lx2, ly2, lz2) @ com.rotate_y(pitch)
      T_3_to_2        = com.rotate_x(ax3) @ com.rotate_y(ay3) @ com.rotate_z(az3) @ com.translate(lx3, ly3, lz3) @ com.rotate_x(roll) 

      T_1_to_camera   = self.T_p_to_c @ T_1_to_platform
      T_2_to_camera   = T_1_to_camera @ T_2_to_1
      T_3_to_camera   = T_2_to_camera @ T_3_to_2

      # Compute the predicted image location of the marks
      arm_marks = np.vstack((marks[:,:3], np.ones(marks[:,:3].shape[1])))
      rotor_marks = np.vstack((marks[:,3:], np.ones(marks[:,3:].shape[1])))

      p1 = T_2_to_camera @ arm_marks
      p2 = T_3_to_camera @ rotor_marks
      uv_hat = com.project(self.K_camera, np.hstack([p1, p2]))

      u = self.detections[i, 1::3]
      v = self.detections[i, 2::3]
      uv_star = np.vstack((u, v))

      w = self.detections[i, ::3]
      u_diff = (w * (uv_hat - uv_star))

      r[2*self.M*i:2*self.M*i + 2*self.M] = np.hstack([u_diff[0], u_diff[1]])

    return r

  def jacobian(
      self
    ) -> np.ndarray:
    # Tried moving the jacobian out of the class, but that did not work 
    sparsity_block = np.ones((2 * self.M, 3), dtype=float)
    state_sparsity = np.kron(np.eye(self.N), sparsity_block)
    return np.hstack([np.ones((2 * self.M * self.N, self.K), dtype=float), state_sparsity])


def ls_optimize(
      residual  : Callable,
      x0        : np.ndarray,
      jacobian  : np.ndarray,
      x_tol     : float       = 1e-8
  ) -> tuple:
  # Python is so fucking weird!
  # When having this inside a function, it somehow works
  # Moving it outside of the function, somehow does not work!
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
# uv = np.vstack((all_detections[:, 1::3], all_detections[:, 2::3]))
heli_points = np.loadtxt(os.path.join(sys.path[0], '../data/data/heli_points.txt')).T

heli_points = heli_points[:3,:]
temp = np.zeros(21)
for col in range(7):
  cols = heli_points[:,col]
  temp[3*col:3*(col+1)] = cols.T
markers = temp.copy()

if __name__ == "__main__":
  # Choosing the model
  chosen_model = 'B'

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
    x0[5:K] = markers.reshape((1,-1)) #np.ones(K - 5, dtype=float)
    x0[K:] = 0.5 * np.ones(3 * N, dtype=float)

    model = ModelA(
      K_camera=K_camera,
      T_p_to_c=T_p_to_c,
      # uv=uv,
      detections=all_detections,
      K=K,
      M=M,
      N=N
    )

  elif chosen_model == 'B':
    # For model B
    M = 7
    K = 33
    N = 351
    x_tol = 1e-2

    # From model A:
    l = [0.1145, 0.325, -0.05, 0.65, -0.03]
    # Thought that the parameters are combined, and the initial values are somewhat
    # close to the original arm and rotor frame. This means is is assumed lx1 = l1 = ly1, 

    # Assuming that the parameters are given as
    # x = [
    #   ax1, ay1, 
    #   lx1, ly1, 
    #   ax2, az2, 
    #   lx2, lz2, 
    #   ay3, az3, 
    #   ly3, lz3, 
    #   marks, states
    # ]

    x0 = np.zeros((K + 3 * N), dtype=float)
    x0[:12] = np.array(
      [
        0.0, 0.0, 
        l[0] / 2.0 , l[0] / 2.0, 
        0.0, 0.0, 
        0.0, l[1] + l[2], 
        0.0, 0.0, 
        0.0, l[4]
      ]
    ).reshape((1,-1))
    # x0[:12] = x0[:12] * np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])   
    x0[12:K] = markers.reshape((1,-1)) #np.ones(K - 12, dtype=float)
    x0[K:] = 0.5 * np.ones(3 * N, dtype=float)

    # print(x0[:12])
    # quit()

    model = ModelB(
      K_camera=K_camera,
      T_p_to_c=T_p_to_c,
      detections=all_detections,
      K=K,
      M=M,
      N=N
    )

  elif chosen_model == 'C':
    # For model C
    M = 7
    K = 39
    N = 351
    x_tol = 1e-2

    # From model A:
    l = [0.1145, 0.325, -0.05, 0.65, -0.03]
    # Thought that the parameters are combined, and the initial values are somewhat
    # close to the original arm and rotor frame. This means is is assumed lx1 = l1 = ly1, 

    # Assuming that the parameters are given as
    # x = [
    # ax1, ay1, az1, 
    # lx1, ly1, lz1, 
    # ax2, ay2, az2, 
    # lx2, ly2, lz2, 
    # ax3, ay3, az3, 
    # lx3, ly3, lz3, 
    # marks, states
    # ]

    x0 = np.zeros((K + 3 * N), dtype=float)
    # x0[:18] = np.array(
    #   [0.0, 0.0, l[0], l[0], 0.0, 0.0, l[3], l[1] + l[2], 0.0, 0.0, 0.5, l[4]]
    # ).reshape((1,-1))
    x0[:18] = np.array(
        [
          0.0, 0.0, 0.0, 
          l[0] / 2.0, l[0] / 2.0, 0.0, 
          0.0, 0.0, 0.0, 
          0.0, 0.0, l[1] + l[2], 
          0.0, 0.0, 0.0, 
          l[3], 0.0, l[4]
        ]).reshape((1,-1))   
    x0[18:K] = markers.reshape((1,-1)) #0.05 * np.ones(K - 18, dtype=float)
    x0[K:] = 0.5 * np.ones(3 * N, dtype=float)

    # print(x0[:12])
    # quit()

    model = ModelC(
      K_camera=K_camera,
      T_p_to_c=T_p_to_c,
      detections=all_detections,
      K=K,
      M=M,
      N=N
    )

  else:
    print("Invalid chosen model")
    quit()

  jacobian = model.jacobian()
  residual = model.residual_function
  optimization_results = ls_optimize(
    residual=residual,
    x0=x0, 
    jacobian=jacobian,
    x_tol=x_tol
  )
  x = optimization_results.x

  if chosen_model == 'A':
    print(x[:5])
  elif chosen_model == 'B':
    print(x[:12])
  elif chosen_model == 'C':
    print(x[:18])

  all_r = []
  all_x = []
  # kinematic_states = x[:K]
  for i in range(N):
    states = x[K + 3 * i : K + 3 * (i + 1)]
    all_x.append(states)
    # all_r.append(residual(np.hstack([kinematic_states, states])))
  all_x = np.array(all_x)
  # all_r = np.array(all_r).reshape((N, 2 * M))
  all_r = residual(x).reshape((N, 2 * M))
  np.savetxt('all_x.txt', all_x)
  np.savetxt('all_r.txt', all_r)
  # all_r = np.zeros((N, 2 * M)) # Fix plz!

  plot_all.plot_all(all_x, all_r, all_detections, subtract_initial_offset=True)
  # plt.savefig('out_part3.png')
  plt.show()
