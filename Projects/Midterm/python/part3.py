import os
import sys
from turtle import shape

import matplotlib.pyplot as plt
import numpy as np
import plot_all
import common as com

from scipy.optimize import least_squares, minimize
from quanser import Quanser

from scipy.linalg import block_diag

from numpy import character, ndarray

counter = 0

all_detections = np.loadtxt(os.path.join(sys.path[0], '../data/data/detections.txt'))
quanser = Quanser()

# # Batch residual function
# def residual_function(
#       x           : ndarray,
#       uv          : ndarray,
#       weights     : ndarray,
#       K           : ndarray,
#       heli_points : ndarray,
#       T_p_to_c    : ndarray,
#       method      : int = 0
#     )->np.ndarray:
#   assert method >= 'A' and method <= 'C', "Desired method out of range"
#   assert isinstance(method, str), "Method could only be passed as a string"

#   def residual_A(      
#       x           : ndarray,
#       uv          : ndarray,
#       weights     : ndarray,
#       K           : ndarray,
#       T_p_to_c    : ndarray
#     )->ndarray:
#     # Extract the values from the optimization algorithm

#     # r = np.zeros((2,7)) # residual function stats at 0
#     # The residuals must have dim 2 * M * N
#     r = np.zeros((2 * 7 * 351))

#     marks = x[:21].reshape(3, -1)

#     l1 = x[0]
#     l2 = x[1]
#     l3 = x[2]
#     l4 = x[3]
#     l5 = x[4]
   
#     # x[5:25] = ?
#     for i in range(351):
#       roll = x[26 + 3*i]
#       pitch = x[27 + 3*i]
#       yaw = x[28 +3*i]

#       # Compute the helicopter coordinate frames given the estimated states
#       base_to_platform = com.translate(l1 / 2, l1 / 2, 0.00) @ com.rotate_z(yaw)
#       hinge_to_base    = com.translate(0.00, 0.00,  l2) @ com.rotate_y(pitch)
#       arm_to_hinge     = com.translate(0.00, 0.00, l3)
#       rotors_to_arm    = com.translate(l4, 0.00, l5) @ com.rotate_x(roll)
#       base_to_camera   = T_p_to_c @ base_to_platform
#       hinge_to_camera  = base_to_camera @ hinge_to_base
#       arm_to_camera    = hinge_to_camera @ arm_to_hinge
#       rotors_to_camera = arm_to_camera @ rotors_to_arm

#       # Compute the predicted image location of the marks
#       # The first three marks corresponds to the arm, while the last four corresponds to the rotor
#       # Must also have it such that the marks have dimension of 4x21
#       arm_marks = np.vstack((marks[:,:3], np.ones(marks[:,:3].shape[1])))
#       rotor_marks = np.vstack((marks[:,3:], np.ones(marks[:,3:].shape[1])))

#       p1 = arm_to_camera @ arm_marks
#       p2 = rotors_to_camera @ rotor_marks
#       uv_hat = com.project(K, np.hstack([p1, p2]))

#       # The residual-function r could only maintain the values for the optimized coordinates
#       # This means that it could only
#       w = weights[i, :]
#       u_diff = uv_hat - uv[i:i+1,:]

#       update_step = u_diff * w

#       r[14 * i : 14 * i + 14] = np.hstack([update_step[0], update_step[1]])
#     # r = np.hstack([r[0], r[1]])

#     global counter 
#     counter += 1
#     if (counter % 100) == 0:
#       print(counter)

#     return r.T
#     # return np.zeros((1, 26))

#   def residual_B(      
#       x           : ndarray,
#       uv          : ndarray,
#       weights     : ndarray,
#       detections  : ndarray,
#       iteration   : int
#     )->ndarray:
#     return np.zeros((1, 33))

#   def residual_C(      
#       x           : ndarray,
#       uv          : ndarray,
#       weights     : ndarray,
#       detections  : ndarray,
#       iteration   : int
#     )->ndarray:
#     return np.zeros((1, 39))

#   if method == 'A':
#     return residual_A(x, uv, weights, K, T_p_to_c) 
#   # elif method == 'B':
#   #   return residual_B(x, uv, weights, detections, iteration)
#   # else:
#   #   return residual_C(x, uv, weights, detections, iteration)
  

def detected_trajectory(do_plot=False):
  all_detections = np.loadtxt(os.path.join(sys.path[0], '../data/data/detections.txt'))
  quanser = Quanser()

  p = np.array([0.0, 0.0, 0.0])
  all_r = []
  all_p = []
  for i in range(len(all_detections)):
    weights = all_detections[i, ::3]
    uv = np.vstack((all_detections[i, 1::3], all_detections[i, 2::3]))

    # Tip: Lambda functions can be defined inside a for-loop, defining
    # a different function in each iteration. Here we pass in the current
    # image's "uv" and "weights", which get loaded at the top of the loop.
    resfun = lambda p : quanser.residuals(uv, weights, p[0], p[1], p[2])

    # Tip: Use the previous image's parameter estimate as initialization
    p = least_squares(resfun, x0=p, method='lm').x

    # Collect residuals and parameter estimate for plotting later
    all_r.append(resfun(p))
    all_p.append(p)

  all_p = np.array(all_p)
  all_r = np.array(all_r)
  if do_plot:
    plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
    #plt.savefig('out_part1b.png')
    plt.show()


  return all_p, all_r

class KinematicModelA:
  def __init__(self):
    self.T_platform_camera = np.loadtxt(os.path.join(sys.path[0], "../data/data/platform_to_camera.txt"))
    self.detections = np.loadtxt(os.path.join(sys.path[0], "../data/data/detections.txt"))
    self.K = np.loadtxt(os.path.join(sys.path[0], "../data/data/K.txt"))

    self.N = self.detections.shape[0]
    self.M = 7

    initial_lengths = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.1]
    )  # actual values: 0.1145, 0.325, 0.050, 0.65, 0.030
    initial_markers = 0.1 * np.ones(21)  # actual values in heli_points[:3, :]
    self.initial_parameters = np.hstack((initial_lengths, initial_markers))
    self.KP = self.initial_parameters.shape[0]

    A1 = np.ones([2 * self.M * self.N, self.KP])
    B = np.ones([2 * self.M, 3])
    A2 = B.copy()
    for _ in range(self.N - 1):
        A2 = block_diag(A2, B)
    self.JS = np.block([A1, A2])

  def T_hat(self, kinematic_parameters, rpy):
    T_base_platform = com.translate(
        kinematic_parameters[0] / 2, kinematic_parameters[0] / 2, 0.0
    ) @ com.rotate_z(rpy[0])
    T_hinge_base = com.translate(0.0, 0.0, kinematic_parameters[1]) @ com.rotate_y(rpy[1])
    T_arm_hinge = com.translate(0.0, 0.0, -kinematic_parameters[2])
    T_rotors_arm = com.translate(
        kinematic_parameters[3], 0.0, -kinematic_parameters[4]
    ) @ com.rotate_x(rpy[2])

    T_base_camera = self.T_platform_camera @ T_base_platform
    T_hinge_camera = T_base_camera @ T_hinge_base
    T_arm_camera = T_hinge_camera @ T_arm_hinge
    T_rotors_camera = T_arm_camera @ T_rotors_arm

    return T_rotors_camera, T_arm_camera


class BatchOptimizer:
  def __init__(self, model, tol=1e-6):
    self.model = model
    self.xtol = tol

    initial_state_parameters, _ = detected_trajectory()
    initial_state_parameters = initial_state_parameters.flatten()
    self.initial_parameters = np.hstack(
      [kinematic_model.initial_parameters, initial_state_parameters]
    )

  def u_hat(self, kinematic_parameters, state):
    marker_points = kinematic_parameters[-3 * self.model.M :].reshape(
      (3, self.model.M)
    )
    marker_points = np.vstack((marker_points, np.ones((1, self.model.M))))

    T_rotors_camera, T_arm_camera = self.model.T_hat(kinematic_parameters, state)

    markers_rotor = T_rotors_camera @ marker_points[:, 3:]
    markers_arm = T_arm_camera @ marker_points[:, :3]

    X = np.hstack([markers_arm, markers_rotor])
    u = com.project(self.model.K, X)
    return u

  def residuals(self, p):
    kinematic_parameters = p[: self.model.KP]
    state_parameters = p[self.model.KP :]

    r = np.zeros(2 * self.model.M * self.model.N)
    for i in range(self.model.N):
      ui = self.model.detections[i, 1::3]
      vi = self.model.detections[i, 2::3]
      weights = self.model.detections[i, ::3]
      u = np.vstack((ui, vi))

      state = state_parameters[3 * i : 3 * (i + 1)]

      r[2 * self.model.M * i : 2 * self.model.M * (i + 1)] = (
        weights * (self.u_hat(kinematic_parameters, state) - u)
      ).flatten()

    return r

  def optimize(self):

    optimized_parameters = least_squares(
      self.residuals,
      self.initial_parameters,
      xtol=self.xtol,
      jac_sparsity=self.model.JS,
    )

    return optimized_parameters.x


if __name__ == "__main__":

  kinematic_model = KinematicModelA()

  optimizer = BatchOptimizer(kinematic_model, tol=1e-2)
  optimized_parameters = optimizer.optimize()

  do_plot = True
  if do_plot:
    all_r = []
    all_p = []
    KP = kinematic_model.KP
    for image_number in range(kinematic_model.N):
      rpy = optimized_parameters[
        KP + 3 * image_number : KP + 3 * (image_number + 1)
      ]
      all_p.append(rpy)
      all_r.append(optimizer.residuals(optimized_parameters))
    all_p = np.array(all_p)
    all_r = np.zeros((351, 2 * 7))

    all_detections = np.loadtxt(os.path.join(sys.path[0], "../data/data/detections.txt"))
    plot_all.plot_all(all_p, all_r, all_detections, subtract_initial_offset=True)
    # plt.savefig('out_part3.png')
    plt.show()

  do_anim = False
  if do_anim:
    plt.ion()
    fig, ax = plt.subplots()
    plt.draw()
    KP = kinematic_model.KP
    for image_number in range(kinematic_model.N):
      kinematic_parameters = optimized_parameters[:KP]
      rpy = optimized_parameters[
        KP + 3 * image_number : KP + 3 * (image_number + 1)
      ]

      plt.imshow(plt.imread(os.path.join(sys.path[0], "../data/quanser/img_sequence/video%04d.jpg" % image_number)))
      ax.scatter(
        *optimizer.u_hat(kinematic_parameters, rpy),
        linewidths=1,
        color="yellow",
        s=10,
      )

      plt.pause(0.05)
      ax.clear()

# K = np.loadtxt(os.path.join(sys.path[0], '../data/data/K.txt'))
# heli_points = np.loadtxt(os.path.join(sys.path[0], '../data/data/heli_points.txt')).T
# T_p_to_c = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_to_camera.txt'))



# N = 351
# Kin = 26
# M = 7

# method = 'A'
# if method == 'A':
#   x = np.zeros((Kin + 3 * N))
#   x[:5] = np.array([0.1145, 0.325, -0.05, 0.65, -0.03]) 

# elif method == 'B':
#   x = np.zeros(33)
# elif method == 'C':
#   x = np.zeros(39)
# else:
#   raise "Invalid option" 


# all_r = []
# all_x = []
# #for i in range(len(all_detections)):
# weights = all_detections[:, ::3]
# uv = np.vstack((all_detections[:, 1::3], all_detections[:, 2::3]))

# sparsity_block = np.ones((2 * M, 3))
# state_sparsity =  np.kron(np.eye(N), sparsity_block)

# jac_sparsity = np.hstack( [ np.ones((2 * M * N, Kin)), state_sparsity ] )

# resfun = lambda x : residual_function(x, uv, weights, K, heli_points, T_p_to_c, method)

#   # But I cannot see how one should make this into a batch-optimization problem...
# x = least_squares(resfun, x0=x, jac_sparsity=jac_sparsity).x
#   # x = minimize(resfun, x0=x).x

#   #print(resfun(x))

# all_r.append(resfun(x))
# all_x.append(x)

# all_p = np.array(all_x)
# all_r = np.array(all_r)
# # Tip: See comment in plot_all.py regarding the last argument.
# plot_all.plot_all(all_p.reshape(-1), all_r, all_detections, subtract_initial_offset=True, is_task_3=True)
# # plt.savefig('out_part1b.png')
# plt.show()

# """
# K??????????
# How can you calculated resedual function for K?

# What should be optimized? roll, pitch, yaw or marker positions?

# Task 2.3: How are we supposed to show difference, not clear what the task asks
# Task 2.3: How big difference is expected? 

# """