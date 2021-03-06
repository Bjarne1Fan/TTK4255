import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import common as com

from numpy import ndarray
from typing import Callable

class Quanser:
  def __init__(self)->None:
    self.K = np.loadtxt(os.path.join(sys.path[0], '../data/data/K.txt'))
    self.heli_points = np.loadtxt(os.path.join(sys.path[0], '../data/data/heli_points.txt')).T
    self.platform_to_camera = np.loadtxt(os.path.join(sys.path[0], '../data/data/platform_to_camera.txt'))

  def residuals(
        self    : Callable, 
        uv      : ndarray, 
        weights : ndarray, 
        yaw     : float, 
        pitch   : float, 
        roll    : float
      )->ndarray:
    # Compute the helicopter coordinate frames
    base_to_platform      = com.translate(0.1145/2, 0.1145/2, 0.0) @ com.rotate_z(yaw)
    hinge_to_base         = com.translate(0.00, 0.00,  0.325) @ com.rotate_y(pitch)
    arm_to_hinge          = com.translate(0.00, 0.00, -0.050)
    rotors_to_arm         = com.translate(0.65, 0.00, -0.030) @ com.rotate_x(roll)
    self.base_to_camera   = self.platform_to_camera @ base_to_platform
    self.hinge_to_camera  = self.base_to_camera @ hinge_to_base
    self.arm_to_camera    = self.hinge_to_camera @ arm_to_hinge
    self.rotors_to_camera = self.arm_to_camera @ rotors_to_arm

    # Compute the predicted image location of the markers
    p1 = self.arm_to_camera @ self.heli_points[:,:3]
    p2 = self.rotors_to_camera @ self.heli_points[:,3:]
    uv_hat = com.project(self.K, np.hstack([p1, p2]))
    self.uv_hat = uv_hat # Save for use in draw()

    # Must include the weights in the residuals. Otherwise, we would get a massive error
    # when the camera has no observations. 
    r = (uv_hat - uv) * weights
    return np.hstack([r[0], r[1]])

  def draw(
        self          : Callable, 
        uv            : ndarray, 
        weights       : ndarray, 
        image_number  : ndarray
      )->None:
    I = plt.imread(os.path.join(sys.path[0], '../data/quanser/video%04d.jpg' % image_number))
    plt.imshow(I)
    plt.scatter(*uv[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
    plt.scatter(*self.uv_hat, color='red', label='Predicted', s=10)
    plt.legend()
    plt.title('Reprojected frames and points on image number %d' % image_number)
    com.draw_frame(self.K, self.platform_to_camera, scale=0.05)
    com.draw_frame(self.K, self.base_to_camera, scale=0.05)
    com.draw_frame(self.K, self.hinge_to_camera, scale=0.05)
    com.draw_frame(self.K, self.arm_to_camera, scale=0.05)
    com.draw_frame(self.K, self.rotors_to_camera, scale=0.05)
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
