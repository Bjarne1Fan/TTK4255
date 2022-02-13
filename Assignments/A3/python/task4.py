import os
import sys 

import numpy as np
import matplotlib.pyplot as plt
import common as com

K = np.loadtxt(os.path.join(sys.path[0], '../data/heli_K.txt'))
T_platform_to_camera = np.loadtxt(os.path.join(sys.path[0], '../data/platform_to_camera.txt')) 
heli_jpg = os.path.join(sys.path[0], '../data/quanser.jpg')
heli_points = np.loadtxt(os.path.join(sys.path[0], '../data/heli_points.txt'))

# I = Image.open(heli_jpg)
I = plt.imread(heli_jpg)
plt.imshow(I)

# Coordinate points
# Platform
P = 0.1145 # Length of the platform in x and y: [m]
P_0 = np.array([0, 0, 0, 1]).T
p_0 = P_0
p_1 = com.translate(P, 0, 0) @ P_0
p_2 = com.translate(0, P, 0) @ P_0
p_3 = com.translate(P, P, 0) @ P_0

# Selcting between task 4.3 and task4.7
is_task_4_7 = True

# Angles
phi = 0
theta = 28.9
psi = 11.6

# Base
B = 1/2 * P
T_base_to_platform = com.translate(B, B, 0) @ com.rotate_z(psi)
#T_base_to_platform = np.linalg.inv(T_base_to_platform) # com.rotate_z(-psi) @ com.translate(-B, -B, 0)

# Hinge
H = 0.325 # Height from base to hinge in z: [m]
T_hinge_to_base = com.translate(0, 0, H) @ com.rotate_y(theta)
#T_hinge_to_base = np.linalg.inv(T_hinge_to_base) # com.rotate_y(-theta) @ com.translate(0, 0, -H)

# Arm
A = -0.05  # Distance from hinge to arm in z: [m]
T_arm_to_hinge = com.translate(0, 0, A)
#T_arm_to_hinge = np.linalg.inv(T_hinge_to_arm) # com.translate(0, 0, -A)

# Rotor
Rx = 0.65 # Distance from arm to rotor in x: [m]
Rz = -0.03# Distance from arm to rotor in z: [m]
T_rotor_to_arm = com.translate(Rx, 0, Rz) @ com.rotate_x(phi)
#T_rotor_to_arm = np.linalg.inv(T_arm_to_rotor) # com.rotate_x(-phi) @ com.translate(-Rx, 0, -Rz)

# Interesting points to map
points = [p_0, p_1, p_2, p_3]
# points = [p_0, p_1]

# The first three points in the heli-coordinates are given in the arm-frame
# The last four are given in rotor frame

if is_task_4_7:
  # T_platform_to_arm = np.linalg.inv(T_base_to_platform @ T_hinge_to_base @ T_hinge_to_arm)
  T_arm_to_platform = T_base_to_platform @ T_hinge_to_base @ T_arm_to_hinge

  a_0 = T_arm_to_platform @ heli_points[0]
  a_1 = T_arm_to_platform @ heli_points[1]
  a_2 = T_arm_to_platform @ heli_points[2]

  T_rotor_to_platform = T_arm_to_platform @ T_rotor_to_arm 
  r_0 = T_rotor_to_platform @ heli_points[3]
  r_1 = T_rotor_to_platform @ heli_points[4]
  r_2 = T_rotor_to_platform @ heli_points[5]
  r_3 = T_rotor_to_platform @ heli_points[6]

  points = points + [a_0, a_1, a_2, r_0, r_1, r_2, r_3]

  # Draw the frames
  com.draw_frame(K, T_platform_to_camera, scale=0.05)
  com.draw_frame(K, T_platform_to_camera @ T_base_to_platform, scale=0.05)
  com.draw_frame(K, T_platform_to_camera @ T_base_to_platform @ T_hinge_to_base, scale=0.05)
  com.draw_frame(K, T_platform_to_camera @ T_base_to_platform @ T_hinge_to_base @ T_arm_to_hinge, scale=0.05)
  com.draw_frame(K, T_platform_to_camera @ T_base_to_platform @ T_hinge_to_base @ T_arm_to_hinge @ T_rotor_to_arm, scale=0.05)


  # base = T_base_to_platform @ P_0
  # hinge = T_base_to_platform @ T_hinge_to_base @ base
  # arm = T_base_to_platform @ T_hinge_to_base @ T_hinge_to_arm @ hinge # Almost like the arm does not take the hinge orientation into account
  # rotor = T_base_to_platform @ T_hinge_to_base @ T_hinge_to_arm @ T_arm_to_rotor @ arm 
  # points = points + [base, hinge, arm, rotor]

# Transform the points into pixel-coordinates
X_p = np.array(points).T
X_c = T_platform_to_camera @ X_p

u,v = com.project(K, X_c)
plt.scatter(u, v, c='yellow', marker='.', s=100)

plt.show()
