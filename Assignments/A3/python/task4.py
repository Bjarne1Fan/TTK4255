import os
import sys 

import numpy as np
import matplotlib.pyplot as plt
import common as com

K = np.loadtxt(os.path.join(sys.path[0], '../data/heli_K.txt'))
T_p_c = np.loadtxt(os.path.join(sys.path[0], '../data/platform_to_camera.txt')) 
heli_jpg = os.path.join(sys.path[0], '../data/quanser.jpg')

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
is_task_4_7 = False

# Angles
phi = 0
theta = 28.9
psi = 11.6

# Base
B = 1/2 * P
T_b_p = com.translate(B, B, 0) @ com.rotate_z(-psi)

# Hinge
H = 0.325 # Height from base to hinge in z: [m]
T_h_b = com.translate(0, 0, H) @ com.rotate_y(theta)

# Arm
A = -0.05  # Distance from hinge to arm in z: [m]
T_a_h = com.translate(0, 0, A)

# Rotor
Rx = 0.65 # Distance from arm to rotor in x: [m]
Rz = -0.03# Distance from arm to rotor in z: [m]
T_r_a = com.translate(Rx, 0, Rz) @ com.rotate_x(phi)

# Interesting points to map
points = [p_0, p_1, p_2, p_3]

if is_task_4_7:
  base = T_b_p @ P_0
  hinge = T_h_b @ base
  arm = T_a_h @ hinge # Almost like the arm does not take the hinge orientation into account
  rotor = T_r_a @ arm 
  points = points + [base, hinge, arm, rotor]

# Transform the points into pixel-coordinates
X_p = np.array(points).T # Fuck this transpose.... Kill me!
X_c = T_p_c @ X_p

u,v = com.project(K, X_c)

plt.scatter(u, v, c='red', marker='.', s=50)
plt.show()
