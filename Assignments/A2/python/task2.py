from ctypes import sizeof
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from Assignments.A2.python.common import extract_local_maxima
import common as com

# This bit of code is from HW1.
edge_threshold = 0.015
blur_sigma     = 1
filename       = '../data/grid.jpg'
I_rgb          = plt.imread(os.path.join(sys.path[0], filename))
I_rgb          = com.im2double(I_rgb) #Ensures that the image is in floating-point with pixel values in [0,1].
I_gray         = com.rgb_to_gray(I_rgb)
Ix, Iy, Im     = com.derivative_of_gaussian(I_gray, sigma=blur_sigma) # See HW1 Task 3.6
x,y,theta      = com.extract_edges(Ix, Iy, Im, edge_threshold)

# You can adjust these for better results
line_threshold = 0.2
N_rho          = 800
N_theta        = 800

###########################################
#
# Task 2.1: Determine appropriate ranges
#
###########################################
theta_max = np.pi
theta_min = -np.pi 

height, width, _ = I_rgb.shape

rho_max = np.sqrt(height**2 + width**2)
rho_min = -rho_max 


###########################################
#
# Task 2.2: Compute the accumulator array
#
# Tip: You can do this without for-loops

# 2) Convert to discrete row,column coordinates
# Tip: Use np.floor(...).astype(np.int) to floor a number to an integer type

# 3) Increment H[row,column]
# Tip: Make sure that you don't try to access values at indices outside
# the valid range: [0,N_rho-1] and [0,N_theta-1]
H = np.zeros((N_rho, N_theta))

rho = x*np.cos(theta) + y*np.sin(theta)

row = np.floor(N_rho*(rho-rho_min)/(rho_max-rho_min)).astype(int)
col = np.floor(N_theta*(theta-theta_min)/(theta_max - theta_min)).astype(int)

for r, c in (zip(row, col)):
  H[r,c] += 1



###########################################
#
# Task 2.3: Extract local maxima
#
###########################################
# 1) Call extract_local_maxima
maxima_rows, maxima_cols = com.extract_local_maxima(H, line_threshold)

# 2) Convert (row, column) back to (rho, theta)
maxima_rho = maxima_rows * (1/N_rho) * (rho_max-rho_min) + rho_min
maxima_theta = maxima_cols * (1/N_theta) * (theta_max - theta_min) + theta_min

###########################################
#
# Figure 2.2: Display the accumulator array and local maxima
#
###########################################
plt.figure()
plt.imshow(H, extent=[theta_min, theta_max, rho_max, rho_min], aspect='auto')
plt.colorbar(label='Votes')
plt.scatter(maxima_theta, maxima_rho, marker='.', color='red')
plt.title('Accumulator array')
plt.xlabel('$\\theta$ (radians)')
plt.ylabel('$\\rho$ (pixels)')
# plt.savefig('out_array.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure

###########################################
#
# Figure 2.3: Draw the lines back onto the input image
#
###########################################
plt.figure()
plt.imshow(I_rgb)
plt.xlim([0, I_rgb.shape[1]])
plt.ylim([I_rgb.shape[0], 0])
for theta,rho in zip(maxima_theta,maxima_rho):
    com.draw_line(theta, rho, color='yellow')
plt.title('Dominant lines')
# plt.savefig('out_lines.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure

plt.show()
