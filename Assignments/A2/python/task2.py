from ctypes import sizeof
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
# Tip: theta is computed using np.arctan2. Check that the
# range of values returned by arctan2 matches your chosen
# ranges (check np.info(np.arctan2) or the internet docs).

# I assume that the image goes from (0,0) to (max, max) for whatever max is 
rho_max   = np.sqrt(I_rgb.shape[0]**2 + I_rgb.shape[1]**2)
# rho_max   = np.ceil(np.linalg.norm(np.array([I_rgb.shape[0], I_rgb.shape[1]]))) # I get many more lines using this method!
rho_min   = -rho_max # 0

print(np.ceil(np.linalg.norm(np.array([I_rgb.shape[0], I_rgb.shape[1]])))) # Will add extra pixcels that do not exist
print(np.sqrt(I_rgb.shape[0]**2 + I_rgb.shape[1]**2))

theta_max = np.pi # np.pi / 2.0
theta_min = -np.pi # 0

###########################################
#
# Task 2.2: Compute the accumulator array
#
###########################################
# Zero-initialize an array to hold our votes
H = np.zeros((N_rho, N_theta))

# 1) Compute rho for each edge (x,y,theta)
# Tip: You can do this without for-loops
# theta is the angle of the detected edge, while x,y are the given coordinates for said edge
# rhos = np.linalg.norm(np.array([x, y])) # Distances to each detected edge - gives just the 2 norm of the full array
# rhos = np.zeros_like(x)
# thetas = np.zeros_like(rhos)
# for i, (x_p, y_p) in enumerate(zip(x, y)):
#   rhos[i] = np.linalg.norm(np.array([x_p, y_p]))
#   thetas[i] = np.arctan2(y_p, x_p)

# Stupid me. Just use the definition of rho and theta.
# I thought in the planar case where rhos = sqrt(x^2 + y^2) and not using
# the Hough transform  
rhos = x * np.cos(theta) + y * np.sin(theta)

# 2) Convert to discrete row,column coordinates
# Tip: Use np.floor(...).astype(np.int) to floor a number to an integer type
rows = np.floor(N_rho * ((rhos - rho_min) / (rho_max - rho_min))).astype(int)
cols = np.floor(N_theta * ((theta - theta_min) / (theta_max - theta_min))).astype(int)

# 3) Increment H[row,column]
# Tip: Make sure that you don't try to access values at indices outside
# the valid range: [0,N_rho-1] and [0,N_theta-1]
def is_valid_coordinate(r, c):
  return (r >= 0) and (r < N_rho) and (c >= 0) and (c < N_theta)

for r, c in zip(rows, cols):
  if not is_valid_coordinate(r, c):
    print((r, c))
    continue 
  H[r][c] += 1

# Ia am a bit unsure what is wrong here. On one hand, one can see that the 
# reported lines are just reappering for each radian, which makes some sence when you ¨
# look at it using the radially view. Hoever, it still does not make sence that 
# it will occur for every radian, as one should expect the angle to change for each
# pixel/edgels that is detected
# I suspect the error is related to how I calculated the thetas  


###########################################
#
# Task 2.3: Extract local maxima
#
###########################################
# 1) Call extract_local_maxima
(rows, cols) = com.extract_local_maxima(H, threshold=line_threshold)

# 2) Convert (row, column) back to (rho, theta)
# Use equation 3 given in the assignment
maxima_rho = (rows * ((rho_max - rho_min) / N_rho)) + rho_min
maxima_theta = (cols * ((theta_max - theta_min) / N_theta)) + theta_min

# maxima_theta = np.zeros_like(rows) # np.arctan2(cols, rows)
# # Cannot quite understand how the conversion to theta should be.
# maxima_rho = cols * np.cos(maxima_theta) + rows * np.sin(maxima_theta)

# for i, (r, c) in enumerate(zip(rows, cols)):
#   maxima_rho[i] = np.linalg.norm(np.array([r, c]))
#   maxima_theta[i] = np.arctan2(r, c)


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
