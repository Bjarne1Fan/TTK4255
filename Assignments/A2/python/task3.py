import os 
import sys
import matplotlib.pyplot as plt
import numpy as np
import common as com

# Note: the sample image is naturally grayscale
filename       = '../data/calibration.jpg'
I = com.rgb_to_gray(
      com.im2double(
        plt.imread(
          os.path.join(sys.path[0], filename)
        )
      )
    )

###########################################
#
# Task 3.1: Compute the Harris-Stephens measure
#
###########################################
sigma_D = 1
sigma_I = 3
alpha = 0.06
Harris_threshold = 0.1

# Smoothing the original image slightly
I_s = com.gaussian(I, sigma_I)

# Calculating the derivatives of the image
Ix, Iy, _ = com.derivative_of_gaussian(I_s, sigma_D)

Ix_square = Ix @ Ix.T 
Iy_square = Iy @ Iy.T
IxIy = Ix @ Iy.T

Ixx, Ixy, _ = com.derivative_of_gaussian(Ix_square, sigma_D)
Ixy, Iyy, _ = com.derivative_of_gaussian(Iy_square, sigma_D)

# Ixx, Ixy, _ = com.derivative_of_gaussian(Ix @ Ix.T, sigma_D)#(np.sqrt(Ix @ Ix.T), sigma_D)
# _, Iyy, _ = com.derivative_of_gaussian(Iy @ Iy.T, sigma_D)#np.sqrt(Iy @ Iy.T), sigma_D)

# Creating the weighted matrix
# WARNING: A does apparently contain infs or nan
A = np.block([[Ixx, Ixy], [Ixy, Iyy]])

# Getting eigenvalues and eigenvectors
# (lambda_0, lambda_1), _ = np.linalg.eig(A)

# Corner response function
print(A.shape)
R = np.linalg.det(A) - alpha * np.trace(A)**2 # How to calculate this? Ask Fanebust!

response = R > Harris_threshold #np.zeros_like(I)

###########################################
#
# Task 3.4: Extract local maxima
#
###########################################
corners_y = [0] # Placeholder
corners_x = [0] # Placeholder

###########################################
#
# Figure 3.1: Display Harris-Stephens corner strength
#
###########################################
plt.figure(figsize=(13,5))
plt.imshow(response)
plt.colorbar(label='Corner strength')
plt.tight_layout()
# plt.savefig('out_corner_strength.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure in working directory

###########################################
#
# Figure 3.4: Display extracted corners
#
###########################################
plt.figure(figsize=(10,5))
plt.imshow(I, cmap='gray')
plt.scatter(corners_x, corners_y, linewidths=1, edgecolor='black', color='yellow', s=9)
plt.tight_layout()
# plt.savefig('out_corners.png', bbox_inches='tight', pad_inches=0) # Uncomment to save figure in working directory

plt.show()
