from inspect import trace
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
alpha = 0.06*0.75
threshold = 5e-5


Ix, Iy, Im = com.derivative_of_gaussian(I, sigma=sigma_D)

IxIx = np.square(Ix)
IyIy = np.square(Iy)
IxIy = np.multiply(Ix,Iy)

A1 = com.gaussian(IxIx, sigma_I)
A2 = com.gaussian(IxIy, sigma_I)
A3 = com.gaussian(IxIy, sigma_I)
A4 = com.gaussian(IyIy, sigma_I)

R = np.zeros_like(I)

h, w = I.shape

for r in range(0,h):
  for c in range(0,w):
    A = [[A1[r,c], A2[r,c]], [A3[r,c], A4[r,c]]]
    R[r,c] = np.linalg.det(A) - alpha * np.trace(A)**2

response = R > threshold  
##########################################
#
# Task 3.4: Extract local maxima
#
###########################################
local_maximum = com.extract_local_maxima(R, 0.001)

corners_y = local_maximum[0]
corners_x = local_maximum[1]

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
