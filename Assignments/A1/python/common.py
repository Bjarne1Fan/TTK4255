import numpy as np
from numpy import random

def rgb_to_gray(I):
  """
  Converts a HxWx3 RGB image to a HxW grayscale image as
  described in the text.
  """
  return I.mean(axis=2) #(R + G + B) / 3

def central_difference(I, kernel = np.array([0.5, 0, -0.5])):
  """
  Computes the gradient in the x and y direction using
  a central difference filter, and returns the resulting
  gradient images (Ix, Iy) and the gradient magnitude Im.
  """
  Ix = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='same'), axis=0, arr=I)
  Iy = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode='same'), axis=1, arr=I)
  Im = np.sqrt(np.square(Ix) + np.square(Iy))

  return Ix, Iy, Im

def gaussian(I, sigma):
  """
  Applies a 2-D Gaussian blur with standard deviation sigma to
  a grayscale image I.
  """
  if sigma == 0:
    return I

  # Hint: The size of the kernel should depend on sigma. A common
  # choice is to make the half-width be 3 standard deviations. The
  # total kernel width is then 2*np.ceil(3*sigma) + 1.
  kernel_width = int(2 * np.ceil(3 * sigma) + 1)

  # Obs! Cannot use the distribution to draw the kernel.
  # This will just generate a kernel that will vary in value, but not represent
  # a gaussian. We must therefore create a discrete gaussian-valued kernel  
  # kernel_row = random.normal(0, sigma, size = (kernel_width))
  # kernel_col = random.normal(0, sigma, size = (kernel_width))
  
  def disc_gaussian(h: float, sigma: float)->float:
    return 1 / (2 * np.pi * sigma**2) * np.exp(-1/(2 * sigma**2) * h**2)

  kernel = np.zeros(kernel_width)
  for i in range(kernel_width):
    kernel[i] = disc_gaussian(i - kernel_width // 2, sigma)

  I_dot, _, _ = central_difference(I, kernel)
  _, result, _ = central_difference(I_dot, kernel)

  # Also necessary to normalize the values
  return result / np.max(result)


def extract_edges(Ix, Iy, Im, threshold):
  """
  Returns the x, y coordinates of pixels whose gradient
  magnitude is greater than the threshold. Also, returns
  the angle of the image gradient at each extracted edge.
  """

  above_treshold = Im > threshold
  ind = np.nonzero(above_treshold)

  theta = np.arctan2(Iy[ind], Ix[ind])
  return ind[1].T, ind[0].T, theta
