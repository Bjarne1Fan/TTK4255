import numpy as np
from numpy import random

def rgb_to_gray(I):
    """
    Converts a HxWx3 RGB image to a HxW grayscale image as
    described in the text.
    """
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]


    return I.mean(axis=2) #(R + G + B) / 3

def central_difference(I, kernel = np.array([0.5, 0, -0.5])):
    """
    Computes the gradient in the x and y direction using
    a central difference filter, and returns the resulting
    gradient images (Ix, Iy) and the gradient magnitude Im.
    """
    # kernel = np.array([0.5, 0, -0.5])

    Ix = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='same'), axis=0, arr=I)
    Iy = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode='same'), axis=1, arr=I)
    Im = np.sqrt(np.square(Ix) + np.square(Iy))

    return Ix, Iy, Im

def gaussian(I, sigma):
    """
    Applies a 2-D Gaussian blur with standard deviation sigma to
    a grayscale image I.
    """
    # Hint: The size of the kernel should depend on sigma. A common
    # choice is to make the half-width be 3 standard deviations. The
    # total kernel width is then 2*np.ceil(3*sigma) + 1.
    kernel_width = int(2 * np.ceil(3 * sigma) + 1)
    cov = np.eye(2) * sigma**2

    kernal_row = random.normal(0, sigma**2, size = (kernel_width))
    kernal_col = random.normal(0, sigma**2, size = (kernel_width))

    I_dot, _, _ = central_difference(I, kernal_row)
    _, result, _ = central_difference(I_dot, kernal_col)

    # result = np.zeros_like(I) # Placeholder
    return result


def extract_edges(Ix, Iy, Im, threshold):
    """

    Returns the x, y coordinates of pixels whose gradient
    magnitude is greater than the threshold. Also, returns
    the angle of the image gradient at each extracted edge.
    """

    above_treshold = Im > threshold
    ind = np.nonzero(above_treshold)

    theta = np.arctan2(Iy[ind], Ix[ind])
    ret_vals = [ind[0,:], ind[1,:], theta] #np.concatenate([ind, theta])
    return ret_vals
