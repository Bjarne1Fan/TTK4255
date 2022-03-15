import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def hline(
      l       : np.ndarray, 
      **args
    ) -> None:
  """
  Draws a homogeneous 2D line.
  You must explicitly set the figure xlim, ylim before or after using this.
  """

  lim = np.array([-1e8, +1e8]) # Surely you don't have a figure bigger than this!
  a, b, c = l
  if np.absolute(a) > np.absolute(b):
    x, y = -(c + b*lim) / a, lim
  else:
    x, y = lim, -(c + a * lim) / b
  plt.plot(x, y, **args)

def draw_correspondences(
      I1          : np.ndarray, 
      I2          : np.ndarray, 
      uv1         : np.ndarray, 
      uv2         : np.ndarray, 
      F           : np.ndarray, 
      sample_size : int         = 8
    ) -> None:
  """
  Draws a random subset of point correspondences and their epipolar lines.
  """

  assert uv1.shape[0] == 3 and uv2.shape[0] == 3, \
     'uv1 and uv2 must be 3 x n arrays of homogeneous 2D coordinates.'
  sample = np.random.choice(range(uv1.shape[1]), size=sample_size, replace=False)
  uv1 = uv1[:,sample]
  uv2 = uv2[:,sample]
  n = uv1.shape[1]
  uv1 /= uv1[2,:]
  uv2 /= uv2[2,:]

  l1 = F.T @ uv2
  l2 = F @ uv1

  colors = plt.cm.get_cmap('Set2', n).colors
  plt.figure('Correspondences', figsize=(10,4))
  plt.subplot(121)
  plt.imshow(I1)
  plt.xlabel('Image 1')
  plt.scatter(*uv1[:2,:], s=100, marker='x', c=colors)
  for i in range(n):
    hline(l1[:,i], linewidth=1, color=colors[i], linestyle='--')
  plt.xlim([0, I1.shape[1]])
  plt.ylim([I1.shape[0], 0])

  plt.subplot(122)
  plt.imshow(I2)
  plt.xlabel('Image 2')
  plt.scatter(*uv2[:2,:], s=100, marker='o', zorder=10, facecolor='none', edgecolors=colors, linewidths=2)
  for i in range(n):
    hline(l2[:,i], linewidth=1, color=colors[i], linestyle='--')
  plt.xlim([0, I2.shape[1]])
  plt.ylim([I2.shape[0], 0])
  plt.tight_layout()
  plt.suptitle('Point correspondences and associated epipolar lines (showing %d randomly drawn pairs)' % sample_size)

def draw_point_cloud(
      X     : np.ndarray, 
      I1    : np.ndarray, 
      uv1   : np.ndarray, 
      xlim  : np.ndarray, 
      ylim  : np.ndarray, 
      zlim  : np.ndarray
    ) -> None:

  assert uv1.shape[1] == X.shape[1], '\
    If you get this error message in Task 4,\
    it probably means that you did not extract the inliers of all the arrays (uv1,uv2,xy1,xy2)\
    before calling draw_point_cloud.'

  # We take I1 and uv1 as arguments in order to assign a color to each
  # 3D point, based on its pixel coordinates in one of the images.
  c = I1[uv1[1,:].astype(np.int32), uv1[0,:].astype(np.int32), :]

  # Matplotlib doesn't let you easily change the up-axis to match the
  # convention we use in the course (it assumes Z is upward). So this
  # code does a silly rearrangement of the Y and Z arguments.
  plt.figure('3D point cloud', figsize=(6,6))
  ax = plt.axes(projection='3d')
  ax.scatter(X[0,:], X[2,:], X[1,:], c=c, marker='.', depthshade=False)
  ax.grid(False)
  ax.set_xlim(xlim)
  ax.set_ylim(zlim)               # Is this correct?????
  ax.set_zlim([ylim[1], ylim[0]]) # Is this correct?????
  ax.set_xlabel('X')
  ax.set_ylabel('Z')
  ax.set_zlabel('Y')
  plt.title('[Click, hold and drag with the mouse to rotate the view]')

def draw_residual_histograms(
      residuals   : np.ndarray, 
      matrix_name : str,
      num_bins    : int         = 1000,
      ranges      : tuple       = (-2000, 2000)
    ) -> None:
  assert residuals.shape[0] == 2, "Incorrect shape"

  norm_residuals = np.linalg.norm(residuals, axis=0)
  avg_residuals = (residuals[0] + residuals[1]) / 2 #np.mean(residuals, axis=0)
  n = residuals.shape[1]

  fig, axs = plt.subplots(3, 1, sharey=True, tight_layout=True)
  fig.suptitle(
    "Histogram for residual errors using {} essential matrix for {} number of values".format(
      matrix_name,
      residuals.shape[1] 
    )
  )

  axs[0].set_title("Residual e1")
  axs[0].hist(residuals[0,:], bins=num_bins, range=ranges)

  axs[1].set_title("Residual e2")
  axs[1].hist(residuals[1,:], bins=num_bins, range=ranges)

  # axs[2].set_title("Two-norm of residuals")
  # axs[2].hist(norm_residuals, bins=num_bins, range=range)

  axs[2].set_title("Average of residuals")
  axs[2].hist(avg_residuals, bins=num_bins, range=ranges)

  # plt.xlabel("Residual errors")
  # plt.ylabel("# occurences")
  plt.setp(axs[:], xlabel='Error')
  plt.setp(axs[:], ylabel='Number times')