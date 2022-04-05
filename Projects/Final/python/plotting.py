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
  plt.suptitle('Point correspondences and associated epipolar lines (showing %d pairs)' % sample_size)


def draw_frame(
      ax    : plt.Axes,  
      T     : np.ndarray, 
      scale : float
    ) -> None:
  X0 = T @ np.array((0,0,0,1))
  X1 = T @ np.array((1,0,0,1))
  X2 = T @ np.array((0,1,0,1))
  X3 = T @ np.array((0,0,1,1))
  ax.plot([X0[0], X1[0]], [X0[2], X1[2]], [X0[1], X1[1]], color='#FF7F0E')
  ax.plot([X0[0], X2[0]], [X0[2], X2[2]], [X0[1], X2[1]], color='#2CA02C')
  ax.plot([X0[0], X3[0]], [X0[2], X3[2]], [X0[1], X3[1]], color='#1F77B4')


def draw_point_cloud(
      X           : np.ndarray, 
      T_m2q       : np.ndarray, 
      xlim        : float, 
      ylim        : float, 
      zlim        : float,  
      colors      : np.ndarray, 
      marker_size : float, 
      frame_size  : float
    ) -> None:
  ax = plt.axes(projection='3d')
  ax.set_box_aspect((1, 1, 1))
  if colors.max() > 1.1:
    colors = colors.copy() / 255
  ax.scatter(X[0,:], X[2,:], X[1,:], c=colors, marker='.', s=marker_size, depthshade=False)
  draw_frame(ax, np.linalg.inv(T_m2q), scale=frame_size)
  ax.grid(False)
  ax.set_xlim(xlim)
  ax.set_ylim(zlim)
  ax.set_zlim([ylim[1], ylim[0]])
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