import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import plotting 

def show_calibration_results():
  folder = os.path.join(sys.path[0], '../data/hw5_ext/calibration')

  K                       = np.loadtxt(join(folder, 'K.txt'))
  distortion_coefficients = np.loadtxt(join(folder, 'dc.txt'))
  std_int                 = np.loadtxt(join(folder, 'std_int.txt'))
  u_all                   = np.load(join(folder, 'u_all.npy'))
  image_size              = np.loadtxt(join(folder, 'image_size.txt')).astype(np.int32) # height,width
  mean_errors             = np.loadtxt(join(folder, 'mean_errors.txt'))

  # Extract components of intrinsics standard deviation vector. See:
  # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
  fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy = std_int

  print()
  print('Calibration results')
  print('================================')
  print('Focal length and principal point')
  print('--------------------------------')
  print('fx:%13.5f +/- %.5f' % (K[0,0], fx))
  print('fy:%13.5f +/- %.5f' % (K[1,1], fy))
  print('cx:%13.5f +/- %.5f' % (K[0,2], cx))
  print('cy:%13.5f +/- %.5f' % (K[1,2], cy))
  print()
  print('Distortion coefficients')
  print('--------------------------------')
  print('k1:%13.5f +/- %.5f' % (distortion_coefficients[0], k1))
  print('k2:%13.5f +/- %.5f' % (distortion_coefficients[1], k2))
  print('k3:%13.5f +/- %.5f' % (distortion_coefficients[4], k3))
  print('p1:%13.5f +/- %.5f' % (distortion_coefficients[2], p1))
  print('p2:%13.5f +/- %.5f' % (distortion_coefficients[3], p2))
  print('--------------------------------')
  print()
  print('The number after "+/-" is the standard deviation.')
  print()

  # Tip: See the "image_names.txt" file in the output folder
  # for the relationship between index and image name.
  plt.figure(figsize=(8,4))
  plt.subplot(121)
  plt.bar(range(len(mean_errors)), mean_errors)
  plt.title('Mean error per image')
  plt.xlabel('Image index')
  plt.ylabel('Mean error (pixels)')

  plt.subplot(122)
  for i in range(u_all.shape[0]):
    plt.scatter(u_all[i, :, 0, 0], u_all[i, :, 0, 1], marker='.')
  plt.axis('image')
  plt.xlim([0, image_size[1]])
  plt.ylim([image_size[0], 0])
  plt.xlabel('u (pixels)')
  plt.ylabel('v (pixels)')
  plt.title('All corner detections')
  plt.tight_layout()
  plt.show()
  

def show_localization_results():
  model = os.path.join(sys.path[0], '../example_localization')
  query = os.path.join(sys.path[0], '../example_localization/query/IMG_8210')

  # 3D points [4 x num_points].
  X = np.loadtxt(f'{model}/X.txt')

  # Model-to-query transformation.
  # If you estimated the query-to-model transformation,
  # then you need to take the inverse.
  T_m2q = np.loadtxt(f'{query}_T_m2q.txt')

  # If you have colors for your point cloud model...
  colors = np.loadtxt(f'{model}/c.txt') # RGB colors [num_points x 3].
  # ...otherwise...
  # colors = np.zeros((X.shape[1], 3))

  # These control the visible volume in the 3D point cloud plot.
  # You may need to adjust these if your model does not show up.
  xlim = [-10,+10]
  ylim = [-10,+10]
  zlim = [0,+20]

  frame_size = 1
  marker_size = 5

  plt.figure('3D point cloud', figsize=(6,6))
  plotting.draw_point_cloud(
    X, T_m2q, xlim, ylim, zlim, 
    colors=colors, marker_size=marker_size, frame_size=frame_size
  )
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  show_calibration_results()
  show_localization_results()
