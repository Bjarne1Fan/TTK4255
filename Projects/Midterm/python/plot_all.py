import os 
import sys
from xmlrpc.client import Boolean

import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray

def plot_all(
      all_p                   : ndarray, 
      all_r                   : ndarray, 
      detections              : ndarray, 
      subtract_initial_offset : bool,
      is_task_3               : bool = False
    )->None:
  """
  Tip: The logs have been time-synchronized with the image sequence,
  but there may be an offset between the motor angles and the vision
  estimates. You may optionally subtract that offset by passing True
  to subtract_initial_offset.
  """

  if is_task_3:
    # Transforming the system into a matrix consisting of residuals
    # For task 3, this is brought in with dimension 2*N*M x 1 
    # In our case, it is required to have it in dimension 2*N x M 
    # 
    all_r = all_r.T
    temp = np.zeros((351, 14))
    for row in range(temp.shape[0]):
      temp[row] = all_r[14 * row : 14 * row + 14].reshape((14,))
    all_r = temp.copy()

    #alter all_p to match plotting code
    temp = np.zeros((351, 3))
    for i in range((all_p.shape[0] - 26)//3):
      temp[ i, :] = np.array([ all_p[ 28 + 3*i ], all_p[ 27 + 3*i ], all_p[26 + 3*i ] ])

    all_p = temp.copy()
  #
  # Print reprojection error statistics
  #
  weights = detections[:, ::3]
  reprojection_errors = []

  for i in range(all_p.shape[0]):
    valid = np.reshape( all_r[i], [2,-1] )[:, weights[i,:] == 1]
    reprojection_errors.extend(np.linalg.norm(valid, axis=0))

  reprojection_errors = np.array(reprojection_errors)
  
  print('Reprojection error over whole image sequence:')
  print('- Maximum: %.04f pixels' % np.max(reprojection_errors))
  print('- Minimum: %.04f pixels' % np.min(reprojection_errors))
  print('- Average: %.04f pixels' % np.mean(reprojection_errors))
  print('- Median: %.04f pixels' % np.median(reprojection_errors))
  print('- Idx for maximum error: %i' % np.argmax(reprojection_errors))
  print('- Idx for minimum error: %i' % np.argmin(reprojection_errors))
  print(reprojection_errors.shape)

  #
  # Figure: Reprojection error distribution
  #
  plt.figure(figsize=(8,3))
  plt.hist(reprojection_errors, bins=80, color='k')
  plt.ylabel('Frequency')
  plt.xlabel('Reprojection error (pixels)')
  plt.title('Reprojection error distribution')
  plt.tight_layout()
  # plt.savefig('out_histogram.png')

  #
  # Figure: Comparison between logged encoder values and vision estimates
  #
  logs       = np.loadtxt(os.path.join(sys.path[0], '../data/data/logs.txt'))
  enc_time   = logs[:,0]
  enc_yaw    = logs[:,1]
  enc_pitch  = logs[:,2]
  enc_roll   = logs[:,3]

  vis_yaw = all_p[:,0]
  vis_pitch = all_p[:,1]
  vis_roll = all_p[:,2]
  if subtract_initial_offset:
    vis_yaw -= vis_yaw[0] - enc_yaw[0]
    vis_pitch -= vis_pitch[0] - enc_pitch[0]
    vis_roll -= vis_roll[0] - enc_roll[0]

  vis_fps   = 16
  enc_frame = enc_time*vis_fps
  vis_frame = np.arange(all_p.shape[0])

  _, axes = plt.subplots(3, 1, figsize=[6,6], sharex='col')
  axes[0].plot(enc_frame, enc_yaw, 'k:', label='Encoder log')
  axes[0].plot(vis_frame, vis_yaw, 'k', label='Vision estimate')
  axes[0].legend()
  axes[0].set_xlim([0, vis_frame[-1]])
  axes[0].set_ylim([-1, 1])
  axes[0].set_ylabel('Yaw (radians)')

  axes[1].plot(enc_frame, enc_pitch, 'k:')
  axes[1].plot(vis_frame, vis_pitch, 'k')
  axes[1].set_xlim([0, vis_frame[-1]])
  axes[1].set_ylim([0.0, 0.6])
  axes[1].set_ylabel('Pitch (radians)')

  axes[2].plot(enc_frame, enc_roll, 'k:')
  axes[2].plot(vis_frame, vis_roll, 'k')
  axes[2].set_xlim([0, vis_frame[-1]])
  axes[2].set_ylim([-0.6, 0.6])
  axes[2].set_ylabel('Roll (radians)')
  axes[2].set_xlabel('Image number')
  plt.tight_layout()

  plt.show()
