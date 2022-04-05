import model_reconstruction
import calibrate_camera
import show_results

def task_1_1():
  calibrate_camera.calibrate()
  show_results.show_calibration_results()

def task_1_2():
  calibrate_camera.test_camera_distortion_n_sigma(n=3.0)

def task_2_1():
  model_reconstruction.two_view_reconstruction()

if __name__ == '__main__':
  # Task 1
  task_1_1()
  task_1_2()

  # Task 2
  task_2_1()

  
