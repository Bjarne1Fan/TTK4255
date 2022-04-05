import calibrate_camera
import show_results

def task_1_1():
  calibrate_camera.calibrate()
  show_results.show_calibration_results()

def task_1_2():
  calibrate_camera.test_camera_distortion_n_sigma(n=3.0)

if __name__ == '__main__':
  # Task 1
  task_1_1()
  task_1_2()

  # Task 2

  
