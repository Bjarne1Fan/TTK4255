import numpy as np

all_r = np.zeros((4914, 1))

temp = np.zeros((2 * 351, 7))
for row in range(temp.shape[0]):
  temp[row] = all_r[7 * row : 7 * row + 7].reshape((7,))
all_r = temp.copy()
print(all_r)
print(all_r.shape)
