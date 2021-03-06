import sys
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# print('Pillow Version:', PIL.__version__)

image = Image.open(os.path.join(sys.path[0], "grass.jpg"))
print(image.getbands())

# print(image.size)
# print(image.mode)

#image.show()
#image.getchannel('R').show()
#image.getchannel('G').show()
#image.getchannel('G').save("green_task2.jpeg", "jpeg")
#image.getchannel('B').show()

#PIL.ImageOps.invert(image) #Should have used this to invert the image and make the code more userfriendly

img_arr = np.asarray(image)

def pixel_treshold(img_arr, threshold = 110):

    img_arr_green = img_arr[:,:,1]
    img_arr_green = img_arr_green > threshold
    
    tresh_img = Image.fromarray(img_arr_green)
    return img_arr_green

#tresh_img.show()
#tresh_img.save("task2c.jpeg", "jpeg")


red = img_arr[:,:,0]
green = img_arr[:,:,1]
blue = img_arr[:,:,2]

sum_arr = np.sum(img_arr, axis=2)
#ind = np.where(sum_arr == 0)[0]
#ind = sum_arr == 0
#np.put(sum_arr, ind, 1)

def divide_arr(lhs, rhs):
  return np.divide(lhs, rhs, out=np.zeros_like(lhs, dtype=float), where=rhs!=0)

red = divide_arr(red, sum_arr)# np.sum(img_arr, axis=2)
green = divide_arr(green, sum_arr)
blue = divide_arr(blue, sum_arr)

red_img = Image.fromarray(red)
green_img = Image.fromarray(green)
blue_img = Image.fromarray(blue)

# red_img.show()

gs = gridspec.GridSpec(2,4)
gs.update(wspace=0.5)
ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:])
ax3 = plt.subplot(gs[1,1:3])

ax1.imshow(red, cmap="gray")
ax1.set_title("Red")

ax2.imshow(green, cmap="gray")
ax2.set_title("Green")

ax3.imshow(blue, cmap="gray")
ax3.set_title("Blue")

plt.show()

img_arr[:,:,0] = red
img_arr[:,:,1] = green 
img_arr[:,:,2] = blue

# green = pixel_treshold(img_arr, 0.25)

threshold = 0.40
img_arr_green = green > threshold

plt.imshow(img_arr_green, cmap="gray")
plt.show()


