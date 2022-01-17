#import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#print('Pillow Version:', PIL.__version__)

image = Image.open('grass.jpg')
print(image.size)
print(image.mode)

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
ind = np.where(sum_arr == 0)[0]
#ind = sum_arr == 0
np.put(sum_arr, ind, 1)

red = red/np.sum(img_arr, axis=2)
green = green/np.sum(img_arr, axis=2)
blue = blue/np.sum(img_arr, axis=2)

red_img = Image.fromarray(red)
green_img = Image.fromarray(green)
blue_img = Image.fromarray(blue)

# red_img.show()

# red_img.save("red_task2d.jpeg", "jpeg")
# green_img.save("green_task2d.jpeg", "jpeg")
# blue_img.save("blue_task2d.jpeg", "jpeg")

# fig0, ax = plt.subplots(2, 2)

# ax[0,0].imshow(red, cmap="gray")

# ax[0,0].set_title("Red")

# ax[0,1].imshow(green, cmap="gray")
# ax[0,1].set_title("Green")

# ax[1,0].imshow(blue, cmap="gray")
# ax[1,0].set_title("Blue")
# plt.show()

img_arr[:,:,0] = red
img_arr[:,:,1] = green 
img_arr[:,:,2] = blue

# green = pixel_treshold(img_arr, 0.25)

threshold = 0.40
img_arr_green = green > threshold

plt.imshow(img_arr_green, cmap="gray")
plt.show()


