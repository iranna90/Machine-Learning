import numpy as np

image = np.ones((28, 28, 3))

print image
image[:, :, :] = [1, 0, 1]
# from 5,5 to 20 height and 20 width make white
row = 5
column = 5
height = 4
width = 5
image[row:row + height, column:column + width, :] = [0, 0, 0]

from matplotlib import pyplot as plt, cm

# plt.imshow(image)

from skimage import data

coins = data.coins()
print coins.shape
print coins.dtype
print coins.max()
print coins.min()

# plt.imshow(image, interpolation='nearest')

from scipy import misc

data = misc.imread('cat.jpeg')

print type(data)
print data.shape
print data.dtype


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


only_red_value = data.copy()
only_green_value = data.copy()
only_blue_value = data.copy()
only_red_value[:, :, 1:] = [0, 0]
only_green_value[:, :, (0, 2)] = [0, 0]
only_blue_value[:, :, :2] = [0, 0]
gray_scale = data.copy()
gray_scale = rgb2gray(gray_scale)
f, axarr = plt.subplots(3, 2)

print("particular point(175,60) at main {} , only red {}, only green {}, only blue {}".format(
    data[175, 60, :],
    only_red_value[175, 60, :],
    only_green_value[175, 60, :],
    only_blue_value[175, 60, :]))
axarr[0, 0].imshow(data)
axarr[0, 1].imshow(gray_scale, cmap=cm.gray)
axarr[1, 0].imshow(only_red_value)
axarr[1, 1].imshow(only_green_value)
axarr[2, 0].imshow(only_blue_value)
checking = data.copy()
checking = checking[:, :, 2]
axarr[2, 1].imshow(checking)

plt.show()
