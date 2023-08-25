# man den dong lai tu tu
import cv2 as cv
import numpy as np


img = cv.imread('images/halloween.jpg',cv.IMREAD_COLOR)

print('shape=', img.shape)

w = img.shape[1]
h = img.shape[0]

out = np.zeros((h,w,3), dtype=np.uint8)
max_y = w-1
max_x = h-1

for i in range(h):
    img[0:i, 0:i, :] = out[max_y-i:max_y, max_x-i:max_x]
    cv.imshow('image', img)
    cv.waitKey(10)

