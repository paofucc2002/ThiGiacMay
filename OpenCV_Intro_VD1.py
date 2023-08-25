# man den tu tren xuong
import cv2 as cv
import numpy as np


img = cv.imread('images/halloween.jpg',cv.IMREAD_COLOR)

print('shape=', img.shape)

w = img.shape[1]
h = img.shape[0]

zero = np.zeros((1,w,3), dtype=np.uint8)

for i in range(h):
    img[i,:,:] = zero
    cv.imshow('image', img)
    cv.waitKey(10)

