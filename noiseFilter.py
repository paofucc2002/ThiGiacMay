import cv2 as cv
import numpy as np

img = cv.imread('images/noise1.jpg', cv.IMREAD_GRAYSCALE)

cv.imshow('raw', img)
# noise filter
# out3 = cv.blur(img,(9,9))
# cv.imshow('blur filter 9x9', out3)

gout = cv.GaussianBlur(img, (9,9), 0)
cv.imshow('Gaussian filter 9x9', gout)

mout = cv.medianBlur(img, 9)
cv.imshow('median filter', mout)

cv.waitKey(0)
cv.destroyAllWindows