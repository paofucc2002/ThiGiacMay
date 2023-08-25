import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/noise1.jpg',cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Filter
out = cv.blur(gray, (3,3))
out2= cv.blur(gray,(7,7))
out3= cv.GaussianBlur(gray,(3,3), 0)
out4= cv.GaussianBlur(gray,(7,7), 0)
out5 = cv.medianBlur(gray,3)
out6 = cv.medianBlur(gray,7)

# k = cv.getGaussianKernel(7,5)
# print(k)

cv.imshow('image', gray)
# cv.imshow('filtered', out)
# cv.imshow('7x7', out2)
cv.imshow('Gaus 3x3' ,out3)
cv.imshow('Gaus 7x7' ,out4)
cv.imshow('Median 3' ,out5)
cv.imshow('Median 7' ,out6)
cv.waitKey(0)
cv.destroyAllWindows()