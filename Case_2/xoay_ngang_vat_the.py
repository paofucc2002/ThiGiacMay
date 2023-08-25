import cv2 as cv
from cv2 import cvtColor
from cv2 import Canny
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Case_2/images/hinh_dien_thoai.jpg',cv.IMREAD_REDUCED_COLOR_4)

gray = cvtColor(img,cv.COLOR_BGR2GRAY)
filter = cv.GaussianBlur(gray,(5,5), 0)
thresh = cv.threshold(filter,100,255,cv.THRESH_BINARY_INV)[1]

kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)

canny = cv.Canny(opening,250,350,apertureSize=3)

lines = cv.HoughLinesP(canny,1,np.pi/180,50,minLineLength=50,maxLineGap=30)


for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow('Raw',img)
cv.imshow('Canny',canny)
# cv.imshow('filter', filter)
# cv.imshow('thresh', thresh)
# cv.imshow('Opening',opening)
# cv.imshow('xoay', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# Chua xong

