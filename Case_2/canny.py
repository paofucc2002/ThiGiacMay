import cv2 as cv
import numpy as np

img = cv.imread('Case_2/images/1.jpg', cv.IMREAD_COLOR)

canny = cv.Canny(img,250,350)
lines = cv.HoughLinesP(canny,1,np.pi/180,50,minLineLength=100,maxLineGap=30)

for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow('Raw', img)
cv.imshow('Canny', canny)
cv.waitKey(0)
cv.destroyAllWindows