import cv2 as cv
import numpy as np

img = cv.imread('images/2cell.jpg', cv.IMREAD_GRAYSCALE)
b_img = cv.threshold(img, 150,255,cv.THRESH_BINARY)[1]

dist_img = cv.distanceTransform(b_img,cv.DIST_L2,3)
cv.normalize(dist_img,dist_img,0,1,cv.NORM_MINMAX)
out = cv.threshold(dist_img,0.8,255,cv.THRESH_BINARY)[1]
print(dist_img.dtype)

cv.imshow('cc Phi', dist_img)
cv.imshow('OUT', out)
cv.waitKey(0)
cv.destroyAllWindows()
