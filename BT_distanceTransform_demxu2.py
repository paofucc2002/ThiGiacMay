import cv2 as cv
from cv2 import GaussianBlur
from matplotlib.pyplot import close
import numpy as np

# Thêm ảnh
img = cv.imread('images/coin2.JPG', cv.IMREAD_REDUCED_COLOR_4)

# Ảnh xám
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Filter
med = cv.medianBlur(gray, 5)

# Nhị phân
b_img = cv.threshold(gray,230,255,cv.THRESH_BINARY_INV)[1]

# Morphology
kernel = np.ones((5,5),np.uint8)
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
kernel4 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
mor_img = cv.morphologyEx(b_img,cv.MORPH_CLOSE,kernel2, iterations = 2)
mor2_img = cv.morphologyEx(mor_img,cv.MORPH_CLOSE,kernel3, iterations = 2)
mor3_img = cv.morphologyEx(mor_img,cv.MORPH_CLOSE,kernel4, iterations = 2)

# distanceTransform
dist_img = cv.distanceTransform(mor2_img,cv.DIST_L2,3)
cv.normalize(dist_img,dist_img,0,1,cv.NORM_MINMAX)
out_dist = cv.threshold(dist_img,0.55,255,cv.THRESH_BINARY)[1]
out_dist = np.uint8(out_dist)

# Countours
contours, hierachy = cv.findContours(out_dist, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Count the number of contours/objects
print('so contours',len(contours))

cv.imshow('anh mau', img)
# cv.imshow('anh xam', gray)
# cv.imshow('filter', med)
cv.imshow('nhi phan', b_img)
# cv.imshow('morphology', mor_img)
cv.imshow('morphology2', mor2_img)
# cv.imshow('morphology3', mor3_img)
cv.imshow('dist img', dist_img)
cv.imshow('dist', out_dist)
cv.waitKey(0)
cv.destroyAllWindows()