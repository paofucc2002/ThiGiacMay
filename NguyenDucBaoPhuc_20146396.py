import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Nhập hình
img = cv.imread('images/Car_license.jpg', cv.IMREAD_REDUCED_COLOR_4)

# anh xam
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Khử nhiễu
filter = cv.GaussianBlur(gray,(5,5), 0)
w = filter.shape[1]
h = filter.shape[0]

#Crop
crop=filter[30:h-30, 30:w-30]

# Nhị phân
b_img = cv.threshold(crop, 150,255, cv.THRESH_BINARY_INV)[1]

# noise removal
kernel = np.ones((3,3),np.uint8)
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
opening = cv.morphologyEx(b_img,cv.MORPH_OPEN,kernel2, iterations = 5)

# xac danh y1 va y2
mean_y = np.mean(opening, axis=1)
y1= np.min(np.where(mean_y>150 & mean_y<200))
y2= np.max(np.where(mean_y>150 & mean_y<200))

print(y1)
print(y2)


# Xuất
# cv.imshow('img', img)
# cv.imshow('xam', gray)
cv.imshow('khu nhieu', filter)
cv.imshow('Nhi phan', b_img)
cv.imshow('noise', opening)
cv.imshow('crop', crop)
# cv.imshow('crop2', crop2)
# cv.imshow('Mor', mor)
# cv.imshow('dist', dist_img)
# cv.imshow('distance', out_dist)
# cv.imshow('ket qua', )
cv.waitKey(0)
cv.destroyAllWindows()
