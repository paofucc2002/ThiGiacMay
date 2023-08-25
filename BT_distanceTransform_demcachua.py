import cv2 as cv
from cv2 import GaussianBlur
import numpy as np

# Thêm ảnh
img = cv.imread('images/10.PNG', cv.IMREAD_COLOR)

# Ảnh xám
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Khử nhiễu
fil_img = GaussianBlur(gray,(3,3),0)

# Nhị phân
b_img = cv.threshold(fil_img,140,255,cv.THRESH_BINARY)[1]

cv.imshow('anh mau', img)
cv.imshow('anh xam', gray)
cv.imshow('khu nhieu', fil_img)
cv.imshow('nhi phan', b_img)
cv.waitKey(0)
cv.destroyAllWindows()