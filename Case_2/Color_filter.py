import cv2 as cv
import numpy as np

low_s = 0
high_s = 255
high_v = 255



def low_S(val):
    global low_s
    low_s = val
    b_img = cv.inRange(hsv, (30,low_s,0),(70,high_s,high_v))
    cv.imshow('ColorFilter',b_img)


def high_S(val):
    global high_s
    high_s = val
    b_img = cv.inRange(hsv, (30,low_s,0),(70,high_s,high_v))
    cv.imshow('ColorFilter',b_img)


def high_V(val):
    global high_v
    high_v = val
    b_img = cv.inRange(hsv, (30,low_s,0),(70,high_s,high_v))
    cv.imshow('ColorFilter',b_img)


img = cv.imread('Case_2/images/1.jpg', cv.IMREAD_COLOR)
img = cv.GaussianBlur(img,(5,5),0)

# Chuyển không gian màu sang hsv
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

# Lọc màu với hàm inRange
b_img = cv.inRange(hsv, (30,0,0),(70,255,255))

# Tạo Trackbar điều chỉnh giá trị V
cv.namedWindow('ColorFilter')
cv.createTrackbar('Low_s','ColorFilter',0,255,low_S)
cv.createTrackbar('High_s','ColorFilter',0,255,high_S)
cv.createTrackbar('High_v','ColorFilter',0,255,high_V)


cv.imshow('Raw', img)
cv.imshow('loc mau', b_img)

cv.waitKey(0)
cv.destroyAllWindows()