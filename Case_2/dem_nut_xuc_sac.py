import cv2 as cv
import numpy as np

low_s = 0
low_v = 0
high_s = 255
high_v = 255

def low_V(val):
    global low_v
    low_v = val
    b_img = cv.inRange(hsv, (0,low_s,low_v),(255,high_s,high_v))
    cv.imshow('ColorFilter',b_img)

def low_S(val):
    global low_s
    low_s = val
    b_img = cv.inRange(hsv, (0,low_s,low_v),(255,high_s,high_v))
    cv.imshow('ColorFilter',b_img)


def high_S(val):
    global high_s
    high_s = val
    b_img = cv.inRange(hsv, (0,low_s,low_v),(255,high_s,high_v))
    cv.imshow('ColorFilter',b_img)


def high_V(val):
    global high_v
    high_v = val
    b_img = cv.inRange(hsv, (0,low_s,low_v),(255,high_s,high_v))
    cv.imshow('ColorFilter',b_img)

img = cv.imread('Case_2/images/dice_2.jpg',cv.IMREAD_COLOR)
fil = cv.GaussianBlur(img,(3,3),0)

# Chuyển không gian màu sang hsv
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

# Lọc màu với hàm inRange
b_img = cv.inRange(hsv, (15,0,86),(140,90,255))

# kernel = np.ones((3,3),np.uint8)
# opening = cv.morphologyEx(b_img,cv.MORPH_OPEN,kernel, iterations = 1)

# kernel2 = np.ones((2,2),np.uint8)
# opening2 = cv.morphologyEx(b_img,cv.MORPH_OPEN,kernel, iterations = 2)

# Tạo Trackbar điều chỉnh giá trị V
cv.namedWindow('ColorFilter')
cv.createTrackbar('Low_v','ColorFilter',0,255,low_V)
cv.createTrackbar('Low_s','ColorFilter',0,255,low_S)
cv.createTrackbar('High_s','ColorFilter',0,255,high_S)
cv.createTrackbar('High_v','ColorFilter',0,255,high_V)


cv.imshow('RAW',img)
# cv.imshow('fil',fil)
cv.imshow('nhi phan', b_img)
# cv.imshow('open', opening)
# cv.imshow('open2', opening2)
cv.waitKey(0)
cv.destroyAllWindows()
