import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Nhập hình
img = cv.imread('images/coin.png', cv.IMREAD_COLOR)

# anh xam
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Khử nhiễu
filter = cv.GaussianBlur(gray,(3,3), 0)

# Nhị phân
b_img = cv.threshold(filter, 150,255, cv.THRESH_BINARY_INV)[1]

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(b_img,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

# distanceTransform
dist_img = cv.distanceTransform(opening,cv.DIST_L2,3)
cv.normalize(dist_img,dist_img,0,1,cv.NORM_MINMAX)
out_dist = cv.threshold(dist_img,0.65,255,cv.THRESH_BINARY)[1]

out_dist = np.uint8(out_dist)
unknown = cv.subtract(sure_bg,out_dist)

# Watershed

# Mor 2
# kernel2 = np.ones((3,3),np.uint8)
# kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
# mor2 = cv.dilate(out_dist,kernel3,iterations = 3)


# Countours
contours, hierachy = cv.findContours(out_dist, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Count the number of contours/objects
print('so contours',len(contours))

# Marker labelling
ret, markers = cv.connectedComponents(out_dist)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [0,255,0]

for index,cnt in enumerate(contours):
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    text = '#' + str(index+1)
    cv.putText(img, text, center, cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

# Xuất
cv.imshow('img', img)
# cv.imshow('khu nhieu', filter)
cv.imshow('Nhi phan', b_img)
# cv.imshow('Mor', mor)
# cv.imshow('dist', dist_img)
# cv.imshow('distance', out_dist)
# cv.imshow('ket qua', )
cv.waitKey(0)
cv.destroyAllWindows()
