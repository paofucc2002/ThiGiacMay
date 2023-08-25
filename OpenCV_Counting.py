import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load image file
img = cv.imread('images/sample1.jpg', cv.IMREAD_COLOR) #BGR
# Convert to gray image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#plot Histogram

            #plt.hist(gray.ravel(),256,[0,256]); plt.show()

# Binarize the gray image
thresh = 210
b_img = cv.threshold(gray,thresh,255,cv.THRESH_BINARY_INV)[1]
# Find contours
contours, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

# Count the number of contours/objects
print('so contours',len(contours))

for index,cnt in enumerate(contours):
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv.circle(img,center,radius,(0,255,0),2)
    text = '#' + str(index)
    cv.putText(img, text, center, cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


cv.imshow('image', img)
cv.imshow('binary', b_img)
cv.waitKey(0)
cv.destroyAllWindows()