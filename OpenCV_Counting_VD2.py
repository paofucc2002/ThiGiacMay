import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load image file
img = cv.imread('images/pcb.jpg', cv.IMREAD_COLOR) #BGR

# Convert to gray image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

im = np.ones(gray.shape, dtype="uint8") *30
bright_img = cv.subtract(gray,im)


#plot Histogram
#plt.hist(bright_img.ravel(),256,[0,256]); plt.show()

# Binarize the gray image
thresh = 200
b_img = cv.threshold(bright_img,thresh,255,cv.THRESH_BINARY)[1]
# Find contours
contours, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

# Count the number of contours/objects
print('so contours',len(contours))

for index,cnt in enumerate(contours):
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img,[box],0,(0,0,255),2)
    #text = '#' + str(index+1)
    #cv.putText(img, text, center, cv.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)


cv.imshow('image', bright_img)
cv.waitKey(0)
cv.destroyAllWindows()