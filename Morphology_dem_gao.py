import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load image file
img = cv.imread('images/rice.jpg', cv.IMREAD_COLOR) #BGR

# Convert to gray image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

im = np.ones(gray.shape, dtype="uint8") *20
bright_img = cv.add(gray,im)

# Filter
# f_img= cv.GaussianBlur(gray,(3,3), 0)
f_img = cv.medianBlur(gray, 9)


# Binarize the gray image
thresh = 140
b_img = cv.threshold(bright_img,thresh,255,cv.THRESH_BINARY)[1]

#plot Histogram
# plt.hist(f_img.ravel(),256,[0,256]); plt.show()

# Morphology demo
# Create Kernel
kernel = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]], dtype=np.uint8)

kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,1))

out = cv.erode(b_img,kernel3, iterations=1)
# out2 = cv.erode(b_img,kernel3, iterations=3)
out2 = cv.morphologyEx(out, cv.MORPH_OPEN, kernel2, iterations = 2)

contours, hierachy = cv.findContours(out2, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

# Count the number of contours/objects
print('so contours',len(contours))

for index,cnt in enumerate(contours):
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv.circle(img,center,radius,(0,255,0),2)
    # text = '#' + str(index+1)
    # cv.putText(img, text, center, cv.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

# Show image
cv.imshow('raw', img)
# cv.imshow('ffilter', f_img)
cv.imshow('Binary', b_img)
# cv.imshow('Bright', bright_img)
cv.imshow('Mor', out)
cv.imshow('Mor 2', out2)
cv.waitKey(0)
cv.destroyAllWindows()