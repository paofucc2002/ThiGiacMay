import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('images/Salt_noise.jpg', cv.IMREAD_GRAYSCALE) #BGR
b_img = cv.threshold(img,150,255,cv.THRESH_BINARY)[1]

# Morphology demo
# Create Kernel
kernel = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]], dtype=np.uint8)

kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
# kernel3 = cv.getStructuringElement(cv.MORPH_CROSS,(3,3),(2,2))

# out = cv.erode(b_img,kernel2, iterations=3)
# out2 = cv.erode(b_img,kernel3, iterations=3)
out = cv.morphologyEx(b_img, cv.MORPH_OPEN, kernel, iterations = 2)

# Show image
cv.imshow('raw', img)
cv.imshow('After Morphology', out)
# cv.imshow('After Morphology 2', out2)
cv.waitKey(0)
cv.destroyAllWindows()