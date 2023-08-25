import cv2 as cv
import numpy as np

img = cv.imread('VSC.png', cv.IMREAD_REDUCED_GRAYSCALE_8)

cv.imshow('image', img)

cv.waitKey(0)
cv.destroyAllWindows()