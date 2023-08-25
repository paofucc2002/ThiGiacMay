import cv2 as cv
import numpy as np

img = cv.imread('images/anchor.png', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

b_img = cv.threshold(gray,150,255,cv.THRESH_BINARY)[1]

# create kernel/element
kernel_sq =  np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]], dtype=np.uint8)
kernel_sq5x5 = np.ones((5,5),dtype=np.uint8)
kernel_cr = np.array([[0,1,0],
                      [1,1,1],
                      [0,1,0]], dtype=np.uint8)
kernel_ci = np.array([[0,0,1,0,0],
                     [0,1,1,1,0],
                     [1,1,1,1,1],
                     [0,1,1,1,0],
                     [0,0,1,0,0]], dtype=np.uint8)
# morphology
# out = cv.erode(b_img,kernel_sq,iterations=2)
# out = cv.dilate(out,kernel_sq,iterations=2)
out = cv.morphologyEx(b_img,cv.MORPH_ERODE,kernel_sq5x5,iterations=15)


cv.imshow('result',out)
cv.waitKey(0)
cv.destroyAllWindows()