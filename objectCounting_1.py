import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/cell1.jpg',cv.IMREAD_COLOR)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray,5)


# plt.hist(gray.ravel(),256,[0,256]); plt.show()
roi = gray[0:15,0:15]
thresh = np.mean(roi) - 10
# binarize the image
b_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY_INV)[1]
cv.imshow('binary image', b_img)
contours, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print('Số contour tìm được = ',len(contours))
n = 1
for c in contours:
    (x,y),radius = cv.minEnclosingCircle(c)
    center = (int(x),int(y))
    radius = int(radius)
    cv.circle(img,center,radius,(0,255,0),2)
    text = "#"+str(n) # #2
    cv.putText(img, text, center, cv.FONT_HERSHEY_PLAIN, 2, (255,0,255),2 )
    n +=1

cv.imshow("result",img)
cv.waitKey(0)
cv.destroyAllWindows()

