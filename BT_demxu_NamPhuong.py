import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/coin.png',cv.IMREAD_COLOR)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray,(9,9),1)
w = img.shape[1]
h = img.shape[0]


# plt.hist(gray.ravel(),256,[0,256]); plt.show()
thresh=115
# binarize the image
b_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY_INV)[1]

kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
# kernel_ci = np.array([[0,0,1,0,0],
#                      [0,1,1,1,0],
#                      [1,1,1,1,1],
#                      [0,1,1,1,0],
#                      [0,0,1,0,0]], dtype=np.uint8)
# kernel_sq =  np.array([[1,1,1],
#                       [1,1,1],
#                       [1,1,1]], dtype=np.uint8)
# new = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 9, -5)
out = cv.morphologyEx(b_img,cv.MORPH_ERODE,kernel,iterations=12)

dist_img=cv.distanceTransform(out,cv.DIST_L2,3)
cv.normalize(dist_img,0,1,cv.NORM_MINMAX)
fresh=cv.threshold(dist_img,0.1,1,cv.THRESH_BINARY)[1]

contours, hierachy = cv.findContours(fresh.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print('a= ',len(contours))
for i,c in enumerate(contours): 
    #print(type(c))
    #print(c.shape)
    #contours area
    Area = cv.contourArea(c)
    if Area >100 and Area <5000:
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text = '#' + str(i+1)
        cv.putText(img, text, (x,y) , cv.FONT_HERSHEY_COMPLEX,1,(255,0,100),1)

dist_img=cv.distanceTransform(out,cv.DIST_L2,3)
cv.normalize(dist_img,0,1,cv.NORM_MINMAX)
fresh=cv.threshold(dist_img,0.1,1,cv.THRESH_BINARY)[1]
cv.imshow('binary image',fresh)
cv.imshow("result",img)
cv.waitKey(0)
cv.destroyAllWindows()