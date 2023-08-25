import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier

# import picture
img = cv.imread('Case_3/images/number.PNG',cv.IMREAD_COLOR)

# convert to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Filter
gray = cv.GaussianBlur(gray,(3,3),1)

# Binary
# plt.hist(gray.ravel(),256,[0,256]); plt.show()
thresh=150
b_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY_INV)[1]

mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=1000)
mlp = pickle.load(open('MNISTsave.sav','rb'))

contours, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print('so contours = ',len(contours))
for i,c in enumerate(contours): 
    x,y,w,h = cv.boundingRect(c)
    print(x)
    print(y)
    print(w)
    print(h)
    roi = b_img[y-10:y+h+10,x-10:x+w+10]
    roi = cv.resize(roi,(28,28))
    # normalize to range [0,1]
    roi = roi.astype(np.float32)
    cv.normalize(roi,roi,0, 1, cv.NORM_MINMAX)
    # print(roi)
    # reshape
    roi = np.reshape(roi,(1,784))
    # predict
    result = mlp.predict(roi)
    if result == 0:
        text = '0'
    elif result == 1:
        text = '1'
    elif result == 2:
        text = '2'
    elif result == 3:
        text = '3'
    elif result == 4:
        text = '4'
    elif result == 5:
        text = '5'
    elif result == 6:
        text = '6'
    elif result == 7:
        text = '7'
    elif result == 8:
        text = '8'
    else:
        text = '9'
    print('hinh so',i,'ket qua',result)
    cv.rectangle(img,(x-10,y-10),(x+w+10,y+h+10),(0,0,255),2)
    cv.putText(img, text,(x,y+30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

# roi = gray[261-10:261+108+10,217-10:217+70+10]
# roi = cv.resize(roi,(28,28))
# test = b_img[261-10:261+108+10,217-10:217+70+10]


cv.imshow("result",img)
# cv.imshow("crop",test)
cv.waitKey(0)
cv.destroyAllWindows()
