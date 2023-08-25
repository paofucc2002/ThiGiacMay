import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print('can not open video clip/camera')
    exit()

face_cascade = cv.CascadeClassifier()
face_cascade.load('Case_3/humanface.xml')

mlp = MLPClassifier(hidden_layer_sizes=(15,15), max_iter=500)
mlp = pickle.load(open('GenderClassifier.sav','rb'))

while True:
    # read frame by frame
    ret, frame = cap.read()
    if not ret:
        print(' can not read video frame. Video ended?')
        break
    # convert to grayscale
frame = cv.imread(cap, cv.IMREAD_COLOR)

gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# detect human face
faces = face_cascade.detectMultiScale(gray)
# crop the face and resize to 45x60     
for (x,y,w,h) in faces:
    roi = gray[y-int(h/5):y+h+int(h/10),x-int(w/10):x+w+2*int(h/10)]
    cv.rectangle(frame, (x-int(w/10),y-int(h/5)), (x+w+2*int(h/10),y+h+int(h/10)), (0,0,255), 2)
    roi = cv.resize(roi,(45,60))
    # normalize to range [0,1]
    roi = roi.astype(float)
    cv.normalize(roi,roi,0, 1.0, cv.NORM_MINMAX)
    # reshape
    roi = np.reshape(roi,(1,2700))
    # predict
    result = mlp.predict(roi)
    if result == 0:
        text = 'Female'
    else:
        text = 'Male'
    cv.putText(frame, text,(x,y),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    cv.imshow('video', frame)
    # close clip
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
# cv.waitKey(0)
cv.destroyAllWindows()

