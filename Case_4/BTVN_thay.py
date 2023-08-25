import cv2 as cv
import numpy as np

while True:
    ret, frame = cap.read()
    if not ret:
        print("cant receive frame")
        break

    if ret:
        p1 = (int(obj[0]),int(obj[1]))
        p2 = (int(obj[0]+obj[2]),int(obj[1]+obj[3]))
        cv.rectangle(frame, p1, p2, (255,0,0), 2,1)

    else:
        print("tracking fail")

    cv.imshow("afadsf", frame)
    k=cv.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv.destroyAllWindows()