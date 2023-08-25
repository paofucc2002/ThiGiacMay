import cv2 as cv
import numpy as np


# open video clip
cap = cv.VideoCapture('Case_4/pingpong_cut_ez.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

tracking = 0

def ball_detect(frame):
    # convert to hsv
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # yellow filter with inRange function
    low_H = 25
    high_H = 35
    low_S = 0
    high_S = 100
    low_V = 50
    high_V = 255
    mask = cv.inRange(hsv,np.array([low_H, low_S, low_V]),np.array([high_H,high_S,high_V]))

    # noise remove with morphology (optional)
    kernel_ci =  np.array([[0,0,1,0,0],
                        [0,1,1,1,0],
                       [1,1,1,1,1],
                       [0,1,1,1,0],
                       [0,0,1,0,0]], dtype=np.uint8)
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel_ci,iterations=2)
    # find contours
    contours, hierachy = cv.findContours(mask,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # detect circle
    min_radius = 30
    max_radius = 60
    balls = np.empty((0, 4),dtype=np.uint8)
    for c in contours:
        (x,y),radius = cv.minEnclosingCircle(c)
        radius = int(radius)
        # print(radius)
        if (radius > min_radius) and (radius < max_radius):
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            balls = np.vstack((balls,np.array([x,y,w,h])))
    cv.imshow("bbox",frame)
    # print(balls.shape)
    return balls

# read each frame
while True:
    # read next frame
    ret, frame = cap.read()
    if tracking == 0:
        # detect ball 
        balls = ball_detect(frame)
        # if 2 balls are detected -> tracking = 1
        print(balls.shape)
        if balls.shape[0] == 2:
            tracking = 1
    else:
        # get balls ROIs
        
        # update tracker   
        
        # Draw bounding boxes on image   
        
        # if tracking fail -> detect
        # if balls.shape[0] < 2:
        #     tracking = 0
        pass
            


    # cv.imshow("first frame", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()