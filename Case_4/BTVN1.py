import cv2 as cv
import numpy as np


# open video clip
cap = cv.VideoCapture('Case_4/banh_phuoc.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

tracking = 0

def khung_hinh():
    fps = cap.read()
    for i in fps:
        
    return i

def ball_detect(frame):
    # convert to hsv
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[240:480,0:854]
    # print(roi.shape)

    # Chuyển không gian màu sang hsv
    hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

    # Lọc màu với hàm inRange
    mask = cv.inRange(hsv, (30,100,100),(60,200,200))
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    balls = np.empty((0, 4),dtype=np.uint8)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        if area > 400:
            # cv.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            balls = np.vstack((balls,np.array([x,y,w,h])))
    cv.imshow("bbox",frame)
    cv.imshow("mask", mask)
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