# import numpy as np
import cv2 as cv
import time
import numpy as np
import math

# from object_detection import ObjectDetection
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv.legacy.TrackerCSRT_create, 
}

trackers = cv.legacy.MultiTracker_create()
# od = ObjectDetection()
center_point=[]
center_point_pre=[]
g=[]
h=[]
tracking_object={}
track_id=0
count=0
cap = cv.VideoCapture("Case_4/pingpong_cut_ez.mp4")

# Object detection from Stable camera
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
fps=100
prev=0


while True: 
    times=time.time() - prev
    if times>1./fps:
        prew=time.time()
    
    
    ret, frame = cap.read()
   
    if not ret:
        break

    
    frame = cv.resize(frame,(750,550))
    height, width, _ = frame.shape
    # Extract Region of interest
    # frame = frame[340: 700,500: 800]
    
    # # 1. Object Detection
    mask = object_detector.apply(frame)
    
    _, mask = cv.threshold(mask, 160, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections=[]
    for cnt in contours:
        # area = cv.contourArea(cnt)
        # if area > 200 and area < 100000:
        #     cv.drawContours(frame, [cnt], -1, (0, 0, 255), 1)
        #     x,y,w,h = cv.boundingRect(cnt)
        # detections.append([x, y, w, h])
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        if area > 200 and area < 1000:
            # cv.drawContours(frame,[cnt],-1,(0,0,255),2)
            x, y, w, h = cv.boundingRect(cnt)
            detections.append([x, y, w, h])
            cx=int((x+x+w)/2)
            cy=int((y+y+h)/2)
            center_point.append((cx,cy))
            cv.circle(frame,(cx,cy),5,(0,0,255),-1)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
     
    center_point_pre=center_point.copy()
    
    if count <= 2:
        for h in center_point:
            for g in center_point_pre:
                distance = math.hypot(g[0] - h[0], g[1] - h[1])
                if distance > 1:
                    tracking_object[track_id] = h
                    track_id += 1
    else:
        tracking_objects_copy = tracking_object.copy()
        for object_id, g in tracking_objects_copy.items():
            object_exists = False
            for h in center_point:
                distance = math.hypot(g[0] - h[0], g[1] - h[1])
                # Update IDs position
                if distance > 1 :
                    tracking_object[object_id] = h
                    object_exists = True
                    if h in center_point:
                        center_point.remove(h)
                    continue
            # Remove IDs lost
            if not object_exists:
                tracking_object.pop(object_id)
        for pt in center_point:
            tracking_object[track_id] = h
            track_id += 1
        for object_id,h in tracking_object.items():
            cv.circle(frame,h,5,(0,0,255),-1)
            cv.putText(frame,str(object_id),h[0],h[1]-7,8,1,(0,0,255),2)


    # (success, boxes) = trackers.update(frame)
    # if success == False:   
    #     bound_boxes = trackers.getObjects()
    #     idx = np.where(bound_boxes.sum(axis= 1) != 0)[0]
    #     bound_boxes = bound_boxes[idx]
        
    #     trackers = cv.legacy.MultiTracker_create()
    #     for bound_box in bound_boxes:
    #         trackers.add(trackers,frame,bound_box)
    # for i,box in enumerate(h):
    #     (x, y, w, h) = [int(v) for v in box]
    #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     chu = "Lable"
    #     cv.putText(frame,chu + str(i+1),(x+10,y-3),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

    cv.imshow('Frame', frame)
    k = cv.waitKey(30)
    if k == ord("s"):

        roi = cv.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)

        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
        trackers.add(tracker, frame, roi)

  
    # cv.imshow("1", mask)
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()