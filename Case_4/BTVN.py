import cv2 as cv
import numpy as np
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv.legacy.TrackerCSRT_create,
	"kcf": cv.legacy.TrackerKCF_create,
	"boosting": cv.legacy.TrackerBoosting_create,
	"mil": cv.legacy.TrackerMIL_create,
	"tld": cv.legacy.TrackerTLD_create,
	"medianflow": cv.legacy.TrackerMedianFlow_create,
	"mosse": cv.legacy.TrackerMOSSE_create
}

trackers = cv.legacy.MultiTracker_create()

cap = cv.VideoCapture('Case_4/banh_phuoc.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Object detection from Stable camera
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
   
    if not ret:
        break
    # frame = cv.resize(frame,(750,550))
    # height, width, _ = frame.shape
    # Extract Region of interest
    # frame = frame[340: 700,500: 800]
    # # 1. Object Detection
    
    mask = object_detector.apply(frame)
    _, mask = cv.threshold(mask, 210, 255, cv.THRESH_BINARY)
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
        if area > 600 and area <3000:
            # cv.drawContours(frame,[cnt],-1,(0,0,255),2)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # detections.append([x, y, w, h])
    # print(detections)
    #         # cv.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # boxes_ids = tracker.update(detections)
    # # for box_id in boxes_ids:
    # #     x, y, w, h, id = box_id
    # #     cv.putText(frame, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    # #     cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # # object  
    frame = cv.resize(frame,(1090,600))

    (success, boxes) = trackers.update(frame)
    #print(success,boxes)
    # loop over the bounding boxes and draw then on the frame
    if success == False:
        bound_boxes = trackers.getObjects()
        idx = np.where(bound_boxes.sum(axis= 1) != 0)[0]
        bound_boxes = bound_boxes[idx]
        trackers = cv.legacy.MultiTracker_create()
        for bound_box in bound_boxes:
            trackers.add(trackers,frame,bound_box)
    for i,box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv.putText(frame,"banh "+str(i+1),(x+10,y-3),cv.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)

    cv.imshow('Frame', frame)
    k = cv.waitKey(30)
    if k == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        roi = cv.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
        trackers.add(tracker, frame, roi)

  
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()