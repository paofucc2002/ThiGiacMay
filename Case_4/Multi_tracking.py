# import numpy as np
import cv2 as cv
import numpy as np

cap = cv.VideoCapture('Case_4/rio2016.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
# Create MultiTracker object
multiTracker = cv.legacy.MultiTracker_create()

# # Load 1st frame
ret, frame = cap.read()
# frame = cv.resize(frame,(320,240))
# detect objects location
bboxes = []
colors = [] 
while True:
# draw bounding boxes over objects
# selectROI's default behaviour is to draw box starting from the center
# when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
        break
print('Selected bounding boxes {}'.format(bboxes))

# r = np.array([x,y,w,h])
# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(cv.legacy.TrackerCSRT.create(), frame, bbox)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.resize(frame,(320,240))
    ret, objects = multiTracker.update(frame)
    if ret:
        for i, obj in enumerate(objects):
            p1 = (int(obj[0]),int(obj[1]))
            p2 = (int(obj[0] + obj[2]),int(obj[1] + obj[3]))
            cv.rectangle(frame, p1,p2, colors[i], 2, 1)
    else:
        print("tracking fail")
        # detect object

    cv.imshow("first frame", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

