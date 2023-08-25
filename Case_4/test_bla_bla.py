import cv2 as cv

cap = cv.VideoCapture('Case_4/track2.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    for i in enumerate(frame):
        print(i[0])

    cv.imshow("first frame", frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
