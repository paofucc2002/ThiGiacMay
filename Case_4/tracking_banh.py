# Danh sách nhóm:
# 	-Nguyễn Đức Bảo Phúc - 20146396 (Lớp thứ 6 - Tiết 7,8,9,10)
# 	-Nguyễn Tấn Phát - 20146389 (Lớp thứ 6 - Tiết 7,8,9,10)
# 	-Lương Tuấn Phi - 20146392 (Lớp thứ 6 - Tiết 7,8,9,10)
# 	-Điền Nguyễn Hữu Phước - 20146399 (Lớp thứ 7 - Tiết 2,3,4,5)

import cv2
import math

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("Case_4/banh_phuoc.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[240:480,0:854]
    # print(roi.shape)

    # Chuyển không gian màu sang hsv
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    # Lọc màu với hàm inRange
    mask = cv2.inRange(hsv, (30,100,100),(60,200,200))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 400:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        chu = "banh "
        cv2.putText(roi,chu+str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 3)

    # cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()