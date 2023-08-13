import cv2
import numpy as np
from operator import itemgetter
from ultralytics import YOLO
from yolo_segmentation import YOLOSegmentation

cap = cv2.VideoCapture('soccerDavis.mp4')

model = YOLO("yolov8m.pt")
yolo_seg = YOLOSegmentation("yolov8m-seg.pt")

def get_average_color(a):
    # avg_color_per_row = np.average(a, axis=0)
    # avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = np.mean(a, axis=(0,1))
    return avg_color


while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame2 = np.array(frame)
    bboxes, classes, seg, socres = yolo_seg.detect(frame)


    for cls, bbox in zip(classes, bboxes):

        if cls == 0:  
            (x, y, x2, y2) = bbox
            newX = int((x2 - x)/3 + x)
            newY = int((y2 - y)/5 + y)
            newX2 = int(2*(x2 - x)/3 + x)
            newY2 = int(2*(y2 - y)/5 + y)

            roi = frame2[newY:newY2, newX:newX2]
            color = get_average_color(roi)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, "Player", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        if cls == 32:
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 225, ), 2)
            cv2.putText(frame, "Ball", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 225, 0), 2)

            

    #cv2.line(frame, (250, 620), (515, 530), (0, 0, 225), 12)
    cv2.imshow("Img", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

        
cap.release()
cv2.destroyAllWindows() 