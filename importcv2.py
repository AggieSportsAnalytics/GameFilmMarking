import cv2
import numpy as np
from ultralytics import YOLO
from yolo_segmentation import YOLOSegmentation
from operator import itemgetter

cap = cv2.VideoCapture('soccerDavis.mp4')

model = YOLO("yolov8m.pt")
yolo_seg = YOLOSegmentation("yolov8m-seg.pt")

# identifies most common color
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame2 = np.array(frame)

    bboxes, classes, segmentations, scores = yolo_seg.detect(frame)

    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        if class_id == 0:
            (x, y, x2, y2) = bbox
            
            minY = np.max(seg[:, 1])
            bottomVal = int(2*(minY - seg[0][1])/3 + seg[0][1])
            
            a = frame2[seg[0][1]:bottomVal, seg[0][0]:seg[len(seg)-1][0]]
            x = max(seg,key=itemgetter(1))[0]

            cv2.polylines(frame, [seg], True, (0, 0, 225), 2)
            #cv2.rectangle(frame, (seg[0][0], seg[0][1]), (seg[len(seg)-1][0], bottomVal), (225, 0, 0), 2)
            #cv2.rectangle(frame, (seg[0][0], seg[0][1]), (x, bottomVal), (0, 225, 225), 2)

            rect = cv2.minAreaRect(seg)
            box = cv2.boxPoints(rect)
            box2 = box

            box2[0][1] = box[0][1] + ((box[2][1] - box[0][1]) * 0.2)
            box2[1][1] = box[1][1] + ((box[3][1] - box[1][1]) * 0.2)

            

            
            

            box2 = np.int0(box2)

            cv2.drawContours(frame, [box2], 0, (225, 0, 225), 2)

            
            #print(temp)
            #cv2.rectangle(frame, (seg[0][0], seg[0][1]), (seg[][0], bottomVal), (225, 0, 0), 2)
            cv2.putText(frame, str(unique_count_app(a)), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (int(unique_count_app(a)[0]), int(unique_count_app(a)[1]), int(unique_count_app(a)[2])), 4)

    cv2.imshow("Img", frame)

    key = cv2.waitKey(0)
    if key == 27:
        break

        
cap.release()
cv2.destroyAllWindows()