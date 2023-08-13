import cv2
import numpy as np
from operator import itemgetter
from ultralytics import YOLO
from yolo_segmentation import YOLOSegmentation

cap = cv2.VideoCapture('goal.mov')

model = YOLO("yolov8m.pt")
yolo_seg = YOLOSegmentation("yolov8m-seg.pt")

def get_average_color(a):
    # avg_color_per_row = np.average(a, axis=0)
    # avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = np.mean(a, axis=(0,1))
    return avg_color

file1 = open(r"output.txt", "w")
counter = 0
test = False
while True:
    counter = counter + 1
    ret, frame = cap.read()
    if not ret:
        break

    frame2 = np.array(frame)

   

    bboxes, classes, seg, scores = yolo_seg.detect(frame)
    counter1 = 0
    for cls, bbox in zip(classes, bboxes):

        if cls == 0:  
            (x, y, x2, y2) = bbox
            newX = int((x2 - x)/3 + x)
            newY = int((y2 - y)/5 + y)
            newX2 = int(2*(x2 - x)/3 + x)
            newY2 = int(2*(y2 - y)/5 + y)

            roi = frame2[newY:newY2, newX:newX2]
            color = get_average_color(roi)
            #print(color)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, str(scores[0]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        
        if x < 400:
            counter1 = counter1 + 1
        if counter1 > 6 and test == False:
            file1.write(str(counter))
            test = True

            #print(scores)
            
            
        


    cv2.imshow("Img", frame)

    key = cv2.waitKey(10)
    if key == 27:
        break
file1.close()
        
cap.release()
cv2.destroyAllWindows()