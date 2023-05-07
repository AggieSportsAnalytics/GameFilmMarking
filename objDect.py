import cv2
from ultralytics import YOLO
import numpy as np

def unique_count_app(a, pixel):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    for x in 1:20:
        color.remove("")
    for x in 1:20:
        
    return colors[count.argmax()]


cap = cv2.VideoCapture("soccerDavis.mp4")

model = YOLO("yolov8m.pt")

#while True:
ret, frame = cap.read()
#if not ret:
    #   break

frame2 = np.array(frame)
rows, col, temp = frame2.shape
pixel = frame[rows - 10, 10]
results = model(frame, device="mps")
result = results[0]
bboxes = np.array(result.boxes.xyxy.cpu(), dtype = "int")
classes = np.array(result.boxes.cls.cpu(), dtype = "int")
for cls, bbox in zip(classes, bboxes):
    (x, y, x2, y2) = bbox
    if cls == 0:
       # pixel = frame[(x - x2), (y - y2)]
        #print("x: " + str(x2 - x) + "y: " + str(y2-y))
        #print(pixel)

        a = frame2[y:y2, x:x2]
        #cv2.rectangle(frame, (int((x+x2)/2) + 3, int((y+y2)/2) +3), (int((x+x2)/2) - 3, int((y+y2)/2)- 3), (225, 0, 225), 2)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(unique_count_app(a, pixel)), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
    if cls == 32:
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 225, ), 2)
        cv2.putText(frame, "Ball", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 225, 0), 2)



cv2.imshow("Imp", frame)
key = cv2.waitKey(0)
#if key == 27:
   #break
cap.release()
cv2.destroyAllWindows()

