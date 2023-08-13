import cv2
from operator import itemgetter
import numpy as np
from yolo_segmentation import YOLOSegmentation

cap = cv2.VideoCapture("video.mov")

ys = YOLOSegmentation("yolov8m-seg.pt")

# Average color function (pass in box) (returns BGR)
def get_average_color(a):
    # avg_color_per_row = np.average(a, axis=0)
    # avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = np.mean(a, axis=(0,1))
    return avg_color

# Perspective transform function (pass in a point) (returns a point)

# Loop through each frame
while True:
    # Video frame = frame
    ret, frame = cap.read()

    # 2D image = dst
    dst = cv2.imread("dst.jpg")

    if not ret:
        break

    # Copy of frame
    frame2 = np.array(frame)

    # Detect objects
    bboxes, classes, segmentations, scores = ys.detect(frame)

    # Loop through each object
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        # If object is a player
        if class_id == 0:
            # Set corner coordinates for bounding box around player
            (x, y, x2, y2) = bbox
            
            # Draw segmentation around player
            minX = min(seg, key=itemgetter(0))[0]
            maxX = max(seg, key=itemgetter(0))[0]
            maxY = max(seg, key=itemgetter(1))[1]

            # Create smaller rectangle around player to use for color detection
            distLeft = int(abs(seg[0][0] - minX))
            distRight = int(abs(seg[0][0] - maxX))

            newX = int((x2 - x)/3 + x)
            newY = int((y2 - y)/5 + y)
            newX2 = int(2*(x2 - x)/3 + x)
            newY2 = int(2*(y2 - y)/5 + y)

            # Shift based on player orientation
            if(distRight > distLeft):
                # Shift left
                newX = int(newX - ((distLeft + 30)/distRight)/5)
                newX2 = int(newX2 - ((distLeft + 30)/distRight)/5)
            else:
                # Shift right
                newX = int(newX + ((distLeft + 30)/distRight)/5)
                newX2 = int(newX2 + ((distLeft + 30)/distRight)/5)

            roi = frame2[newY:newY2, newX:newX2]

            # Get average color of smaller rectangle
            dominant_color = get_average_color(roi)

            # Get point to draw on 2D image based on the minimum X value (farthest right) and maximum Y value (lowest point)

            # Draw segmentation with the color of the dominant color of the player
            cv2.polylines(frame, [seg], True, dominant_color, 3)
            # Draw smaller box used for color detection
            cv2.rectangle(frame, (newX, newY), (newX2, newY2), dominant_color, 3)

    # Show images
    cv2.imshow("Img", frame)

    # Space to move forward a frame
    key = cv2.waitKey(0)
    # Esc to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()