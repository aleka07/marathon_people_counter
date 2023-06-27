import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from tracker import *

model = YOLO('yolov8x.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#In the input folder there is a link to download the video
video_path = os.path.join('.', 'input', 'marathon.mp4')
video_out_path = os.path.join('.', 'output', 'marathon_out.mp4')

cap = cv2.VideoCapture(video_path)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()
cy1 = 669
offset = 6
counter = []

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec (e.g., 'XVID' for AVI or 'mp4v' for MP4)
output_video = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1280, 720))

    results = model.predict(frame)

    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a.numpy()).astype("float")

    list = []         
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        
        if cy1 < (y4 + offset) and cy1 > (y4 - offset):
            if counter.count(id) == 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 200), 1)
                cv2.circle(frame, (cx, y4), 3, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                counter.append(id)

    cv2.line(frame, (362, cy1), (953, cy1), (0, 255, 0), 2)
    l = len(counter)
    cv2.putText(frame, f'People: {l}', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("RGB", frame)
    output_video.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Writing the total count into a txt file
with open("count.txt", "w") as file:
    file.write(f"The total count of persons is {len(counter)}.\n")

cap.release()
output_video.release()
cv2.destroyAllWindows()
