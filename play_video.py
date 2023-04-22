import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math
import numpy as np

cap = cv2.VideoCapture("videos/crossVideo01.mp4")
model = YOLO('runs/detect/train/weights/best.pt')
classNames = ['crosswalk']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cv2.rectangle(img, (x1, y1), (x2, y2),(255,0,255),3 )
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            # print(box.cls)
            if (conf > 0.5 ):
                cv2.putText(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,0,0), thickness=2 )

    cv2.imshow("Image", img)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

