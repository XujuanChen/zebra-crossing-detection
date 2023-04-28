import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math
import numpy as np
import matplotlib.pyplot as plt
from pygame import mixer

mixer.init()

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    # blank image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope <0.1 and slope >-0.1:
                cv2.line(line_image, (x1,y1),(x2,y2), (0,0,255), 10) # draw lines
                sound = mixer.Sound("walk.wav")
                sound.play()
    return line_image

cap = cv2.VideoCapture("videos/crossVideo01.mp4")
model = YOLO('runs/detect/train3/weights/best.pt')
classNames = ['crosswalk']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    blank = np.zeros(img.shape[:2], dtype="uint8")
    canny_image = canny(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            rectangle = cv2.rectangle(img, (x1, y1), (x2, y2),(255,0,255), 3 )
            load_image = np.copy(img)
            canny_image = canny(load_image)
            polygons = np.array([[(x1, y2),(x1+int(w/2), y1),(x2,y2)]])
            cv2.polylines(img, polygons, True, (0,0,255), 3)
            mask = cv2.fillPoly(blank, polygons, 255)
            masked_image = cv2.bitwise_and(canny_image, mask)
            cv2.imshow("Copped Image", masked_image)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            # print(box.cls)
            if (conf > 0.9 ):             
                cv2.putText(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,0,0), thickness=2 )
                lines = cv2.HoughLinesP(masked_image, 1, np.pi/180, 200, np.array([]), minLineLength=100, maxLineGap=5)
                line_image = display_lines(img, lines)
                combo_image = cv2.addWeighted(img, 0.6, line_image, 1, 1)
    cv2.imshow('results', combo_image)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()