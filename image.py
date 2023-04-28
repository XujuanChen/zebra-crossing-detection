import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from pygame import mixer

image = cv2.imread('Masks/0.jpg')
# model = YOLO('runs/detect/train/weights/best.pt')
# results = model(img, show=True)

load_image = np.copy(image)

mixer.init()
# mixer.music.load("background.wav")
# mixer.music.play(-1)

# step5. Region of Interest
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

canny_image = canny(load_image)
# cv2.imshow("Canny", canny_image)

def region_of_interest(image):
    # draw a triangle
    polygons = np.array([[(30, 500),(1280, 500), (1280,100),(30,100)]])
    blank = np.zeros_like(image) # the same shape of image filled with zeros
    mask = cv2.fillPoly(blank, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cropped_image = region_of_interest(canny_image)
cv2.imshow("Copped Image", cropped_image)

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


# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=15)
# line_image = display_lines(load_image, lines)
# cv2.imshow("line_image", line_image)

# combine 
# combo_image = cv2.addWeighted(load_image, 0.8, line_image, 1, 1)

# cv2.imshow("line_image", line_image)
# cv2.imshow("combo_image", combo_image)

# cv2.waitKey(0)

cap = cv2.VideoCapture('videos/crossVideo01.mp4')

while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 200, np.array([]), minLineLength=100, maxLineGap=5)
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.6, line_image, 1, 1)
    cv2.imshow('results', combo_image)
    # wait for 1 ms
    # cv.waitKey(1) 
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()