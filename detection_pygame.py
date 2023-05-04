import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import pygame
import math

# Initialize
pygame.init()

# Create Window/Display
width, height = 1280, 720
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Balloon Pop Game")

# Initialize Clock for FPS
fps = 30
clock = pygame.time.Clock()

# Webcam
cap = cv2.VideoCapture('Videos/crossVideo01.mp4')
cap.set(3, 1280)  # width
cap.set(4, 720)  # height
model = YOLO('runs/detect/train3/weights/best.pt')
classNames = ['crosswalk']

# Images
img_man = pygame.image.load('Masks/man200.png').convert_alpha()
rect_man = img_man.get_rect()
rect_man.x, rect_man.y = width/2, 450

# Variables
speed_x = 10
score = 0
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Main loop
start = True
while start:
    # Get Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            start = False
            pygame.quit()

    # OpenCV
    success, img = cap.read()
    results = model(img, stream=True)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    frame = pygame.surfarray.make_surface(imgRGB).convert()
    frame = pygame.transform.flip(frame, True, False)
    window.blit(frame, (0, 0))
    # pygame.draw.rect(window, (0,255,0), rect_man)
    window.blit(img_man, rect_man)

    blank = np.zeros(img.shape[:2], dtype="uint8")
    canny_image = canny(img)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            d = abs(x2-x1)
            # print('d',d)
            # rectangle = cv2.rectangle(img, (x1, y1), (x2, y2),(255,0,255), 3)
            # pygame.draw.rect(window, (255,0,255), (x1,y1,w,h),3)
            load_image = np.copy(img)
            canny_image = canny(load_image)    
            z = x1+int(w/2)
            y2 = y2-100
            polygons = np.array([[(x1, y2),(z, y1),(x2,y2)]])
            points1 = [x1, y2, z, y1]
            points2 = [z, y1, x2, y2]
            points3 = [z, y2, z, y1+int(h/2)]
            # cv2.line(img,(points1[0],points1[1]),(points1[2],points1[3]), (255,0,0),3)
            # cv2.line(img,(points2[0],points2[1]),(points2[2],points2[3]), (255,0,0),3)

            # l1 = pygame.draw.line(window,(0,255,0),(points1[0],points1[1]),(points1[2],points1[3]),10)
            # l2 = pygame.draw.line(window,(0,255,0),(points2[0],points2[1]),(points2[2],points2[3]),10)
            sensor1 = pygame.draw.line(window,(0,255,0),(points1[0]*2/5,points1[1]),(points1[2]*2/5,400),10)
            sensor2 = pygame.draw.line(window,(0,255,0),(points2[0]*5/3,400),(points2[2],points2[3]),10)
            rect_man.x += speed_x

            if rect_man.left <= 0 or rect_man.right >= width:
                speed_x *= -1

            if rect_man.colliderect(sensor1):
                # l1 = pygame.draw.line(window,(255,0,0),(points1[0],points1[1]),(points1[2],points1[3]),10)
                sensor1 = pygame.draw.line(window,(255,0,0),(points1[0]*2/5,points1[1]),(points1[2]*2/5,400),10)
                speed_x *= -1

                font = pygame.font.Font('Resources/Marcellus-llzw.ttf', 50)
                textScore = font.render(f'Go Right Hand', True, (255, 255, 255))
                window.blit(textScore, (100, 35))

                sound = pygame.mixer.Sound("right.wav")
                sound.play()
            if rect_man.colliderect(sensor2):
                # l2 = pygame.draw.line(window,(255,0,0),(points2[0],points2[1]),(points2[2],points2[3]),10)
                sensor2 = pygame.draw.line(window,(255,0,0),(points2[0]*5/3,400),(points2[2],points2[3]),10)
                speed_x *= -1

                font = pygame.font.Font('Resources/Marcellus-llzw.ttf', 50)
                textScore = font.render(f'Go Left Hand', True, (255, 255, 255))
                window.blit(textScore, (width-300, 35))

                sound = pygame.mixer.Sound("left.wav")
                sound.play()
            
            mask = cv2.fillPoly(blank, polygons, 255)
            masked_image = cv2.bitwise_and(canny_image, mask)
            cv2.imshow("Copped Image", masked_image)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            if (conf > 0.9 ):             
                cv2.putText(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,0,0), thickness=2 )
                
            line_image = np.zeros_like(img)
            minLen = d*4/5
            lines = cv2.HoughLinesP(masked_image, 1, np.pi/180, 200, np.array([]), minLineLength=minLen, maxLineGap=300)

            FinalLines = []
            if lines is not None:
                for line in lines:
                    a1,b1,a2,b2 = line.reshape(4)
                    a1,b1,a2,b2 = int(a1), int(b1), int(a2), int(b2)
                    parameters = np.polyfit((a1,a2),(b1,b2), 1)
                    slope = parameters[0]
                    intercept = parameters[1]
                    # theta = math.degrees(math.atan(slope))
                    dis = int(math.sqrt(a2-a1)**2+(b2-b1)**2)

                    if (abs(slope)<0.1 and d<dis<d*2):
                        FinalLines.append([a1,b1,a2,b2,slope, intercept,dis])
                        FinalLines = sorted(FinalLines, key=lambda x: x[-1], reverse=True)
                        # cv2.line(line_image,(FinalLines[0][0], FinalLines[0][1]),(FinalLines[0][2], FinalLines[0][3]),(0,0,255),10)
                        # cv2.circle(line_image,(FinalLines[0][0], FinalLines[0][1]),15,(0,255,0),10)
                        # cv2.circle(line_image,(FinalLines[0][2], FinalLines[0][3]),15,(0,255,0),10)
                        pygame.draw.line(window,(255,0,255),(FinalLines[0][0], FinalLines[0][1]),(FinalLines[0][2], FinalLines[0][3]),10 )
                        pygame.draw.circle(window,(255,0,255),(FinalLines[0][0], FinalLines[0][1]),15)
                        pygame.draw.circle(window,(255,0,255),(FinalLines[0][2], FinalLines[0][3]),15)
                        font = pygame.font.Font('Resources/Marcellus-llzw.ttf', 50)
                        textScore = font.render(f'Go Straight', True, (255, 255, 255))
                        window.blit(textScore, (width/2-100, 35))

                        sound = pygame.mixer.Sound("go.wav")
                        sound.play()
                        
            
    # line_image = display_lines(img, lines)
    # combo_image = cv2.addWeighted(img, 0.6, line_image, 1, 1)
    # cv2.imshow('results', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update Display
    pygame.display.update()
    # Set FPS
    clock.tick(fps)