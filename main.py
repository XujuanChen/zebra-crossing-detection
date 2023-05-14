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
pygame.display.set_caption("Crosswalk Detection")

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

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines, sensor1, sensor2, rect_man):
    # blank image
    line_image = np.zeros_like(image)
    longLines = []

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2),(y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            dis = int(math.sqrt(x2-x1)**2+(y2-y1)**2)

            if -0.1<slope <0.1:
                longLines.append([x1,y1,x2,y2,slope, intercept,dis])
                longLines = sorted(longLines, key=lambda x: x[-1], reverse=True)
                longLines = longLines[:1]

                pl = pygame.draw.line(window,(0,255,0),(longLines[0][0], longLines[0][1]),(longLines[0][2], longLines[0][3]),10 )
                c1 = pygame.draw.circle(window,(0,255,0),(longLines[0][0], longLines[0][1]),15)
                c2 = pygame.draw.circle(window,(0,255,0),(longLines[0][2], longLines[0][3]),15)

                # Play "Go Straight" when the person walking on the lines
                if (pl.colliderect(rect_man)):
                    font = pygame.font.Font('Resources/Marcellus-llzw.ttf', 50)
                    text = font.render(f'Go Straight', True, (255, 255, 255))
                    window.blit(text, (width/2-100, 35))
                
                # only play sound when the line touched one sensor, otherwise too noisy.
                if (pl.colliderect(sensor1) or pl.colliderect(sensor2)):
                    pygame.draw.line(window,(255,0,255),(longLines[0][0], longLines[0][1]),(longLines[0][2], longLines[0][3]),10 )
                    pygame.draw.circle(window,(255,0,255),(longLines[0][0], longLines[0][1]),15)
                    pygame.draw.circle(window,(255,0,255),(longLines[0][2], longLines[0][3]),15)
                    sound = pygame.mixer.Sound("go.wav")
                    sound.play()
    return line_image

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
    pygame.draw.rect(window, (0,255,0), rect_man)
    window.blit(img_man, rect_man)


    # show display screen for the warnings
    BLUE = (0, 0, 255, 150)  # This color contains an extra integer. It's the alpha value.
    size = (width, 125)
    blue_image = pygame.Surface(size, pygame.SRCALPHA)  # Contains a flag telling pygame that the Surface is per-pixel alpha
    # For the 'blue_image' it's the alpha value of the color that's been drawn to each pixel that determines transparency.
    pygame.Surface.fill(blue_image,BLUE,(0,0,width, 125))
    pygame.draw.rect(blue_image, BLUE, blue_image.get_rect(), 10)
    window.blit(blue_image, (0, 0))


    blank = np.zeros(img.shape[:2], dtype="uint8")
    canny_image = canny(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            pygame.draw.rect(window, (255,0,255), (x1,y1,w,h),3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            # print(box.cls)
            # cv2.putText(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255,0,0), thickness=2 )
            font = pygame.font.Font('Resources/Marcellus-llzw.ttf', 24)
            text = font.render(f'{classNames[cls]} {conf}', True, (255, 0, 0))
            window.blit(text, (x1, max(0, y1-35)))

            load_image = np.copy(img)
            canny_image = canny(load_image)    
            z = x1+int(w/2)
            y2 = y2-100
            polygons = np.array([[(x1, y2),(z, y1),(x2,y2)]])
            points1 = [x1, y2, z, y1]
            points2 = [z, y1, x2, y2]
            points3 = [z, y2, z, y1+int(h/2)]
            rect_man.x += speed_x

            l1 = pygame.draw.line(window,(0,255,0),(points1[0],points1[1]),(points1[2],points1[3]),10)
            l2 = pygame.draw.line(window,(0,255,0),(points2[0],points2[1]),(points2[2],points2[3]),10)
            sensor1 = pygame.draw.line(window,(255,0,0),(points1[0]+50,500),(points1[0]+50,600),10)
            sensor2 = pygame.draw.line(window,(255,0,0),(points2[2]-50,500),(points2[2]-50,600),10)
            
            if rect_man.left <= 0 or rect_man.right >= width:
                speed_x *= -1
                sound = pygame.mixer.Sound("danger.wav")
                sound.play()
            if rect_man.colliderect(sensor1):
                speed_x *= -1
                pygame.draw.line(window,(255,0,0),(points1[0],points1[1]),(points1[2],points1[3]),10)
                font = pygame.font.Font('Resources/Marcellus-llzw.ttf', 50)
                text = font.render(f'Go Right Hand', True, (255, 255, 255))
                window.blit(text, (50, 35))
                sound = pygame.mixer.Sound("right.wav")
                sound.play()
            if rect_man.colliderect(sensor2):
                speed_x *= -1
                pygame.draw.line(window,(255,0,0),(points2[0],points2[1]),(points2[2],points2[3]),10)
                font = pygame.font.Font('Resources/Marcellus-llzw.ttf', 50)
                text = font.render(f'Go Left Hand', True, (255, 255, 255))
                window.blit(text, (width/2+250, 35))
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
                lines = cv2.HoughLinesP(masked_image, 1, np.pi/180, 200, np.array([]), minLineLength=200, maxLineGap=200)
                line_image = display_lines(img, lines, sensor1, sensor2, rect_man)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update Display
    pygame.display.update()
    # Set FPS
    clock.tick(fps)
