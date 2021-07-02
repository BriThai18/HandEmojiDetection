import cv2
import time
import os
import HandDetectionModule as hm

#Dimension size
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
#Size of cam
cap.set(3, wCam)
cap.set(4, hCam)

#Obtain file of images
path = "images"
myList = os.listdir(path)
print(myList)

#Append images to list
overLay = []
for i in myList:
    image = cv2.imread(f'{path}/{i}')
    overLay.append(image)

pTime = 0
detector = hm.handDetector(minDetection=0.75, minTrack=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    #print(landmarkList)

    if len(landmarkList) != 0:
        fingers = []

        #Check condition for thumbs
        if landmarkList[tipIds[0]][1] > landmarkList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #Check condition for 4 fingers
        for i in range(1, 5):
            if landmarkList[tipIds[i]][2] < landmarkList[tipIds[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        print(fingers)
        #totalFinger = fingers.count(1)
        #print(totalFinger)

        #Fist
        if fingers == [0, 0, 0, 0, 0]:
            height, width, channel = overLay[5].shape
            img[0:height, 0:width] = overLay[5]
        #Thumbs up
        elif fingers == [1, 0, 0, 0, 0]:
            height, width, channel = overLay[7].shape
            img[0:height, 0:width] = overLay[7]
        #Index up
        elif fingers == [0, 1, 0, 0, 0]:
            height, width, channel = overLay[0].shape
            img[0:height, 0:width] = overLay[0]
        #Two finger
        elif fingers == [0, 1, 1, 0, 0]:
            height, width, channel = overLay[1].shape
            img[0:height, 0:width] = overLay[1]
        #Three finger
        elif fingers == [0, 1, 1, 1, 0]:
            height, width, channel = overLay[2].shape
            img[0:height, 0:width] = overLay[2]
        #Four finger
        elif fingers == [0, 1, 1, 1, 1]:
            height, width, channel = overLay[3].shape
            img[0:height, 0:width] = overLay[3]
        #All finger
        elif fingers == [1, 1, 1, 1, 1]:
            height, width, channel = overLay[4].shape
            img[0:height, 0:width] = overLay[4]
        #Emoji 1
        elif fingers == [1, 1, 0, 0, 1]:
            height, width, channel = overLay[6].shape
            img[0:height, 0:width] = overLay[6]
        #Emoji 2
        elif fingers == [1, 0, 0, 0, 1]:
            height, width, channel = overLay[8].shape
            img[0:height, 0:width] = overLay[8]
        #Emoji 3
        elif fingers == [0, 0, 1, 1, 1]:
            height, width, channel = overLay[9].shape
            img[0:height, 0:width] = overLay[9]
        #Emoji 4
        elif fingers == [1, 1, 0, 0, 0]:
            height, width, channel = overLay[10].shape
            img[0:height, 0:width] = overLay[10]

    #Display fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)