#https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81

import numpy.core.multiarray
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detectedFaces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in detectedFaces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('img', img)
    key = cv2.waitKey(30) & 0xff
    if key ==27:
        break

cam.release()
