import cv2 as cv
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video feed
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    cv.imshow('Mediapipe Feed', frame)
    if cv.waitKey(10) & 0xFF == ord('x'):
        break

cap.release()
cv.destroyAllWindows()

