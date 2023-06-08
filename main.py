import cv2 as cv
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video feed
cap = cv.VideoCapture(0)

#setting up the mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #Recoloring to RGB to pass img to mediapipe
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False #image is no longer writeable

        results = pose.process(image)

        #Recoloring back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass

        #Rendering detections onto the feed
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv.imshow('Video Feed', image)
        if cv.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv.destroyAllWindows()

