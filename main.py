import cv2 as cv
import numpy as np
import mediapipe as mp
from angle import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video feed
cap = cv.VideoCapture(0)

#Bicel Curl Counter 
counter = 0
position = None

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
            
            #Getting the landmarks for the left and right arm
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            #Calculating the angle
            angle1 = angle(right_shoulder, right_elbow, right_wrist)
            angle2 = angle(left_shoulder, left_elbow, left_wrist)

            #Drawing the angle onto the feed
            cv.putText(image, str(angle1),tuple(np.multiply(right_elbow, [640,480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv.LINE_AA)
            cv.putText(image, str(angle2),tuple(np.multiply(left_elbow, [640,480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv.LINE_AA)

            if angle1 > 160 and angle2 > 160:
                position = 'down'
            if angle1 < 30 and angle2 < 30 and position == 'down':
                position = 'up'
                counter += 1

        except:
            pass
        
        cv.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        cv.putText(image, 'COUNT', (15,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        cv.putText(image, str(counter), (10,60), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)

        cv.putText(image, 'Position', (300,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        cv.putText(image, position, (300,60), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2, cv.LINE_AA)


        #Rendering detections onto the feed
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv.imshow('Video Feed', image)
        if cv.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv.destroyAllWindows()