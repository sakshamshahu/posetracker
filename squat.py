import cv2 as cv
import numpy as np
import mediapipe as mp
from angleCalc import angle


cap = cv.VideoCapture(
    '/run/media/saksham/T7/pexels-ketut-subiyanto-4859747-3840x2160-25fps.mp4')

# x and y are the dimensions of the video
x = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
y = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# can use CAP_PROP_FPS to get the fps of the video
fps = int(cap.get(cv.CAP_PROP_FPS))
out = cv.VideoWriter('sampleoutput.avi', cv.VideoWriter_fourcc(*'XVID'), fps, (x, y), True)

# can use CAP_FFMPEG to open and record video file or stream using ffmpeg library

def rescaleFrame(frame, scale=0.4):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

counter = 0
position = None
# setting up the mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recoloring to RGB to pass img to mediapipe
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False  # image is no longer writeable

        results = pose.process(image)

        # Recoloring back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Getting the landmarks for the left and right arm
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculating the angle
            angle_1 = angle(right_hip, right_knee, right_ankle)
            angle_2 = angle(left_hip, left_knee, left_ankle)

            print(angle_1, angle_2)

            # Drawing the angle onto the feed
            cv.putText(image, str(angle_1), tuple(np.multiply(right_knee, [x, y]).astype(
                int)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 59, 150), 3, cv.LINE_AA)

            cv.putText(image, str(angle_2), tuple(np.multiply(left_knee, [x, y]).astype(
                int)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv.LINE_AA)
            
            if angle_1 > 160 and angle_2 > 160:
                position = 'up'
            if angle_1 < 100 and angle_2 < 100 and position == 'up':
                position = 'down'
                counter += 1

        except:
            pass

        cv.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv.putText(image, 'COUNT', (15, 12), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, str(counter), (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        cv.putText(image, 'Position', (300, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, position, (300, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2, cv.LINE_AA)

        # Rendering detections onto the feed
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
        frame_resized = rescaleFrame(image)

        out.write(image)
        cv.imshow('Squaat', frame_resized)

        if cv.waitKey(fps) & 0xFF == ord('x'):
            break

    cap.release()
    cv.destroyAllWindows()