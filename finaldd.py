from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

mixer.init()
mixer.music.load("music.wav")
ambulance_sound = mixer.Sound("ambulance.mp3")
ambulance_channel = mixer.Channel(1)  # Create a separate channel for the ambulance sound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
frame_count = 0
line_y = None
below_line_count = 0  # Counter for frames where the dot is below the line

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # New code to find the point above the nose and between the eyes
        nose_bridge_top = shape[27]
        
        # Define a point above the nose bridge
        forehead_point = (nose_bridge_top[0], nose_bridge_top[1] - 20)
        
        # Draw a circle at this point
        cv2.circle(frame, forehead_point, 3, (255, 0, 0), -1)
        
        # Calculate the position for the line 1 cm below the forehead point
        pixels_for_1_cm = 38
        
        if frame_count < 5:
            line_y = forehead_point[1] + pixels_for_1_cm
        
        if line_y is not None:
            # Draw the fixed line
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)  # Line color green and thickness 2
        
        # Check if the blue dot goes below the green line
        if forehead_point[1] > line_y:
            below_line_count += 1
        else:
            below_line_count = 0  # Reset counter if the dot goes above the line

        # Play ambulance sound if the dot has been below the line for 20 frames
        if below_line_count >= 20 and not ambulance_channel.get_busy():
            mixer.music.stop()  # Stop any other sound playing
            ambulance_channel.play(ambulance_sound)
        
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    ambulance_channel.stop()  # Stop any other sound playing
                    mixer.music.play()
        else:
            flag = 0
    
    frame_count += 1
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
