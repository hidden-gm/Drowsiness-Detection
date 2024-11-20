from scipy.spatial import distance
from pygame import mixer
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

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
flag = 0
frame_count = 0
line_y = None
below_line_count = 0  # Counter for frames where the dot is below the line

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (450, 450))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        if len(eyes) >= 2:  # Ensure at least two eyes are detected
            for (ex, ey, ew, eh) in eyes[:2]:  # Use the first two detected eyes
                eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                if eye_center[0] < x + w // 2:
                    left_eye = (x + ex, y + ey, ew, eh)
                else:
                    right_eye = (x + ex, y + ey, ew, eh)
        
            left_eye_points = [(left_eye[0], left_eye[1]), (left_eye[0] + left_eye[2], left_eye[1] + left_eye[3])]
            right_eye_points = [(right_eye[0], right_eye[1]), (right_eye[0] + right_eye[2], right_eye[1] + right_eye[3])]

            leftEAR = eye_aspect_ratio(left_eye_points)
            rightEAR = eye_aspect_ratio(right_eye_points)
            ear = (leftEAR + rightEAR) / 2.0

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

            # Forehead point calculation
            nose_bridge_top = (x + w // 2, y + int(h / 4))
            forehead_point = (nose_bridge_top[0], nose_bridge_top[1] - 20)

            cv2.circle(frame, forehead_point, 3, (255, 0, 0), -1)

            pixels_for_1_cm = 38

            if frame_count < 5:
                line_y = forehead_point[1] + pixels_for_1_cm

            if line_y is not None:
                cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

            if forehead_point[1] > line_y:
                below_line_count += 1
            else:
                below_line_count = 0

            if below_line_count >= 20 and not ambulance_channel.get_busy():
                mixer.music.stop()
                ambulance_channel.play(ambulance_sound)

    frame_count += 1
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
