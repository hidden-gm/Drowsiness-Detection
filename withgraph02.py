from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt

mixer.init()
music_channel_yawn = mixer.Channel(0)  # Channel for yawning detection

yawn_sound = mixer.Sound("ambulance.mp3")  # Load yawn_alert.wav as a Sound object

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

yawn_thresh = 0.75
yawn_frame_check = 15

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

cap = cv2.VideoCapture(0)
yawn_flag = 0
frame_count = 0
mar_values = []  # List to store MAR values for plotting

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        
        mar_values.append(mar)  # Collect MAR values for plotting

        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
        # Check for yawning
        if mar > yawn_thresh:
            yawn_flag += 1
            if yawn_flag >= yawn_frame_check:
                cv2.putText(frame, "**************YAWN ALERT!*************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not music_channel_yawn.get_busy():
                    music_channel_yawn.play(yawn_sound)
        else:
            yawn_flag = 0
            if music_channel_yawn.get_busy():
                music_channel_yawn.stop()  # Stop the sound if yawning stops
    
    frame_count += 1
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

# Plot MAR values over time
plt.plot(mar_values)
plt.title("Mouth Aspect Ratio (MAR) over time")
plt.xlabel("Time (frames)")
plt.ylabel("MAR value")
plt.show()
