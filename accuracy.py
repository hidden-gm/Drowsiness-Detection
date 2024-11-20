from scipy.spatial import distance
from imutils import face_utils
import dlib
import cv2

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# Threshold values for EAR and MAR
thresh = 0.30
yawn_thresh = 0.75

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Load an image
# Load an image
image_path = r"C:\Users\dm215\Desktop\Projects\drowsiness detection\dataset\not drowsy\8.jpg"  # Change this to the path of your image
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
subjects = detector(gray, 0)

for subject in subjects:
    shape = predictor(gray, subject)
    shape = face_utils.shape_to_np(shape)

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    mar = mouth_aspect_ratio(mouth)

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    mouthHull = cv2.convexHull(mouth)

    cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(image, [mouthHull], -1, (0, 255, 0), 1)

    if ear < thresh:
        cv2.putText(image, "Drowsiness detected by EAR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if mar > yawn_thresh:
        cv2.putText(image, "Drowsiness detected by MAR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Display the image
cv2.imshow("Drowsiness Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
