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

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Load an image
image_path = "sample.jpg"  # Change this to the path of your image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
subjects = detector(gray, 0)

for subject in subjects:
    shape = predictor(gray, subject)
    shape = face_utils.shape_to_np(shape)

    # Define landmarks for the eyes and mouth
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]

    # Determine the position of the nose bridge top (used for placing the dot)
    nose_bridge_top = shape[27]
    
    # Define a point above the nose bridge (forehead point)
    forehead_point = (nose_bridge_top[0], nose_bridge_top[1] - 20)
    
    # Draw a blue dot at this point
    cv2.circle(image, forehead_point, 5, (255, 0, 0), -1)  # Blue dot

    # Calculate the position for the line 1 cm below the forehead point
    pixels_for_1_cm = 38  # Approximate conversion for 1 cm in pixels
    line_y = forehead_point[1] + pixels_for_1_cm
    
    # Draw the threshold line
    cv2.line(image, (0, line_y), (image.shape[1], line_y), (0, 255, 0), 2)  # Green line

# Display the image with the blue dot and threshold line
cv2.imshow("Forehead Dot and Threshold Line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
