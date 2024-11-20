import cv2
import dlib
from imutils import face_utils

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image_path = "eyeclosed.jpg"  # Update this path to your image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
rects = detector(gray, 1)

# Loop over the face detections
for (i, rect) in enumerate(rects):
    # Determine the facial landmarks for the face region, then convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Define the indices for the relevant facial landmarks
    eye_and_mouth_indices = list(range(36, 48)) + list(range(48, 60))

    # Loop over the relevant (x, y)-coordinates for the facial landmarks and draw them on the image
    for (j, (x, y)) in enumerate(shape):
        if j in eye_and_mouth_indices:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            if j < 48:  # For eyes, position text above the points
                cv2.putText(image, str(j + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:  # For mouth, position text below the points
                cv2.putText(image, str(j + 1), (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()