import cv2
import dlib
from imutils import face_utils
import argparse

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image_path = "eyeclosed.jpg"  # Change this to the path of your image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
rects = detector(gray, 1)

# Loop over the face detections
for (i, rect) in enumerate(rects):
    # Determine the facial landmarks for the face region, then convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
