import cv2
import pyglet.media
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
import csv
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
import os

# Check if model and sound file exist
if not os.path.exists("eye_detector.keras"):
    print("Error: Model file 'eye_detector.keras' not found.")
    exit()

if not os.path.exists("alarm.wav"):
    print("Error: Alarm sound file 'alarm.wav' not found.")
    exit()

# Set up the camera
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

# Check if camera opened successfully
if not cap.isOpened():
    print("Camera couldn't access!")
    exit()

# Initialize face detector and face mesh detector
face_detector = FaceDetector()
face_mesh_detector = FaceMeshDetector(maxFaces=1)

# Load the eye detection model
best_model = load_model('eye_detector.keras')

# Initialize variables for drowsiness detection
breakcount_s, breakcount_y = 0, 0
counter_s, counter_y = 0, 0
state_s, state_y = False, False

# Load alert sound once outside the loop
sound = pyglet.media.load("alarm.wav", streaming=False)

# Counter for eye closure duration
eye_closed_frames = 0
threshold_closed_frames = 60  # Adjust based on desired sensitivity

# Head movement thresholds
pitch_threshold = 20  # Head tilting forward
yaw_threshold = 25    # Head turning too far left or right

def alert():
    sound.play()
    # Alert overlay
    cv2.rectangle(img, (700, 20), (1250, 80), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, "DROWSINESS ALERT!!!", (710, 60),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

def recordData(condition):
    with open("database.csv", "a", newline="") as file:
        now = datetime.now()
        dtString = now.strftime("%d-%m-%Y %H:%M:%S")
        writer = csv.writer(file)
        writer.writerow((dtString, condition))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Face tracking/head tracking
    img, bboxs = face_detector.findFaces(img, draw=False)
    if bboxs:
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
        pos = [fx, fy]

        # Draw target indicators
        cv2.putText(img, f'Position (x,y): {pos}', (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    else:
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)  # x line
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)  # y line

    img, faces = face_mesh_detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        mouth = [11, 16, 57, 287]  # up, down, left, right
        faceId = [11, 16, 57, 287]

        mouth_ver, _ = face_mesh_detector.findDistance(face[mouth[0]], face[mouth[1]])
        mouth_hor, _ = face_mesh_detector.findDistance(face[mouth[2]], face[mouth[3]])
        mouth_ratio = int((mouth_ver / mouth_hor) * 100)

        # Display text on image
        cv2.putText(img, f'Mouth Ratio: {mouth_ratio}', (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, f'Yawn Count: {counter_y}', (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        if mouth_ratio > 60:
            breakcount_y += 1
            if breakcount_y >= 30:
                alert()
                if not state_y:
                    counter_y += 1
                    sound.play()
                    recordData("Yawn")
                    state_y = not state_y
        else:
            breakcount_y = 0
            if state_y:
                state_y = not state_y

        for id in faceId:
            cv2.circle(img, face[id], 5, (0, 0, 255), cv2.FILLED)

        # Get the first detected face (since maxFaces=1)
        face = faces[0]
        
        # Coordinates for the left eye (based on typical indices for eye landmarks)
        left_eye_points = [face[i] for i in [145, 159, 153, 154, 155, 133, 144, 163, 7, 33]]
        
        # Get bounding box around the eye
        x_coords = [p[0] for p in left_eye_points]
        y_coords = [p[1] for p in left_eye_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Increase bounding box slightly to include surrounding area
        # Adjust padding relative to face size
        padding = int(0.1 * (x_max - x_min))
        x_min, x_max = max(x_min - padding, 0), min(x_max + padding, img.shape[1])
        y_min, y_max = max(y_min - padding, 0), min(y_max + padding, img.shape[0])
        
        # Crop the eye region
        eye_crop = img[y_min:y_max, x_min:x_max]
        
        # Preprocess eye_crop for the model
        gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_eye = cv2.resize(gray_eye, (64, 64))           # Resize to 64x64
        normalized_eye = resized_eye / 255.0                   # Normalize pixel values
        model_input = np.expand_dims(normalized_eye, axis=(0, -1))  # Reshape to (1, 64, 64, 1)
        
        # Make prediction
        result = best_model.predict(model_input)
        
        # Determine if eye is open or closed
        if result > 0.5:
            label = "Open"
            eye_closed_frames = 0  # Reset if eye is open
        else:
            label = "Closed"
            eye_closed_frames += 1

        if eye_closed_frames >= threshold_closed_frames:
            alert()
            recordData("Drowsiness detected")
            eye_closed_frames = 0  # Reset after alert
        
        # Display the label on the original frame
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw rectangle around detected eye
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Show the resulting frame
    cv2.imshow("Image", img)
    # Check for ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()