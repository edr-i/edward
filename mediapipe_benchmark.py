import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import sys
import json

# Download the hand landmarker model if not present
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Downloaded.")

def get_finger_states(landmarks):
    tips = [4, 8, 12, 16, 20]
    pip  = [3, 6, 10, 14, 18]
    fingers = []

    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip, p in zip(tips[1:], pip[1:]):
        fingers.append(1 if landmarks[tip].y < landmarks[p].y else 0)

    return fingers

def classify_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]: return "fist"
    if fingers == [1, 1, 1, 1, 1]: return "open_palm"
    if fingers == [1, 0, 0, 0, 0]: return "thumbs_up"
    if fingers == [0, 1, 1, 0, 0]: return "peace"
    if fingers == [0, 1, 0, 0, 0]: return "pointing"
    return "unknown"

def get_finger_states(landmarks, handedness):
    tips = [4, 8, 12, 16, 20]
    pip  = [3, 6, 10, 14, 18]
    fingers = []

    # For right hand thumb points right (larger x = up), left hand opposite
    if handedness == "Right":
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)

    for tip, p in zip(tips[1:], pip[1:]):
        fingers.append(1 if landmarks[tip].y < landmarks[p].y else 0)

    return fingers

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)


with vision.HandLandmarker.create_from_options(options) as detector:
    image_path = sys.argv[1]
    image = mp.Image.create_from_file(image_path)
    result = detector.detect(image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        handedness = result.handedness[0][0].display_name
        fingers = get_finger_states(landmarks, handedness)
        gesture = classify_gesture(fingers)
        print(json.dumps({"gesture": gesture, "raw": str(fingers)}))
    else:
        print(json.dumps({"gesture": "no_hand", "raw": ""}))
