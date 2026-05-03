import sys
import urllib.request
import os
import cv2
import numpy as np
import json

# Download model if not present
model_path = "gesture_recognizer.task"
if not os.path.exists(model_path):
    print("Downloading gesture recognizer model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        model_path
    )
    print("Downloaded.")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=1)

# Map TFLite model's gesture names to yours
GESTURE_MAP = {
    "Closed_Fist":    "fist",
    "Open_Palm":      "open_palm",
    "Thumb_Up":       "thumbs_up",
    "Victory":        "peace",
    "Pointing_Up":    "pointing",
}

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    import mediapipe as mp
    image_path = sys.argv[1]
    image = mp.Image.create_from_file(image_path)
    result = recognizer.recognize(image)

    if result.gestures:
        top = result.gestures[0][0]
        gesture = GESTURE_MAP.get(top.category_name, "unknown")
        print(json.dumps({"gesture": gesture, "raw": top.category_name, "confidence": round(top.score, 2)}))
    else:
        print(json.dumps({"gesture": "no_hand", "raw": ""}))