import sys
from inference_sdk import InferenceHTTPClient
import json

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6vD9ZojPGRVWl8OTIrNI"  # paste your key here
)

MODEL_ID = "hand-gesture-recognition-ktods/1"

GESTURE_MAP = {
    "fist":      "fist",
    "open_palm": "open_palm",
    "thumbs_up": "thumbs_up",
    "v-sign":    "peace",
    "pointing":  "pointing",
}

image_path = sys.argv[1]
result = CLIENT.infer(image_path, model_id=MODEL_ID)

if result["predictions"]:
    top = max(result["predictions"], key=lambda x: x["confidence"])
    gesture = GESTURE_MAP.get(top["class"], "unknown")
    print(json.dumps({"gesture": gesture, "raw": top["class"], "confidence": round(top["confidence"], 2)}))
else:
    print(json.dumps({"gesture": "no_hand", "raw": ""}))