import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "svm_pipeline.pkl")


# LOAD TRAINED PIPELINE

pipeline = joblib.load(PIPELINE_PATH)

svm = pipeline["model"]
scaler = pipeline["scaler"]
pca = pipeline["pca"]
threshold = pipeline["threshold"]


cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

print("MobileNetV2 loaded successfully")


# CLASS LABELS (MATCH TRAINING FOLDERS)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
    "unknown"
]

UNKNOWN_CLASS_ID = 6


# IMAGE PREPROCESSING

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32)
    frame = preprocess_input(frame)
    return frame


# FEATURE EXTRACTION

def extract_features(frame):
    frame = np.expand_dims(frame, axis=0)  # (1,224,224,3)
    features = cnn.predict(frame, verbose=0)
    return features.flatten().reshape(1, -1)


# PREDICTION WITH REJECTION

def predict(frame):
    processed = preprocess_frame(frame)
    features = extract_features(processed)

    features = scaler.transform(features)
    features = pca.transform(features)

    probs = svm.predict_proba(features)
    max_prob = np.max(probs)
    pred = svm.predict(features)[0]

    if max_prob < threshold:
        return "Unknown", max_prob

    return CLASS_NAMES[int(pred)], max_prob


# LIVE CAMERA LOOP

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, confidence = predict(frame)

    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)

    cv2.putText(
        frame,
        f"Class: {label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Waste Classification System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
