import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset_split", "train_aug")
DATA_DIR_VAL = os.path.join(PROJECT_ROOT, "dataset_split", "val")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")

# --------------------------------------------------
# LOAD CNN (FEATURE EXTRACTOR)
# --------------------------------------------------
model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",     # IMPORTANT: global average pooling
    input_shape=(224, 224, 3)
)

print("MobileNetV2 loaded successfully")

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)

    img = preprocess_input(img)
    return img

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------
def extract_cnn_features(base_folder):
    features = []
    labels = []

    classes = sorted(os.listdir(base_folder))

    for cls in classes:
        cls_path = os.path.join(base_folder, cls)
        if not os.path.isdir(cls_path):
            continue

        for img_name in tqdm(os.listdir(cls_path), desc=f"Processing {cls}"):
            img_path = os.path.join(cls_path, img_name)
            img = load_and_preprocess_image(img_path)

            if img is None:
                continue

            img = np.expand_dims(img, axis=0)  # (1,224,224,3)

            feature = model.predict(img, verbose=0)
            feature = feature.flatten()  # (1280,)

            features.append(feature)
            labels.append(cls)

    return np.array(features), np.array(labels)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    X_train, y_train = extract_cnn_features(DATA_DIR)
    # Use the separate validation folder for validation features
    X_val, y_val = extract_cnn_features(DATA_DIR_VAL)

    joblib.dump(X_train, os.path.join(FEATURES_DIR, "cnn_features_train.pkl"))
    joblib.dump(y_train, os.path.join(FEATURES_DIR, "cnn_labels_train.pkl"))
    joblib.dump(X_val, os.path.join(FEATURES_DIR, "cnn_features_val.pkl"))
    joblib.dump(y_val, os.path.join(FEATURES_DIR, "cnn_labels_val.pkl"))

    print("CNN feature shape:", X_train.shape)
    print("Labels shape:", y_train.shape)
    print("Features and labels saved successfully!")
    
