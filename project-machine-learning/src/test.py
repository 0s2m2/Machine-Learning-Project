# test.py
import os
import cv2
import numpy as np
import joblib
from collections import Counter
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

test_images_path = os.path.join(PROJECT_ROOT, 'test_images')
bestModel_path = os.path.join(PROJECT_ROOT, 'models', 'svm_pipeline.pkl')

def predict(dataFilePath, bestModelPath):
    
    # Load saved pipeline
    
    pipeline = joblib.load(bestModelPath)
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    le = pipeline.get("label_encoder", None)
    threshold = pipeline.get("threshold", 0.6)
    
    if le is None:
        raise ValueError("Label encoder not found in the saved pipeline.")
    
    
    # Load and preprocess images
    
    image_files = [f for f in os.listdir(dataFilePath)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    images = []
    valid_files = []
    
    for img_name in image_files:
        img_path = os.path.join(dataFilePath, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img.astype(np.float32))
        images.append(img)
        valid_files.append(img_name)
    
    if not images:
        return []  # No valid images found
    
    images = np.array(images)  
    
    
    # Extract CNN features
    feature_extractor = MobileNetV2(weights="imagenet", include_top=False,
                                    pooling="avg", input_shape=(224,224,3))
    
    features = feature_extractor.predict(images, verbose=0)
    
    
    # Scale and PCA
    
    features = scaler.transform(features)
    features = pca.transform(features)
    
    
    # Model prediction with rejection
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        max_prob = np.max(probs, axis=1)
        preds = model.predict(features)
        # Assign -1 for low-confidence predictions
        preds[max_prob < threshold] = -1
    else:
        
        preds = model.predict(features)
        max_prob = None  # skip rejection
    
    # Convert numeric labels to class names
    final_preds = []
    for p in preds:
        if p == -1:
            final_preds.append("Unknown")
        else:
            final_preds.append(le.inverse_transform([p])[0])
    
    class_counts = Counter(final_preds)


    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
    
    return final_preds

