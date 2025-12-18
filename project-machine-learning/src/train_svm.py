import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD FEATURES
# --------------------------------------------------
X_train = joblib.load(os.path.join(FEATURES_DIR, "cnn_features_train.pkl"))
y_labels = joblib.load(os.path.join(FEATURES_DIR, "cnn_labels_train.pkl"))
X_val = joblib.load(os.path.join(FEATURES_DIR, "cnn_features_val.pkl"))
y_val = joblib.load(os.path.join(FEATURES_DIR, "cnn_labels_val.pkl"))

# Encode class labels
le = LabelEncoder()
y_train = le.fit_transform(y_labels)
y_val = le.transform(y_val)

# --------------------------------------------------
# SCALE AND PCA
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)

print(f"PCA Components Retained: {pca.n_components_}")

# --------------------------------------------------
# TRAIN SVM
# --------------------------------------------------
svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True
)

svm.fit(X_train, y_train)

# --------------------------------------------------
# PREDICTION WITH REJECTION
# --------------------------------------------------
def svm_predict_with_rejection(model, X, threshold=0.6):
    probs = model.predict_proba(X)
    max_prob = np.max(probs, axis=1)
    preds = model.predict(X)

    # Assign -1 for low-confidence (unknown) predictions
    preds[max_prob < threshold] = -1
    return preds

# Apply rejection on validation set
threshold = 0.6
y_pred = svm_predict_with_rejection(svm, X_val, threshold)

# Evaluate only confident predictions
mask = y_pred != -1
coverage = np.mean(mask)
print(f"Coverage (fraction of confident predictions): {coverage:.2f}")
print("SVM Accuracy (confident predictions only):", accuracy_score(y_val[mask], y_pred[mask]))
print("\nClassification Report (confident predictions only):\n")
print(classification_report(y_val[mask], y_pred[mask], target_names=le.classes_))

# --------------------------------------------------
# SAVE MODEL AND PIPELINE
# --------------------------------------------------
joblib.dump(svm, os.path.join(MODELS_DIR, 'svm_model.pkl'))

joblib.dump(
    {
        "model": svm,
        "scaler": scaler,
        "pca": pca,
        "label_encoder": le,
        "threshold": threshold
    },
    os.path.join(MODELS_DIR, "svm_pipeline.pkl")
)

print("SVM pipeline saved successfully!")
