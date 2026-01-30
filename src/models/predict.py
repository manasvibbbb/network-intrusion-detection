# src/models/predict.py
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
import sys

print("="*60)
print("NETWORK INTRUSION DETECTION - PREDICTION")
print("="*60)

# Load models
print("\n📦 Loading trained models...")
try:
    rf_model = joblib.load('models/random_forest.pkl')
    print("✅ Random Forest loaded")
    
    xgb_model = joblib.load('models/xgboost.pkl')
    print("✅ XGBoost loaded")
    
    nn_model = keras.models.load_model('models/neural_network.h5')
    print("✅ Neural Network loaded")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit(1)

# Load test data
print("\n📂 Loading test data...")
try:
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    print(f"✅ Loaded {X_test.shape[0]} test samples")
except Exception as e:
    print(f"❌ Error loading test data: {e}")
    sys.exit(1)

# Make predictions
print("\n" + "="*60)
print("🔍 MAKING PREDICTIONS")
print("="*60)

# Take first 10 samples
sample_size = 10
X_sample = X_test[:sample_size]
y_true = y_test[:sample_size]

print(f"\nPredicting on {sample_size} samples...\n")

# Random Forest predictions
print("1️⃣  Random Forest:")
rf_pred = rf_model.predict(X_sample)
rf_proba = rf_model.predict_proba(X_sample)[:, 1]

for i in range(sample_size):
    label = "Attack" if rf_pred[i] == 1 else "Normal"
    true_label = "Attack" if y_true[i] == 1 else "Normal"
    confidence = rf_proba[i] * 100
    match = "✅" if rf_pred[i] == y_true[i] else "❌"
    print(f"   Sample {i+1}: {label:8} (Confidence: {confidence:5.2f}%) | True: {true_label:8} {match}")

# XGBoost predictions
print("\n2️⃣  XGBoost:")
xgb_pred = xgb_model.predict(X_sample)
xgb_proba = xgb_model.predict_proba(X_sample)[:, 1]

for i in range(sample_size):
    label = "Attack" if xgb_pred[i] == 1 else "Normal"
    true_label = "Attack" if y_true[i] == 1 else "Normal"
    confidence = xgb_proba[i] * 100
    match = "✅" if xgb_pred[i] == y_true[i] else "❌"
    print(f"   Sample {i+1}: {label:8} (Confidence: {confidence:5.2f}%) | True: {true_label:8} {match}")

# Neural Network predictions
print("\n3️⃣  Neural Network:")
nn_proba = nn_model.predict(X_sample, verbose=0).flatten()
nn_pred = (nn_proba > 0.5).astype(int)

for i in range(sample_size):
    label = "Attack" if nn_pred[i] == 1 else "Normal"
    true_label = "Attack" if y_true[i] == 1 else "Normal"
    confidence = nn_proba[i] * 100
    match = "✅" if nn_pred[i] == y_true[i] else "❌"
    print(f"   Sample {i+1}: {label:8} (Confidence: {confidence:5.2f}%) | True: {true_label:8} {match}")

# Accuracy on sample
rf_accuracy = np.mean(rf_pred == y_true) * 100
xgb_accuracy = np.mean(xgb_pred == y_true) * 100
nn_accuracy = np.mean(nn_pred == y_true) * 100

print("\n" + "="*60)
print("📊 SAMPLE ACCURACY")
print("="*60)
print(f"Random Forest:  {rf_accuracy:.2f}%")
print(f"XGBoost:        {xgb_accuracy:.2f}%")
print(f"Neural Network: {nn_accuracy:.2f}%")

print("\n" + "="*60)
print("✅ PREDICTION COMPLETE")
print("="*60)
