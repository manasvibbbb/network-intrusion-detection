# src/models/train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

print("="*60)
print("TRAINING NETWORK INTRUSION DETECTION MODELS")
print("="*60)

# Load processed data
print("\n1. Loading processed data...")
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')
print(f"✅ Loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")

# MODEL 1: Random Forest
print("\n" + "="*60)
print("2. TRAINING RANDOM FOREST")
print("="*60)
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

print(f"\n✅ Random Forest trained in {rf_time:.2f} seconds")

# Evaluate Random Forest
print("\n📊 Evaluating Random Forest...")
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_pred_proba_rf)

print(f"Accuracy: {rf_accuracy*100:.2f}%")
print(f"ROC-AUC: {rf_roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Attack']))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('data/Random_Forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved: data/Random_Forest_confusion_matrix.png")

# Save Random Forest model
joblib.dump(rf_model, 'models/random_forest.pkl')
print("✅ Model saved: models/random_forest.pkl")

# MODEL 2: XGBoost
print("\n" + "="*60)
print("3. TRAINING XGBOOST")
print("="*60)
start_time = time.time()

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train, verbose=False)
xgb_time = time.time() - start_time

print(f"\n✅ XGBoost trained in {xgb_time:.2f} seconds")

# Evaluate XGBoost
print("\n📊 Evaluating XGBoost...")
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_roc_auc = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"Accuracy: {xgb_accuracy*100:.2f}%")
print(f"ROC-AUC: {xgb_roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Normal', 'Attack']))

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix - XGBoost')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('data/XGBoost_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved: data/XGBoost_confusion_matrix.png")

# Save XGBoost model
joblib.dump(xgb_model, 'models/xgboost.pkl')
print("✅ Model saved: models/xgboost.pkl")

# MODEL 3: Neural Network
print("\n" + "="*60)
print("4. TRAINING NEURAL NETWORK")
print("="*60)
start_time = time.time()

# Build model - FIXED: Using X_train.shape[1] for input dimension
nn_model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train
history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)

nn_time = time.time() - start_time
print(f"\n✅ Neural Network trained in {nn_time:.2f} seconds")

# Evaluate Neural Network
print("\n📊 Evaluating Neural Network...")
y_pred_proba_nn = nn_model.predict(X_test, verbose=0).flatten()
y_pred_nn = (y_pred_proba_nn > 0.5).astype(int)

nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_roc_auc = roc_auc_score(y_test, y_pred_proba_nn)

print(f"Accuracy: {nn_accuracy*100:.2f}%")
print(f"ROC-AUC: {nn_roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn, target_names=['Normal', 'Attack']))

# Confusion Matrix
cm_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix - Neural Network')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('data/Neural_Network_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved: data/Neural_Network_confusion_matrix.png")

# Save Neural Network model
nn_model.save('models/neural_network.h5')
print("✅ Model saved: models/neural_network.h5")

# ROC Curves Comparison
print("\n" + "="*60)
print("5. GENERATING ROC CURVES COMPARISON")
print("="*60)

plt.figure(figsize=(10, 8))

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_roc_auc:.4f})', linewidth=2)

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_roc_auc:.4f})', linewidth=2)

# Neural Network ROC
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_proba_nn)
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {nn_roc_auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('data/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
print("✅ ROC curves saved: data/roc_curves_comparison.png")

# Model Comparison Summary
print("\n" + "="*60)
print("6. MODEL COMPARISON SUMMARY")
print("="*60)

results_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
    'Accuracy': [rf_accuracy, xgb_accuracy, nn_accuracy],
    'ROC-AUC': [rf_roc_auc, xgb_roc_auc, nn_roc_auc],
    'Training Time (s)': [rf_time, xgb_time, nn_time]
})

results_df.to_csv('data/model_comparison.csv', index=False)
print("\n" + results_df.to_string(index=False))
print("\n✅ Results saved: data/model_comparison.csv")

print("\n" + "="*60)
print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)
print("\n✅ Next step: streamlit run src/dashboard/app.py")