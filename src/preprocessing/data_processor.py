# src/preprocessing/data_processor.py - WORKING VERSION
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

print("="*60)
print("STARTING DATA PREPROCESSING")
print("="*60)

# Column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Step 1: Load data
print("\n1. Loading data...")
train_df = pd.read_csv('data/raw/KDDTrain+.txt', header=None, names=columns)
test_df = pd.read_csv('data/raw/KDDTest+.txt', header=None, names=columns)
print(f"✅ Training samples: {len(train_df)}")
print(f"✅ Testing samples: {len(test_df)}")

# Step 2: Create binary labels
print("\n2. Creating binary labels...")
train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
print(f"✅ Normal: {sum(train_df['binary_label']==0)}, Attack: {sum(train_df['binary_label']==1)}")

# Step 3: Drop unnecessary columns
print("\n3. Dropping unnecessary columns...")
train_df = train_df.drop(columns=['difficulty'])
test_df = test_df.drop(columns=['difficulty'])

# Step 4: Encode categorical features
print("\n4. Encoding categorical features...")
label_encoders = {}
for col in ['protocol_type', 'service', 'flag']:
    label_encoders[col] = LabelEncoder()
    train_df[col] = label_encoders[col].fit_transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])
print("✅ Encoded: protocol_type, service, flag")

# Step 5: Separate features and labels
print("\n5. Separating features and labels...")
X_train = train_df.drop(columns=['label', 'binary_label']).values
y_train = train_df['binary_label'].values
X_test = test_df.drop(columns=['label', 'binary_label']).values
y_test = test_df['binary_label'].values
print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape: {X_test.shape}")

# Step 6: Scale features
print("\n6. Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✅ Features scaled")

# Step 7: Save processed data
print("\n7. Saving processed data...")
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)

joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE!")
print("="*60)
print(f"\nSaved files:")
print(f"  - data/processed/X_train.npy ({X_train.shape})")
print(f"  - data/processed/X_test.npy ({X_test.shape})")
print(f"  - data/processed/y_train.npy ({y_train.shape})")
print(f"  - data/processed/y_test.npy ({y_test.shape})")
print(f"  - models/label_encoders.pkl")
print(f"  - models/scaler.pkl")
print("\n✅ Next step: python src/models/train.py")
