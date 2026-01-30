# test_preprocessing.py
import os
import sys

print("="*60)
print("PREPROCESSING DEBUG TEST")
print("="*60)

# Check current directory
print(f"\n1. Current directory: {os.getcwd()}")

# Check if data files exist
print("\n2. Checking dataset files...")
train_file = 'data/raw/KDDTrain+.txt'
test_file = 'data/raw/KDDTest+.txt'

if os.path.exists(train_file):
    size = os.path.getsize(train_file) / (1024 * 1024)
    print(f"✅ {train_file} exists ({size:.2f} MB)")
else:
    print(f"❌ {train_file} NOT FOUND")
    print("\n   Files in data/raw:")
    if os.path.exists('data/raw'):
        for f in os.listdir('data/raw'):
            print(f"   - {f}")
    else:
        print("   ❌ data/raw directory doesn't exist!")
    sys.exit(1)

if os.path.exists(test_file):
    size = os.path.getsize(test_file) / (1024 * 1024)
    print(f"✅ {test_file} exists ({size:.2f} MB)")
else:
    print(f"❌ {test_file} NOT FOUND")
    sys.exit(1)

# Check packages
print("\n3. Checking packages...")
try:
    import pandas as pd
    print("✅ pandas imported")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✅ numpy imported")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")
    sys.exit(1)

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    print("✅ scikit-learn imported")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")
    sys.exit(1)

# Try loading a few rows
print("\n4. Testing data loading...")
try:
    df = pd.read_csv(train_file, header=None, nrows=10)
    print(f"✅ Successfully loaded 10 rows")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.shape[1]}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL CHECKS PASSED! Ready for preprocessing.")
print("="*60)
print("\nNow run: python src/preprocessing/data_processor.py")
