"""
Quick verification script for prediction fix
Checks that dataset and models are correctly configured
"""
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv

# Load environment
env_path = backend_dir / ".env"
load_dotenv(env_path)

print("=" * 60)
print("PREDICTION FIX VERIFICATION")
print("=" * 60)

# Check environment variables
models_dir = os.getenv("MODELS_DIR", "")
dataset_path = os.getenv("DATASET_PATH", "")

print("\n[*] Configuration:")
print(f"   MODELS_DIR: {models_dir}")
print(f"   DATASET_PATH: {dataset_path}")

# Check if paths exist
models_path = Path(models_dir)
dataset_file = Path(dataset_path)

print("\n[*] Path Verification:")
if models_path.exists():
    print(f"   [OK] Models directory exists: {models_path}")
    model_files = list(models_path.glob("*.joblib")) + list(models_path.glob("metadata.json"))
    print(f"   [OK] Found {len(model_files)} model files")
    for f in model_files[:5]:  # Show first 5
        print(f"      - {f.name}")
else:
    print(f"   [FAIL] Models directory NOT FOUND: {models_path}")

if dataset_file.exists():
    print(f"   [OK] Dataset file exists: {dataset_file}")
    import pandas as pd
    df = pd.read_csv(dataset_file, nrows=5)
    print(f"   [OK] Dataset has {len(df.columns)} columns")
    print(f"   [OK] Sample columns: {list(df.columns[:10])}")
else:
    print(f"   [FAIL] Dataset file NOT FOUND: {dataset_file}")

print("\n[*] Expected Configuration:")
print("   MODELS_DIR should be: backend/data/prod-models/models")
print("   DATASET_PATH should be: backend/data/game_features_20251213.csv")

print("\n[*] If paths are incorrect, update backend/.env file")
print("=" * 60)
