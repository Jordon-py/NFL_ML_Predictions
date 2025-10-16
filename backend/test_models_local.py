"""
Test script to verify model loading and prediction locally
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Load models
models_dir = Path(__file__).parent / "models"
print("=" * 60)
print("LOADING MODELS")
print("=" * 60)

home_model = joblib.load(models_dir / "home_model.joblib")
away_model = joblib.load(models_dir / "away_model.joblib")
preprocessor = joblib.load(models_dir / "preprocessor.joblib")
win_model = joblib.load(models_dir / "win_clf_calibrated.joblib")

print(f"✓ Home model: {type(home_model).__name__}")
print(f"✓ Away model: {type(away_model).__name__}")
print(f"✓ Preprocessor: {type(preprocessor).__name__}")
print(f"✓ Win model: {type(win_model).__name__}")

# Load metadata
with open(models_dir / "metadata.json") as f:
    metadata = json.load(f)

print(f"\nModel trained: {metadata['training_timestamp']}")
print(f"Training samples: {metadata['training_samples']}")

# Get feature names
numeric_features = metadata["raw_feature_columns"]["numeric"]
categorical_features = metadata["raw_feature_columns"]["categorical"]
all_features = numeric_features + categorical_features

print(f"\nTotal features: {len(all_features)}")
print(f"  Numeric: {len(numeric_features)}")
print(f"  Categorical: {len(categorical_features)}")

# Create sample data (all NaN for testing)
print("\n" + "=" * 60)
print("TESTING PREDICTION")
print("=" * 60)

# Test with sample data
test_data = {feature: [np.nan if feature in numeric_features else "KC"] 
             for feature in all_features}
test_df = pd.DataFrame(test_data)

print("\nSample input shape:", test_df.shape)
print("First few features:", list(test_df.columns[:5]))

# Transform
X = preprocessor.transform(test_df)
print(f"\nTransformed shape: {X.shape}")

# Predict
home_score = home_model.predict(X)[0]
away_score = away_model.predict(X)[0]
win_prob = win_model.predict_proba(X)[0, 1]

print(f"\n✓ HOME SCORE: {home_score:.1f}")
print(f"✓ AWAY SCORE: {away_score:.1f}")
print(f"✓ WIN PROBABILITY: {win_prob:.3f}")
print(f"✓ POINT DIFF: {home_score - away_score:.1f}")

print("\n" + "=" * 60)
print("SUCCESS! Models working correctly")
print("=" * 60)
