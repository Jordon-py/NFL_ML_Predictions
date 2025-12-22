"""Test dataset loading and full prediction pipeline."""
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib

BASE = Path(r"c:\Users\goku\Documents\NFL_ML_Predictions\backend\data\prod-models")
DATASET_PATH = BASE / "game_features_20251210.csv"
MODELS_DIR = BASE / "models"

print(f"Dataset path: {DATASET_PATH}")
print(f"Exists: {DATASET_PATH.exists()}")

# Load dataset
df = pd.read_csv(DATASET_PATH)
print(f"Loaded {len(df)} rows")

# Load metadata
meta = json.load(open(MODELS_DIR / "metadata.json"))
num_cols = meta["raw_feature_columns"]["numeric"]
cat_cols = meta["raw_feature_columns"]["categorical"]
all_cols = num_cols + cat_cols
print(f"Expected features: {len(num_cols)} numeric + {len(cat_cols)} categorical = {len(all_cols)} total")

# Find row for TB vs ATL
h, a, s, w = "TB", "ATL", 2025, 15
mask = (df["season"] == s) & (df["week"] == w) & (df["home_team"] == h) & (df["away_team"] == a)
matches = mask.sum()
print(f"Match TB vs ATL Week 15 2025: {matches} matches")

if matches > 0:
    row = df[mask].iloc[0]

    # Extract features
    X = {}
    for col in all_cols:
        if col in row.index:
            v = row[col]
            X[col] = v if not pd.isna(v) else np.nan
        elif col == "home_team":
            X[col] = h
        elif col == "away_team":
            X[col] = a
        else:
            X[col] = np.nan

    X_df = pd.DataFrame([X])
    print(f"\nExtracted {len(X)} features")
    print(f"Sample values:")
    print(f"  home_prior_pf_avg_3: {X.get('home_prior_pf_avg_3')}")
    print(f"  away_prior_pf_avg_3: {X.get('away_prior_pf_avg_3')}")
    print(f"  home_elo_pre: {X.get('home_elo_pre')}")

    # Count NaN values
    nan_count = sum(1 for v in X.values() if pd.isna(v))
    print(f"  NaN values: {nan_count}/{len(X)}")

    # Load and run model
    print("\nLoading home model...")
    home_model = joblib.load(MODELS_DIR / "home_model.joblib")

    # Align columns to what model expects
    try:
        if hasattr(home_model, "feature_names_in_"):
            expected = list(home_model.feature_names_in_)
            X_aligned = X_df.reindex(columns=expected, fill_value=np.nan)
            pred = home_model.predict(X_aligned)
            print(f"Home score prediction: {pred[0]:.1f}")
        else:
            pred = home_model.predict(X_df)
            print(f"Home score prediction: {pred[0]:.1f}")
    except Exception as e:
        print(f"Prediction failed: {e}")
